"""Model-agnostic (LIME, KernelSHAP) and model-specific (TreeSHAP, linear SHAP) explainers.

* **LIME** (all models): ``LimeTabularExplainer`` in model input space,
  ``mode="classification"``, ``discretize_continuous=False``,
  ``sample_around_instance=True`` (perturbations are drawn around the explained
  instance, scaled by the training standard deviations), default kernel width,
  ``num_samples=5000``, ``num_features=20``, ``random_state`` = model seed, explaining
  P(defective). Weights are mapped to features through ``explanation.as_map()[1]``
  (feature index → weight), never by list order. The weights are the coefficients
  (slopes) of the local surrogate. ``LIME_mean_sampling`` is the step 4 (v1) setting,
  ``sample_around_instance=False``, which samples around the training mean and
  therefore gives almost the same weights for every instance.
* **KernelSHAP** (MLP only): ``shap.KernelExplainer`` on P(defective) with the
  background set, ``nsamples=1000``. ``l1_reg=False``: no lasso feature selection, so
  that all 20 features get an attribution (shap's default since 0.47 keeps only 10).
* **TreeSHAP** (RF, XGBoost): ``shap.TreeExplainer`` with
  ``feature_perturbation="interventional"``, ``data`` = background set and
  ``model_output="probability"``. XGBoost is explained up to its early-stopping best
  iteration, exactly as it predicts. Additivity: ``sum(SHAP) + E[f] ≈ f(x)``.
  shap's interventional C++ code stores split thresholds as float32, whereas
  scikit-learn compares float32 inputs with float64 thresholds; where rounding a
  threshold to float32 crosses a data value, shap follows the wrong branch and
  additivity fails (errors up to 0.02 on our forests). The forest handed to shap is
  therefore a copy whose thresholds are rounded *down* to float32, and inputs are cast
  to float32 (as scikit-learn and XGBoost do internally); for float32 inputs this
  partitions every node exactly as the original model (see :func:`float32_safe_forest`).
* **Linear SHAP** (LR): exact interventional SHAP on the probability scale is not
  linear, so ``shap.LinearExplainer`` is used on the **log-odds** scale
  (``output_scale="log_odds"``). Additivity is checked against the decision function.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from sdp.explain.base import LOG_ODDS, ExplainContext, Explainer, check_frame

logger = logging.getLogger(__name__)


class LimeExplainer(Explainer):
    """LIME; parameters (all optional): ``num_samples``, ``num_features``,
    ``sample_around_instance`` (default True), ``discretize_continuous`` (default False),
    ``kernel_width`` (default: LIME's 0.75 × sqrt(n_features)) and ``random_state``
    (default: the model seed). The local surrogate's R² (``explanation.score``) of the
    last call is kept in ``scores_``."""

    name = "LIME"
    models = ("mlp", "lr", "rf", "xgb")
    default_sample_around_instance = True

    def _attribute(self, ctx: ExplainContext):
        from lime.lime_tabular import LimeTabularExplainer

        feats = ctx.feature_names
        p = self.params
        explainer = LimeTabularExplainer(
            training_data=ctx.X_train.to_numpy(dtype="float64"),
            feature_names=feats,
            class_names=["clean", "defective"],
            mode="classification",
            discretize_continuous=bool(p.get("discretize_continuous", False)),
            sample_around_instance=bool(p.get("sample_around_instance", self.default_sample_around_instance)),
            kernel_width=p.get("kernel_width"),
            random_state=int(p.get("random_state", ctx.seed)),
        )
        num_features = int(p.get("num_features", len(feats)))
        num_samples = int(p.get("num_samples", 5000))
        out = np.zeros((len(ctx.X_test), len(feats)))
        scores = np.zeros(len(ctx.X_test))
        for i, row in enumerate(ctx.X_test.to_numpy(dtype="float64")):
            exp = explainer.explain_instance(row, ctx.predict_proba, labels=(1,), num_features=num_features,
                                             num_samples=num_samples)  # fmt: skip
            out[i] = lime_weights(exp, len(feats))
            scores[i] = float(np.atleast_1d(exp.score)[0]) if not isinstance(exp.score, dict) else exp.score[1]
        self.scores_ = scores
        return pd.DataFrame(out, columns=feats), None


class LimeMeanSamplingExplainer(LimeExplainer):
    """The step 4 (v1) LIME setting: perturbations around the training mean."""

    name = "LIME_mean_sampling"
    default_sample_around_instance = False


def lime_weights(explanation, n_features: int, label: int = 1) -> np.ndarray:
    """Dense weight vector from ``explanation.as_map()[label]`` (index → weight)."""
    w = np.zeros(n_features)
    for idx, weight in explanation.as_map()[label]:
        w[int(idx)] = weight
    return w


class KernelShapExplainer(Explainer):
    name = "KernelSHAP"
    models = ("mlp",)

    def _attribute(self, ctx: ExplainContext):
        import shap

        np.random.seed(ctx.seed)  # KernelExplainer samples coalitions with NumPy's global generator
        explainer = shap.KernelExplainer(ctx.predict, ctx.background.to_numpy(dtype="float64"))
        values = explainer.shap_values(ctx.X_test.to_numpy(dtype="float64"), nsamples=int(self.params.get("nsamples", 1000)),
                                       l1_reg=self.params.get("l1_reg", False), silent=True)  # fmt: skip
        return np.asarray(values).reshape(len(ctx.X_test), -1), None


def _positive_class(values, expected) -> tuple[np.ndarray, float]:
    """SHAP values and expected value for the defective class, whatever shap returns."""
    values = np.asarray(values)
    expected = np.atleast_1d(np.asarray(expected, dtype=float))
    if values.ndim == 3:  # (n, features, classes) or (classes, n, features)
        values = values[..., 1] if values.shape[-1] == 2 else values[1]
        return values, float(expected[-1])
    return values, float(expected[0])


def _as_float32(frame) -> np.ndarray:
    """Values rounded to float32 (what the tree models see), returned as float64."""
    return np.asarray(frame, dtype="float64").astype(np.float32).astype(np.float64)


def float32_safe_forest(forest):
    """Copy of a scikit-learn forest with every split threshold rounded down to float32.

    For a float32 input ``x`` and float64 threshold ``t``, ``x <= t`` holds exactly when
    ``x <= t32`` with ``t32`` the largest float32 not above ``t``; the copy therefore
    predicts identically on float32 inputs while being exact for shap's float32
    thresholds.
    """
    import copy

    safe = copy.deepcopy(forest)
    for est in safe.estimators_:
        th = est.tree_.threshold  # a writable view of the tree's nodes
        t32 = th.astype(np.float32)
        down = np.where(t32.astype(np.float64) > th, np.nextafter(t32, np.float32(-np.inf)), t32)
        split = est.tree_.children_left >= 0
        th[split] = down.astype(np.float64)[split]
    return safe


class TreeShapExplainer(Explainer):
    name = "TreeSHAP"
    models = ("rf", "xgb")

    @staticmethod
    def tree_model(model):
        """The object handed to shap: XGBoost's booster up to the best iteration, or a
        float32-safe copy of the random forest."""
        if hasattr(model, "model_"):  # XGBClassifierES (thresholds are float32 already)
            booster = model.model_.get_booster()
            best = getattr(model, "best_iteration_", None)
            return booster[: best + 1] if best is not None else booster
        return float32_safe_forest(model)

    def _attribute(self, ctx: ExplainContext):
        import shap

        explainer = shap.TreeExplainer(self.tree_model(ctx.model), data=_as_float32(ctx.background),
                                       feature_perturbation="interventional", model_output="probability")  # fmt: skip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            raw = explainer.shap_values(_as_float32(ctx.X_test), check_additivity=False)
        values, expected = _positive_class(raw, explainer.expected_value)
        checks = check_frame(ctx, self.name, "additivity", ctx.y_prob, expected, values.sum(axis=1))
        return values, checks


class LinearShapExplainer(Explainer):
    name = "LinearSHAP"
    models = ("lr",)
    output_scale = LOG_ODDS

    def _attribute(self, ctx: ExplainContext):
        import shap

        bg = ctx.background.to_numpy(dtype="float64")
        explainer = shap.LinearExplainer(ctx.model, shap.maskers.Independent(bg, max_samples=len(bg)))
        X = ctx.X_test.to_numpy(dtype="float64")
        values = np.asarray(explainer.shap_values(X)).reshape(len(X), -1)
        expected = float(np.atleast_1d(explainer.expected_value)[0])
        logit = ctx.model.decision_function(ctx.X_test)
        checks = check_frame(ctx, self.name, "additivity", logit, expected, values.sum(axis=1))
        return values, checks


OTHER_EXPLAINERS = {
    cls.name: cls
    for cls in (LimeExplainer, LimeMeanSamplingExplainer, KernelShapExplainer, TreeShapExplainer, LinearShapExplainer)
}
