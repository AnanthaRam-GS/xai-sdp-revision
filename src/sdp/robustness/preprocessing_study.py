"""Explanations of the same models under different preprocessing (step 5).

The persisted ``ablation_preprocessing`` models (MLP and LR × {none, standard,
log_standard} × 10 seeds × 24 time-aware splits) are explained with the step 4 runner:

* MLP: DeepLiftShap (step 4 background, built in each model's own input space) and
  DeepLift (baseline = background mean); for preprocessing ``none`` also DeepLift
  with an all-zero baseline (``DeepLift_zero_baseline``), i.e. the original paper's
  setting of unscaled inputs and a zero reference;
* LR: LinearSHAP (log-odds).

Each preprocessing setting has its own store,
``<artifacts>/attributions_ablation_preprocessing/<preprocessing>/``.
"""

from __future__ import annotations

import dataclasses
import logging

import pandas as pd

from sdp.analysis import explanations as E
from sdp.analysis.tables import MetricSpec, all_pairs_comparisons
from sdp.config import Config
from sdp.explain.runner import ExplainSettings, build_jobs, experiment_splits, run_explanations

logger = logging.getLogger(__name__)

ZERO = "DeepLift_zero_baseline"
DIRECTION_FEATURES = ["loc", "rfc", "amc", "lcom", "cam"]
BY = ("model", "method", "preprocessing")


def prep_settings(es: ExplainSettings, ps: dict, prep: str) -> ExplainSettings:
    """Step 4 settings redirected to the ablation models of one preprocessing setting."""
    methods = {m: list(ps["methods"][m]) for m in ps["models"]}
    if prep == ps.get("zero_baseline"):
        methods["mlp"] = methods["mlp"] + [ZERO]
    return dataclasses.replace(
        es, experiment=ps["experiment"], models={m: prep for m in ps["models"]}, methods=methods,
        expensive_methods=[], output_dir=f"attributions_{ps['experiment']}/{prep}",
    )  # fmt: skip


def run_preprocessing_study(cfg: Config, es: ExplainSettings, rs: dict, n_jobs: int, splits=None, seeds=None) -> list[dict]:
    ps = rs["preprocessing"]
    infos = []
    for prep in ps["settings"]:
        s = prep_settings(es, ps, prep)
        all_splits, frames = experiment_splits(cfg, s)
        use = splits if splits is not None else all_splits
        jobs = build_jobs(cfg, s, frames, use, seeds=seeds)
        infos.append(run_explanations(cfg, s, jobs, frames, n_jobs=n_jobs, tag=f"preprocessing_{prep}") | {"preprocessing": prep})
    return infos


def load_preprocessing_explanations(cfg: Config, es: ExplainSettings, rs: dict) -> E.Explanations:
    ps = rs["preprocessing"]
    parts = []
    for prep in ps["settings"]:
        ex = E.load_explanations(prep_settings(es, ps, prep).root(cfg))
        ex.meta["preprocessing"] = prep
        parts.append(ex)
    return E.Explanations.concat(parts)


def label(model: str, method: str, prep: str) -> str:
    return f"{E.label(model, method)} · {prep}"


def setting_name(method: str, prep: str) -> str:
    """The preprocessing column of the comparisons; the zero baseline is its own setting."""
    return f"{prep} (zero baseline)" if method == ZERO else prep


def comparisons(ex: E.Explanations) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-split loc and size-group shares (seeds averaged) and Wilcoxon signed-rank
    tests with Holm correction and Cliff's delta between preprocessing settings, within
    each explanation family (MLP-DeepLiftShap, MLP-DeepLift incl. the zero baseline,
    LR-LinearSHAP)."""
    norm = E.normalised_abs(ex.attr)
    vals = pd.DataFrame({"loc_share": norm[:, ex.features.index("loc")],
                         "size_group_share": norm[:, [ex.features.index(f) for f in E.SIZE_GROUP]].sum(axis=1)})  # fmt: skip
    per_split = E.per_split_values(ex.meta, vals, BY)
    per_split["family"] = [E.label(m, "DeepLift" if k == ZERO else k) for m, k in zip(per_split["model"], per_split["method"])]
    per_split["setting"] = [setting_name(k, p) for k, p in zip(per_split["method"], per_split["preprocessing"])]
    long = per_split.melt(id_vars=["family", "setting", "project", "split_id"], value_vars=["loc_share", "size_group_share"],
                          var_name="metric", value_name="value").assign(threshold_mode="none", seed=0)  # fmt: skip
    specs = [MetricSpec("loc_share", "none", "loc share"), MetricSpec("size_group_share", "none", "size-group share")]
    tests = all_pairs_comparisons(long, specs, column="setting", within="family")
    return per_split, tests


def direction_table(ex: E.Explanations) -> pd.DataFrame:
    d = E.direction(ex, by=BY)
    d = d[(d["scope"] == "ALL") & d["feature"].isin(DIRECTION_FEATURES)]
    return d.assign(explanation=[label(m, k, p) for m, k, p in zip(d["model"], d["method"], d["preprocessing"])])


def setting_order(pairs) -> list[tuple[str, str, str]]:
    preps = ["none", "standard", "log_standard"]
    methods = ["DeepLiftShap", "DeepLift", ZERO, "LinearSHAP"]
    return sorted(set(pairs), key=lambda t: (E.MODEL_ORDER.index(t[0]), methods.index(t[1]), preps.index(t[2])))


def summary_numbers(per_split: pd.DataFrame) -> pd.DataFrame:
    """Mean over splits of the per-split shares (for figures and quick tables)."""
    out = per_split.groupby(list(BY))[["loc_share", "size_group_share"]].mean().reset_index()
    order = setting_order(map(tuple, out[list(BY)].to_numpy()))
    return out.set_index(list(BY)).loc[order].reset_index()
