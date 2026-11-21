"""Experiment configurations (``configs/experiments/*.yaml``).

Example::

    name: main_time_aware
    protocol: time_aware            # or within_release
    time_aware_variant: consecutive # or cumulative
    default_imbalance: class_weight
    models:
      mlp: {preprocessing: [log_standard]}
      rf:  {preprocessing: [standard], imbalance: [none, class_weight, smote]}
      manual_up: {}
    seeds: [0, 1, 2]                # learned models; rule-based baselines use seed 0 only
    n_trials: null                  # null = protocol default from configs/default.yaml
    persist_models: true
    n_jobs: 10
    feature_selection: {method: autospearman, threshold: 0.7, vif_threshold: 5}   # optional

Rule-based baselines always use preprocessing ``none`` and imbalance ``none``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from sdp.models.registry import ALL_MODELS, is_learned


@dataclass(frozen=True)
class ModelSetting:
    model: str
    preprocessing: str
    imbalance: str

    @property
    def label(self) -> str:
        return f"{self.model}|{self.preprocessing}|{self.imbalance}"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    protocol: str
    settings: tuple[ModelSetting, ...]
    seeds: tuple[int, ...]
    time_aware_variant: str = "consecutive"
    baseline_seeds: tuple[int, ...] = (0,)
    n_trials: int | None = None
    persist_models: bool = False
    n_jobs: int = 1
    projects: tuple[str, ...] | None = None  # restrict to these projects (smoke tests)
    description: str = ""
    extra: dict = field(default_factory=dict)
    feature_selection: dict | None = None  # e.g. {method: autospearman, threshold: 0.7, vif_threshold: 5}

    def seeds_for(self, model: str) -> tuple[int, ...]:
        return self.seeds if is_learned(model) else self.baseline_seeds

    def smoke(self, project: str, seeds: int = 2, n_trials: int = 5) -> "ExperimentConfig":
        """A small version of this experiment: one project, few seeds and trials."""
        return replace(
            self,
            name=f"smoke_{self.name}",
            projects=(project,),
            seeds=self.seeds[:seeds],
            n_trials=n_trials,
            persist_models=False,
        )


def _as_list(v, default) -> list:
    if v is None:
        return list(default)
    return list(v) if isinstance(v, (list, tuple)) else [v]


def load_experiment(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    default_imb = raw.get("default_imbalance", "class_weight")
    settings = []
    for model, opts in (raw.get("models") or {}).items():
        if model not in ALL_MODELS:
            raise ValueError(f"{path}: unknown model {model!r}")
        opts = opts or {}
        if not is_learned(model):
            settings.append(ModelSetting(model, "none", "none"))
            continue
        for prep in _as_list(opts.get("preprocessing"), ["standard"]):
            for imb in _as_list(opts.get("imbalance"), [default_imb]):
                settings.append(ModelSetting(model, str(prep), str(imb)))
    if raw.get("protocol") not in ("time_aware", "within_release"):
        raise ValueError(f"{path}: protocol must be time_aware or within_release")
    return ExperimentConfig(
        name=raw["name"],
        protocol=raw["protocol"],
        settings=tuple(settings),
        seeds=tuple(int(s) for s in _as_list(raw.get("seeds"), [0])),
        time_aware_variant=raw.get("time_aware_variant", "consecutive"),
        baseline_seeds=tuple(int(s) for s in _as_list(raw.get("baseline_seeds"), [0])),
        n_trials=raw.get("n_trials"),
        persist_models=bool(raw.get("persist_models", False)),
        n_jobs=int(raw.get("n_jobs", 1)),
        description=raw.get("description", ""),
        feature_selection=raw.get("feature_selection"),
    )
