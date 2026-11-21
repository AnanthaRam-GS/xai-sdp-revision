"""The permuted-label refit must reproduce the step 6 data-randomisation results."""

from __future__ import annotations

from pathlib import Path

import pytest

from sdp.config import REPO_ROOT, load_config
from sdp.stability import permuted as P

CFG = load_config()
STEP6 = Path(CFG.paths.artifacts_dir) / "fidelity" / "sanity"
SPLIT = "ta-consecutive:xerces:init>1.2"


@pytest.mark.skipif(not STEP6.exists(), reason="needs the step 6 artefacts")
@pytest.mark.parametrize("model", ["lr", "rf"])
def test_refit_reproduces_step6(model, tmp_path, monkeypatch):
    from sdp.explain.runner import experiment_splits, load_explain_settings

    es = load_explain_settings(REPO_ROOT / "configs" / "explain.yaml")
    splits, frames = experiment_splits(CFG, es)
    split = next(s for s in splits if s.split_id == SPLIT)
    saved = {}
    monkeypatch.setattr(P, "save_artifact", lambda path, pre, mdl, meta: saved.setdefault("meta", meta))
    monkeypatch.setattr(P, "write_long", lambda frame, path: saved.setdefault("subset", set(frame["subset"])))
    res = P.permuted_job(CFG, es, split, frames[split.project], model)
    import pandas as pd

    check = P.reproduction_check(pd.DataFrame(res["rows"]), P.step6_rows(STEP6.parent))
    assert len(check) == len(res["rows"]) > 0
    assert check["diff_permuted_auc"].max() < 1e-12
    assert check["diff_spearman"].max() < 1e-9 and check["diff_fa5"].max() < 1e-9
    assert saved["subset"] == {"permuted_labels"} and saved["meta"]["experiment"] == "permuted_labels"
