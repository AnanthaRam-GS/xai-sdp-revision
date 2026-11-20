"""PyTorch and XGBoost must work in the same process (single OpenMP runtime)."""

import sys

import numpy as np
import pandas as pd
import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific")
def test_single_openmp_runtime_is_loaded():
    import threadpoolctl
    import torch  # noqa: F401
    import xgboost  # noqa: F401
    from sklearn import ensemble  # noqa: F401

    runtimes = {d["filepath"] for d in threadpoolctl.threadpool_info() if d["internal_api"] == "openmp"}
    assert len(runtimes) == 1


def test_torch_and_xgboost_in_one_process():
    from sdp.models.registry import make_model

    X = pd.DataFrame(np.random.default_rng(0).random((120, 3)), columns=list("abc"))
    y = (X["a"] > 0.5).astype(int).to_numpy()
    for name in ("mlp", "xgb", "mlp", "rf", "xgb"):
        params = {"max_epochs": 5} if name == "mlp" else {"n_estimators": 10}
        make_model(name, params).fit(X, y)
