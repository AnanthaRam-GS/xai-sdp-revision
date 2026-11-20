"""Multi-layer perceptron in PyTorch with a scikit-learn style wrapper.

Architecture constraints (required by Captum's DeepLift / DeepLiftShap in step 4):

* the network is built ONLY from ``nn.Linear``, ``nn.ReLU(inplace=False)`` and
  ``nn.Dropout`` modules, stored in a ``ModuleList`` and applied in order in
  ``forward``;
* no functional activations, no in-place operations, no BatchNorm, and no module is
  used twice in the forward pass.

The network outputs a single logit of shape ``(N, 1)``. :class:`ProbabilityModel` wraps
it and appends ``nn.Sigmoid()`` for the explainers.

Training (:class:`MLPClassifierTorch`): mini-batch Adam with weight decay,
``BCEWithLogitsLoss`` (with ``pos_weight = n_neg / n_pos`` when
``imbalance="class_weight"``), early stopping on the validation loss (patience 20,
at most 500 epochs) and restoration of the best weights. Training is CPU-only and
deterministic for a given ``random_state``.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn

logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")  # deliberately CPU only (no MPS/CUDA) for determinism


class MLPNet(nn.Module):
    """Linear → ReLU → Dropout blocks followed by a linear output layer (one logit)."""

    def __init__(self, n_features: int, hidden_layers: int = 2, units: int = 64, dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(width, int(units)))
            layers.append(nn.ReLU(inplace=False))
            if dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            width = int(units)
        layers.append(nn.Linear(width, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ProbabilityModel(nn.Module):
    """Wraps an :class:`MLPNet` and outputs P(defective) of shape ``(N, 1)``."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.net(x))


def configure_torch(seed: int, threads: int = 1) -> None:
    """Seed torch and make it deterministic and single-threaded (CPU)."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)


def _as_array(X) -> np.ndarray:
    return np.asarray(X.to_numpy() if isinstance(X, pd.DataFrame) else X, dtype=np.float32)


class MLPClassifierTorch(ClassifierMixin, BaseEstimator):
    """Scikit-learn compatible binary classifier around :class:`MLPNet`.

    If ``fit`` is called without validation data (e.g. inside inner cross-validation),
    a stratified ``internal_val_fraction`` of the given training data is held out for
    early stopping.
    """

    def __init__(
        self,
        hidden_layers: int = 2,
        units: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 500,
        patience: int = 20,
        imbalance: str = "class_weight",
        internal_val_fraction: float = 0.2,
        random_state: int = 0,
    ):
        self.hidden_layers = hidden_layers
        self.units = units
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.imbalance = imbalance
        self.internal_val_fraction = internal_val_fraction
        self.random_state = random_state

    # ------------------------------------------------------------------ training

    def fit(self, X, y, X_val=None, y_val=None):
        t0 = time.perf_counter()
        configure_torch(self.random_state, threads=torch.get_num_threads())
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        Xa, ya = _as_array(X), np.asarray(y, dtype=np.float32)
        if X_val is None:
            strat = ya if np.bincount(ya.astype(int), minlength=2).min() >= 2 else None
            Xa, Xv, ya, yv = train_test_split(
                Xa, ya, test_size=self.internal_val_fraction, random_state=self.random_state, stratify=strat
            )
            self.internal_validation_ = True
        else:
            Xv, yv = _as_array(X_val), np.asarray(y_val, dtype=np.float32)
            self.internal_validation_ = False

        self.n_features_in_ = Xa.shape[1]
        self.classes_ = np.array([0, 1])
        n_pos = float(ya.sum())
        n_neg = float(len(ya) - n_pos)
        pos_weight = None
        if self.imbalance == "class_weight" and n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        net = MLPNet(self.n_features_in_, self.hidden_layers, self.units, self.dropout).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xt, yt = torch.from_numpy(Xa), torch.from_numpy(ya).unsqueeze(1)
        Xvt, yvt = torch.from_numpy(Xv), torch.from_numpy(yv).unsqueeze(1)
        gen = torch.Generator().manual_seed(int(self.random_state))
        bs = int(self.batch_size)

        best_loss, best_state, best_epoch, wait = np.inf, copy.deepcopy(net.state_dict()), 0, 0
        epoch = 0
        for epoch in range(1, int(self.max_epochs) + 1):
            net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                loss = loss_fn(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(Xvt), yvt))
            if val_loss < best_loss - 1e-7:
                best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(net.state_dict()), epoch, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.best_epoch_ = best_epoch
        self.n_epochs_ = epoch
        self.best_val_loss_ = float(best_loss)
        self.fit_time_s_ = time.perf_counter() - t0
        return self

    # ---------------------------------------------------------------- inference

    def predict_proba(self, X) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.from_numpy(_as_array(X)))).squeeze(1).numpy().astype(np.float64)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_torch_module(self) -> MLPNet:
        """The trained network (logit output), in evaluation mode."""
        self.net_.eval()
        return self.net_

    def get_probability_module(self) -> ProbabilityModel:
        """The trained network followed by ``nn.Sigmoid()``, in evaluation mode."""
        return ProbabilityModel(self.get_torch_module()).eval()

    # -------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Save weights (``mlp.pt``) and constructor parameters (``mlp.json``) to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net_.state_dict(), path / "mlp.pt")
        meta = {
            "params": self.get_params(),
            "n_features_in": int(self.n_features_in_),
            "feature_names_in": [str(c) for c in getattr(self, "feature_names_in_", [])],
            "best_epoch": int(self.best_epoch_),
            "n_epochs": int(self.n_epochs_),
            "best_val_loss": self.best_val_loss_,
        }
        (path / "mlp.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MLPClassifierTorch":
        path = Path(path)
        meta = json.loads((path / "mlp.json").read_text(encoding="utf-8"))
        model = cls(**meta["params"])
        net = MLPNet(meta["n_features_in"], model.hidden_layers, model.units, model.dropout)
        net.load_state_dict(torch.load(path / "mlp.pt", map_location=DEVICE, weights_only=True))
        net.eval()
        model.net_ = net
        model.n_features_in_ = meta["n_features_in"]
        if meta["feature_names_in"]:
            model.feature_names_in_ = np.asarray(meta["feature_names_in"], dtype=object)
        model.classes_ = np.array([0, 1])
        model.best_epoch_ = meta["best_epoch"]
        model.n_epochs_ = meta["n_epochs"]
        model.best_val_loss_ = meta["best_val_loss"]
        return model
