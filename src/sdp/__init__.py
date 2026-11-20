"""Explainable software defect prediction: experimental pipeline."""

# XGBoost and PyTorch each ship their own OpenMP runtime (libomp). On macOS, loading
# PyTorch's first makes XGBoost crash with a segmentation fault, so XGBoost is always
# imported first, before anything in this package can import torch.
import xgboost  # noqa: F401  (import order matters; see above)

__version__ = "0.3.0"
