"""Pytest configuration.

XGBoost must be imported before PyTorch (see ``sdp/__init__.py``): some test modules
import torch directly, so import XGBoost here, before any test module is collected.
"""

import xgboost  # noqa: F401
