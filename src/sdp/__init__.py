"""Explainable software defect prediction: experimental pipeline.

On macOS, call :func:`sdp.runtime.ensure_single_openmp_runtime` before importing
PyTorch, XGBoost or scikit-learn in a new process (the entry points in ``scripts/``
and the test suite do this); see :mod:`sdp.runtime`.
"""

__version__ = "0.3.0"
