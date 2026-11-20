"""Thin command-line entry points (run as ``python -m scripts.<name>``)."""

# Must run before any library with an OpenMP runtime is imported (see sdp.runtime).
from sdp.runtime import ensure_single_openmp_runtime

ensure_single_openmp_runtime()
