# Environment

- **Python**: 3.11 (recorded run: CPython 3.11.15 on macOS 27, Apple M5, 24 GB RAM, 10 cores).
- **Dependencies**: `requirements.txt` lists the direct dependencies; `requirements.lock` pins all
  58 packages with hashes (generated with `uv pip compile --generate-hashes` against the
  environment used for the study; platform `aarch64-apple-darwin`, macOS ≥ 14 for the PyTorch
  wheel). Create the environment with

  ```bash
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python --require-hashes -r requirements.lock
  uv pip install --python .venv/bin/python --no-deps -e .
  ```

  (`make env` does the same). On other platforms, install `requirements.txt` instead; the
  results may differ in the last digits (floating-point and library builds).
- **Computation**: CPU only, PyTorch deterministic algorithms, one thread per worker; joblib
  `loky` workers (10 by default, set in the configuration files).

## OpenMP on macOS

PyTorch, XGBoost (through Homebrew's `libomp`) and scikit-learn each ship an OpenMP runtime.
Two LLVM OpenMP runtimes in one process crash with a segmentation fault ("OMP: Error #179"),
whatever the import order. `sdp.runtime.ensure_single_openmp_runtime()` therefore re-executes
the process once with `DYLD_LIBRARY_PATH` pointing at PyTorch's `libomp`, so that every library
resolves to the same runtime. Every entry point calls it first (`scripts/__init__.py`), the test
suite calls it from `tests/conftest.py`, and notebooks or new scripts must do the same before
importing torch, xgboost or scikit-learn. It is a no-op on other operating systems.

## R (optional)

The Scott-Knott ESD cross-check (`python -m scripts.run_skesd_crosscheck`) uses `Rscript` with
ScottKnottESD 2.0.3 and 1.2.2 installed into user libraries
(`~/Library/R/sdp-user-lib`, `~/Library/R/sdp-user-lib-skv1`); it is skipped when `Rscript` is
not available. R was never installed or changed system-wide by this project.
