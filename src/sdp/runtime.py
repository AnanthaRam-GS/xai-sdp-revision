"""Process-level runtime fixes.

On macOS, PyTorch, XGBoost (via Homebrew's libomp) and scikit-learn each load their
own OpenMP runtime (libomp). Several LLVM OpenMP runtimes in one process crash with a
segmentation fault ("OMP: Error #179: pthread_mutex_init failed"), whatever the import
order. Pointing ``DYLD_LIBRARY_PATH`` at PyTorch's ``libomp`` makes every library
resolve to that single runtime. The variable only takes effect at process start, so
:func:`ensure_single_openmp_runtime` re-executes the current process once with it set.
Child processes (e.g. joblib workers) inherit it.

It must run before torch, xgboost or scikit-learn are imported: entry points call it
from ``scripts/__init__.py`` and the test suite from ``tests/conftest.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_GUARD = "SDP_OPENMP_REEXEC"


def torch_lib_dir() -> str | None:
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
    return str(lib) if (lib / "libomp.dylib").exists() else None


def needs_reexec() -> bool:
    """True on macOS when the process does not yet use PyTorch's OpenMP runtime only."""
    if sys.platform != "darwin" or os.environ.get(_GUARD):
        return False
    lib = torch_lib_dir()
    return lib is not None and lib not in os.environ.get("DYLD_LIBRARY_PATH", "").split(os.pathsep)


def ensure_single_openmp_runtime() -> None:
    """Re-execute the process with a single OpenMP runtime on macOS (no-op elsewhere)."""
    if not needs_reexec():
        return
    lib = torch_lib_dir()
    current = os.environ.get("DYLD_LIBRARY_PATH", "")
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(p for p in (lib, current) if p)
    env[_GUARD] = "1"
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, sys.orig_argv, env)
