"""Pytest configuration.

On macOS the test process is re-executed once so that PyTorch, XGBoost and
scikit-learn share a single OpenMP runtime (see ``sdp.runtime``). Pytest's output
capturing is stopped first so that the re-executed run writes to the terminal.
"""

from sdp.runtime import ensure_single_openmp_runtime, needs_reexec


def pytest_configure(config):
    if needs_reexec():
        capman = config.pluginmanager.getplugin("capturemanager")
        if capman is not None:
            capman.stop_global_capturing()
        ensure_single_openmp_runtime()
