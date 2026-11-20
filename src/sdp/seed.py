"""Global seeding for reproducibility (Python, NumPy and PyTorch)."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, torch_threads: int | None = 1, quiet: bool = False) -> np.random.Generator:
    """Seed Python's, NumPy's and PyTorch's generators and return a fresh NumPy Generator.

    PyTorch is also switched to deterministic algorithms and, if ``torch_threads`` is
    given, limited to that many intra-op threads (1 by default, as in every worker).
    ``PYTHONHASHSEED`` is set so that subprocesses inherit a fixed hash seed; it cannot
    change the hash seed of the already-running interpreter.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
    except ImportError:  # pragma: no cover - torch is a hard dependency from step 3
        pass
    if not quiet:
        logger.info("Random seed set to %d (Python, NumPy, PyTorch).", seed)
    return np.random.default_rng(seed)
