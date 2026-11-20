"""Global seeding for reproducibility.

PyTorch seeding will be added here once models are introduced (step 2+).
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python's and NumPy's global generators and return a fresh NumPy Generator.

    ``PYTHONHASHSEED`` is also set so that any subprocesses inherit a fixed hash seed;
    it cannot change the hash seed of the already-running interpreter.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Random seed set to %d (Python, NumPy).", seed)
    return np.random.default_rng(seed)
