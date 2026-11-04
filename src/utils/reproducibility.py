"""Random seed setting across standard libraries."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across python, numpy, and environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
