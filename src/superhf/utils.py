"""
Assorted utility functions.
"""

import random

import numpy as np
import torch


def set_seed(seed):
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
