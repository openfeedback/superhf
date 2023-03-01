"""
Assorted utility functions.
"""

import gc
import random

import numpy as np
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def set_seed(seed: int) -> None:
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_gpu_utilization() -> None:
    """
    Print the GPU memory occupied using nvidia-smi. If no GPU is available, do nothing.
    """
    if not torch.cuda.is_available():
        return

    # Garbage collect
    gc.collect()
    torch.cuda.empty_cache()

    nvmlInit()
    # get the number of GPUs
    n_gpu = torch.cuda.device_count()

    # for each GPU, get the name and the memory occupied
    print("GPU memory occupied:", end="")
    for i in range(n_gpu):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f" Device{i}: {info.used//1024**2} MB", end=";")
    print()
