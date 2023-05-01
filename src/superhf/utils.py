"""
Assorted utility functions.
"""

import gc
import random
from tqdm import tqdm

import numpy as np
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import psutil

from superhf import constants


def set_seed(seed: int) -> None:
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def separate_prompt_from_completion(text: str) -> tuple[str, str]:
    """
    Given a completed prompt text, separate the part before and including the
    prompt delimiter from the part after.
    """
    prompt, completion = text.split(constants.PROMPT_DELIMITER, 1)
    prompt += constants.PROMPT_DELIMITER
    return prompt, completion


def print_gpu_utilization() -> None:
    """Deprecated. Use print_memory_utilization instead."""
    # Give a warning
    print(
        "WARNING: print_gpu_utilization is deprecated. Use print_memory_utilization"
        " instead."
    )
    print_memory_utilization()


def print_memory_utilization() -> None:
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

    # Also print CPU RAM usage
    ram_percent = psutil.virtual_memory().percent
    ram_gb = psutil.virtual_memory().used / 1024**3
    output = f"CPU RAM Usage: {ram_gb:.2f} GB ({ram_percent:.2f}%)"

    # for each GPU, get the name and the memory occupied
    output += " | GPU VRAM Usage:"
    for i in range(n_gpu):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        vram_gb = info.used / 1024**3
        vram_percent = info.used / info.total * 100
        output += f" {i}: {vram_gb:.2f} GB ({vram_percent:.2f}%)"
    tqdm.write(output)
