"""
Assorted utility functions.
"""

import json
import gc
from itertools import permutations
import random
from typing import Any, Optional
from tqdm import tqdm

from nltk.translate.meteor_score import meteor_score
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
    output = f"💾 CPU RAM Usage: {ram_gb:.2f} GB ({ram_percent:.2f}%)"

    # for each GPU, get the name and the memory occupied
    output += " | GPU VRAM Usage:"
    for i in range(n_gpu):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        vram_gb = info.used / 1024**3
        vram_percent = info.used / info.total * 100
        output += f" {i}: {vram_gb:.2f} GB ({vram_percent:.2f}%)"
    tqdm.write(output)


def _calculate_meteor_similarity(strings: list[str]) -> float:
    """Calculates the average METEOR similarity between all pairs of strings."""
    scores = []
    for string1, string2 in permutations(strings, 2):
        score = meteor_score([string1.split()], string2.split())
        scores.append(score)
    return float(np.mean(scores))


def calculate_meteor_similarity_only_completions(
    completions: list[str],
) -> Optional[float]:
    """Separate completions from a superbatch then calculate METEOR similarity."""
    if len(completions) < 2:
        return None
    non_prompt_completions = [
        separate_prompt_from_completion(completion)[1] for completion in completions
    ]
    return _calculate_meteor_similarity(non_prompt_completions)


def bootstrap_meteor_similarity_from_completions(
    completion_filepath: str, comparisons_per_dataset: int = 200
) -> list[Any]:
    """Given a completions filename, calculate many of pairwise meteor scores per dataset."""
    all_scores = []
    with open(completion_filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
        for completions in tqdm(data.values(), desc="METEOR (Dataset)"):
            for _ in range(comparisons_per_dataset):
                samples = random.sample(completions, 2)
                score = calculate_meteor_similarity_only_completions(samples)
                all_scores.append(score)
    return all_scores
