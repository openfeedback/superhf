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


class BestOfNWrapper(torch.nn.Module):
    """
    Accepts a language model and a reward model.
    Modifies the language model forward method to do best of n.
    """

    def __init__(
        self,
        language_model: Any,
        reward_model: Any,
        language_tokenizer: Any,
        reward_tokenizer: Any,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.reward_model = reward_model
        self.language_tokenizer = language_tokenizer
        self.reward_tokenizer = reward_tokenizer

    def generate(self, best_of_n=2, **kwargs: Any) -> Any:
        """
        Runs the language model best_of_n times and returns the best outputs.
        """
        # run the language model n times
        lm_outputs = []  # size best_of_n
        for _ in range(best_of_n):
            out = self.language_model.generate(**kwargs)
            lm_outputs.append(out)
            # out has shape [batch_size, seq_len]
        batch_size, seq_len = lm_outputs[0].shape[0], lm_outputs[0].shape[1]
        lm_outputs_stacked = torch.stack(lm_outputs)
        # ^ size [best_of_n, batch_size, seq_len]
        result = []
        for batch_id in range(batch_size):
            out_str = self.language_tokenizer.batch_decode(
                lm_outputs_stacked[:, batch_id, :], skip_special_tokens=True
            )
            out_tokens = self.reward_tokenizer(
                out_str, return_tensors="pt", padding=True
            ).to(self.reward_model.device)

            # get the rewards for each output
            reward_output = self.reward_model(**out_tokens)
            try:
                reward_tensor = reward_output.logits
            except AttributeError:
                reward_tensor = reward_output
            # ^ shape [best_of_n, 1]
            reward_tensor = reward_tensor.to("cpu")
            result.append(lm_outputs[np.argmax(reward_tensor, axis=0)][batch_id, :])
        result_stacked = torch.stack(result, dim=0)
        assert (result_stacked.shape[0], result_stacked.shape[1]) == (
            batch_size,
            seq_len,
        )
        return result_stacked

    def forward(self, **kwargs: Any) -> Any:
        """Uses the language model for forward pass."""
        return self.language_model.forward(**kwargs)


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
        for completions in data.values():
            for _ in range(comparisons_per_dataset):
                samples = random.sample(completions, 2)
                score = calculate_meteor_similarity_only_completions(samples)
                all_scores.append(score)
    return all_scores
