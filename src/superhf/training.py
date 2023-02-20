"""
Trainers for finetuning models according to SuperHF (maximizing the scores
from a reward model with expert iteration using supervised learning).
"""

# import os
from dataclasses import dataclass, field
from typing import List, Any, Optional, Callable

# import random

# import torch
# from torch import nn

# import numpy as np
# from tqdm import tqdm
# from transformers import (
#     # Trainer,
#     # DataCollatorForLanguageModeling,
#     # PreTrainedModel,
#     # PreTrainedTokenizerBase,
#     # pipeline,
#     # EvalPrediction,
# )
# from transformers.pipelines.pt_utils import KeyDataset
# from datasets.arrow_dataset import Dataset
# from datasets.utils import logging

# from torchtyping import TensorType


@dataclass
class SuperHFTrainingArguments:
    """
    Training arguments for SuperHF trainers.
    """

    # pylint: disable=too-many-instance-attributes

    # Models
    language_model: Any = None
    reward_model: Any = None
    language_tokenizer: Any = None
    reward_tokenizer: Any = None

    # Data
    prompts: List[str] = field(default_factory=list)

    # Generation
    temperature: float = 1.0
    completions_per_prompt: int = 2

    # Filtering
    filter_function: Optional[Callable[[List[float]], List[bool]]] = None

    # Metrics
    report_to: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )


class SuperHFTrainer:
    """
    The most basic form of Super HF: filtering completions by the reward model
    and fine-tuning the language model on the filtered completions.

    Iteratively, in a loop, we:
        1. Sample $p$ prompts from the training set without replacement.
        2. Use $M$ to generate $n$ completions for each prompt ($p*completions_per_prompt$ total).
        3. Use $R$ to select the top 1 of the $n$ completions for each prompt ($p$ total).
        4. Fine-tune $M$ on the $p$ best-of-$n$ completions.
        5. Store the fine-tuning loss and average reward across the $p$ best-of-$n$ completions.
    """

    def __init__(self, training_args: SuperHFTrainingArguments) -> None:
        self.training_args = training_args

    def train(self) -> None:
        """
        Main training and evaluation loop.
        """
        raise NotImplementedError
