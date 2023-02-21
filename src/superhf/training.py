"""
Trainers for finetuning models according to SuperHF (maximizing the scores
from a reward model with expert iteration using supervised learning).
"""

# import os
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

# import random
# import torch
# from torch import nn
from torch.utils.data import DataLoader

# import numpy as np
# from tqdm import tqdm
from transformers import (
    # Trainer,
    # DataCollatorForLanguageModeling,
    # PreTrainedModel,
    PreTrainedTokenizerBase,
    BatchEncoding,
    # pipeline,
    # EvalPrediction,
)

# from transformers.pipelines.pt_utils import KeyDataset
# from datasets.arrow_dataset import Dataset
# from datasets.utils import logging

# from torchtyping import TensorType

from superhf.data import ListDataset


@dataclass
class SuperHFTrainingArguments:
    """
    Training arguments for SuperHF trainers.
    """

    # pylint: disable=too-many-instance-attributes

    # Generation
    temperature: float = 1.0
    top_p: float = 0.9
    superbatch_size: int = field(
        default=128,
        metadata={
            "help": "Number of completions to generate with the current "
            "policy before filtering and fine-tuning."
        },
    )
    max_length_lm: int = 256
    max_length_rm: int = 1024

    # Batching
    minibatch_size_initial: int = field(
        default=32,
        metadata={"help": "Size of minibatches for not running out of memory."},
    )

    # Filtering
    filter_function: Optional[Callable[[list[float]], list[bool]]] = None

    # Metrics
    report_to: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )


class SuperHFTrainer:
    """
    A basic form of Super HF: filtering completions by the reward model
    and fine-tuning the language model on the filtered completions.

    Iteratively, in a loop, we:
        1. Sample a megabatch of prompts from the training set without replacement.
        2. Use the language model to generate a completion for each prompt.
        3. Use the reward model to score the completions.
        4. Use some filter function to filter the top completions.
        5. Fine-tune the language model on the top completions.
        6. Optionally report metrics.

    Note that the model is updated for each megabatch, so its sampling
    distribution changes over time. This is a form of curriculum learning or
    expert iteration.
    """

    def __init__(
        self,
        language_model: Any,
        reward_model: Any,
        language_tokenizer: PreTrainedTokenizerBase,
        reward_tokenizer: PreTrainedTokenizerBase,
        training_args: SuperHFTrainingArguments,
    ) -> None:
        self.language_model = language_model
        self.reward_model = reward_model
        self.language_tokenizer = language_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.training_args = training_args

        # Add padding tokens if they are not already there
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token
            print("Added pad token to language tokenizer.")
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
            print("Added pad token to reward tokenizer.")

    def train(self, prompts: list[str]) -> None:
        """
        Main training and evaluation loop.
        """

        # First, put all the prompts into a Dataset and DataLoader
        prompts_dataloader = DataLoader(
            ListDataset(prompts), batch_size=self.training_args.superbatch_size
        )

        # Then, iterate over the prompts in megabatches
        for megabatch in prompts_dataloader:
            # Generate completions for each prompt in the megabatch
            completions: list[str] = []
            completion_dataloader = DataLoader(
                ListDataset(megabatch),
                batch_size=self.training_args.minibatch_size_initial,
                collate_fn=self.collate_fn,
            )
            for minibatch in completion_dataloader:
                completions.extend(
                    self.language_model.generate(
                        minibatch,
                        max_length=self.training_args.max_length_lm,
                        temperature=self.training_args.temperature,
                        top_p=self.training_args.top_p,
                        do_sample=True,
                        num_return_sequences=1,
                    )
                )

            # Score the completions
            scores = self.reward_model(completions)

            # Filter the completions
            filtered_completions = self.filter_completions(completions, scores)

            # Fine-tune the language model on the filtered completions
            self.finetune_language_model(filtered_completions)

            # Optionally report metrics
            self.report_metrics()

            # Optionally, save the model

    def collate_fn(self, batch: list[str]) -> BatchEncoding:
        """
        Collate function for the DataLoader.
        """
        return self.language_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_args.max_length_rm,
        )

    def filter_completions(
        self, completions: list[str], scores: list[float]
    ) -> list[str]:
        """
        Filter the completions by the scores.
        """
        if self.training_args.filter_function is None:
            return completions
        return [
            completion
            for completion, keep in zip(
                completions, self.training_args.filter_function(scores)
            )
            if keep
        ]

    def finetune_language_model(self, completions: list[str]) -> None:
        """
        Fine-tune the language model on the completions.
        """
        raise NotImplementedError

    def report_metrics(self) -> None:
        """
        Report metrics.
        """
        raise NotImplementedError
