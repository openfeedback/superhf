"""
Trainers for finetuning models according to SuperHF (maximizing the scores
from a reward model with expert iteration using supervised learning).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
    BatchEncoding,
)
from torchtyping import TensorType  # type: ignore

from superhf.data import ListDataset
from superhf.filtering import CompletionFilterBase


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
    report_to: list[str] = field(
        default_factory=lambda: [],
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )


class SuperHFTrainer:
    """
    A basic form of Super HF: filtering completions by the reward model
    and fine-tuning the language model on the filtered completions.

    Iteratively, in a loop, we:
        1. Sample a superbatch of prompts from the training set without replacement.
        2. Use the language model to generate a completion for each prompt.
        3. Use the reward model to score the completions.
        4. Use some filter function to filter the top completions.
        5. Fine-tune the language model on the top completions.
        6. Optionally report metrics.

    Note that the model is updated for each superbatch, so its sampling
    distribution changes over time. This is a form of curriculum learning or
    expert iteration.
    """

    def __init__(
        self,
        language_model: Any,
        reward_model: Any,
        language_tokenizer: PreTrainedTokenizerBase,
        reward_tokenizer: PreTrainedTokenizerBase,
        completion_filter: CompletionFilterBase,
        training_args: SuperHFTrainingArguments,
    ) -> None:
        self.language_model = language_model
        self.reward_model = reward_model
        self.language_tokenizer = language_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.completion_filter = completion_filter
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

        # Then, iterate over the prompts in superbatches
        for superbatch_index, superbatch_prompts in tqdm(
            enumerate(prompts_dataloader), total=len(prompts_dataloader)
        ):
            # Generate completions for each prompt in the superbatch
            completions_encoded = self.generate_completions(superbatch_prompts)

            # Score the completions
            completions, scores = self.score_completions(completions_encoded)

            # Filter the completions
            filtered_completions, filtered_scores = self.completion_filter.filter(
                completions, scores
            )

            # Fine-tune the language model on the filtered completions
            average_loss = self.finetune_language_model(filtered_completions)

            # Optionally report metrics
            self.report_metrics(
                superbatch_index,
                completions,
                filtered_completions,
                scores,
                filtered_scores,
                average_loss,
            )

            # Optionally, save the model
            # self.save_model()

    def collate_fn_lm(self, batch: list[str]) -> BatchEncoding:
        """
        Collate function for the language model DataLoader.
        """
        return self.language_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_args.max_length_rm,
        )

    def generate_completions(
        self, superbatch_prompts: list[str]
    ) -> list[TensorType["batch", "seq_len"]]:
        """Generate completions for the prompts in the superbatch."""
        completion_dataloader = DataLoader(
            ListDataset(superbatch_prompts),
            batch_size=self.training_args.minibatch_size_initial,
            collate_fn=self.collate_fn_lm,
        )
        completions: list[str] = []
        for minibatch in completion_dataloader:
            encodings = minibatch
            completions.extend(
                self.language_model.generate(
                    encodings,
                    max_length=self.training_args.max_length_lm,
                    temperature=self.training_args.temperature,
                    top_p=self.training_args.top_p,
                    do_sample=True,
                    num_return_sequences=1,
                )
            )
        return completions

    def collate_fn_rm(
        self, batch: list[TensorType["batch", "seq_len"]]
    ) -> tuple[list[str], list[TensorType["batch", "seq_len"]]]:
        """
        Collate function for the reward model's dataloader.

        Takes encoded completions from the language model, decodes them, encodes them for the
        reward model, and returns both the decoded completion text and re-encoded completions.
        """
        completions = self.language_tokenizer.batch_decode(
            batch, skip_special_tokens=True
        )
        return (
            completions,
            self.reward_tokenizer(
                completions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.training_args.max_length_rm,
            ),
        )

    def score_completions(
        self, completions_encoded: list[TensorType["batch", "seq_len"]]
    ) -> tuple[list[str], list[TensorType[1]]]:
        """
        Score the completions.

        Returns a tuple of the decoded completions and the scores.
        """
        all_completions: list[str] = []
        all_scores: list[TensorType[1]] = []

        score_dataloader = DataLoader(
            ListDataset(completions_encoded),
            batch_size=self.training_args.minibatch_size_initial,
            collate_fn=self.collate_fn_rm,
        )

        for minibatch in score_dataloader:
            completions, completion_encodings = minibatch
            scores = self.reward_model(**completion_encodings)
            all_completions.extend(completions)
            all_scores.extend(scores)
        return all_completions, all_scores

    def finetune_language_model(
        self, filtered_completions: list[BatchEncoding]
    ) -> TensorType[1]:
        """
        Fine-tune the language model on the completions.
        """
        finetuning_dataloader = DataLoader(
            ListDataset(filtered_completions),
            batch_size=self.training_args.minibatch_size_initial,
            collate_fn=self.collate_fn_lm,
        )
        self.language_model.train()
        for minibatch in finetuning_dataloader:
            encodings = minibatch
            self.language_model(**encodings)
            # TODO loss and optimizer
        average_loss = torch.rand(1) * 10
        return average_loss

    def report_metrics(
        self,
        superbatch_index: int,
        completions: list[str],
        filtered_completions: list[str],
        scores: list[float],
        filtered_scores: list[float],
        average_loss: TensorType[1],
    ) -> None:
        """
        Report metrics.
        """
        # if self.training_args.report_metrics:
        #     raise NotImplementedError
        average_completion_length = np.mean([len(c) for c in completions])
        average_filtered_completion_length = np.mean(
            [len(c) for c in filtered_completions]
        )
        average_score = np.mean(scores)
        average_filtered_score = np.mean(filtered_scores)
        if "print" in self.training_args.report_to:
            print(
                f"Superbatch {superbatch_index}: {len(completions)} completions, "
                f"{len(filtered_completions)} filtered completions, average completion length "
                f"{average_completion_length}, average filtered completion length "
                f"{average_filtered_completion_length}, average score {average_score}, average "
                f"filtered score {average_filtered_score}, average loss {average_loss}."
            )

    def save_model(self) -> None:
        """
        Save the model.
        """
        raise NotImplementedError
