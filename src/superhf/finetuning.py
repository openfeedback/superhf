"""
Trainers for finetuning models according to SuperHF (maximizing the scores
from a reward model without the use of reinforcement learning).
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Union, Any, Optional

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    pipeline,
    EvalPrediction,
)
from transformers.pipelines.pt_utils import KeyDataset
from datasets.arrow_dataset import Dataset
from datasets.utils import logging

# from torchtyping import TensorType


@dataclass
class SuperHFTrainingArguments(TrainingArguments):
    """
    Training arguments for SuperHF trainers.
    """

    reward_model: Optional[nn.Module] = None


class SuperHFTrainer(ABC):
    """
    Base class for SuperHF trainers.

    Fine-tuned a language model to maximize the scores from a reward model.
    """

    def __init__(
        self,
        language_model: PreTrainedModel,
        reward_model: PreTrainedModel,
        language_tokenizer: PreTrainedTokenizerBase,
        reward_tokenizer: PreTrainedTokenizerBase,
        train_prompts: List[str],
        test_prompts: List[str],
    ) -> None:
        self.language_model = language_model
        self.reward_model = reward_model
        self.language_tokenizer = language_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.train_prompts = train_prompts
        self.test_prompts = test_prompts

    @abstractmethod
    def train(self) -> None:
        """
        The main training and evaluation loop.
        """
        raise NotImplementedError


class SinglePassBestOfNTrainer(SuperHFTrainer):
    """
    The most basic form of Super HF: filtering completions by the reward model
    and fine-tuning the language model on the filtered completions.

    As one long training sequence
        1. Use $M$ to generate $n$ completions for each of $d$ training train_dataset prompts
              ($d*completions_per_prompt$ total).
        2. Use $R$ to select the top 1 of the $n$ completions for each prompt ($d$ total).
        3. Fine-tune $M$ on the $d$ best-of-$n$ completions.
        4. Evaluate the loss and average reward during training.
    """

    def __init__(
        self,
        language_model: PreTrainedModel,
        reward_model: PreTrainedModel,
        language_tokenizer: PreTrainedTokenizerBase,
        reward_tokenizer: PreTrainedTokenizerBase,
        train_prompts: List[str],
        test_prompts: List[str],
        temperature: float = 0.7,
        completions_per_prompt: int = 4,
        output_dir: str = "output",
    ) -> None:
        super().__init__(
            language_model,
            reward_model,
            language_tokenizer,
            reward_tokenizer,
            train_prompts,
            test_prompts,
        )
        self.temperature = temperature
        self.completions_per_prompt = completions_per_prompt
        self.output_dir = output_dir
        self.training_args: Any = None
        self.eval_dataset: Any = None
        # Make output dir if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self) -> None:
        """
        Main training and evaluation loop.
        """

    def generate_completions(self, batch_size: int, max_new_tokens: int) -> None:
        """
        Use $M$ to generate $n$ completions for each of $d$ training train_dataset prompts.
        """
        # Set up tokenizer
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

        # Switch to eval mode
        self.language_model.eval()

        # Debug: only use a subset of the prompts
        # self.train_prompts = self.train_prompts[:8]

        # Duplicate each of the prompts $n$ times.
        prompts = self.train_prompts * self.completions_per_prompt

        # Convert prompts into a train_dataset
        train_dataset = Dataset.from_dict({"prompt": prompts})

        # Use $M$ to generate $n$ completions for each of $d$ training train_dataset prompts
        # ($d*completions_per_prompt$ total). Iterate in groups of batch_size.
        pipe = pipeline(
            "text-generation",
            model=self.language_model,
            tokenizer=self.language_tokenizer,
            device=self.language_model.device,
        )

        print("Generating completions...")
        completions: List[str] = []

        # for out in pipe(KeyDataset(train_dataset, "prompt")):
        for out in tqdm(
            pipe(
                KeyDataset(train_dataset, "prompt"),
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                pad_token_id=self.language_tokenizer.pad_token_id,
                early_stopping=True,
                do_sample=True,
            ),
            total=len(train_dataset),
        ):
            completion = out[0]["generated_text"]
            # Filter out everything including and after the second "\n\nHuman:"
            # sequence in case the model repeats the prompt
            completion = "\n\nHuman:".join(completion.split("\n\nHuman:")[:2])
            completion = "\n\nAssistant:".join(completion.split("\n\nAssistant:")[:2])
            completions.append(completion)

        # Save it to a file, writing raw string outputs (e.g. keeping '\n' in plaintext)
        torch.save(completions, os.path.join(self.output_dir, "completions.pt"))

    def score_completions(self, batch_size: int) -> None:
        """
        Use $R$ to evaluate each completion.
        """

        completions: List[str] = torch.load(
            os.path.join(self.output_dir, "completions.pt")
        )

        num_prompts: int = len(self.train_prompts)
        # Debug: only use a subset of the completions
        # completions = [
        #     completion
        #     for i, completion in enumerate(completions)
        #     if i % num_prompts < 1024
        # ]

        # OOM Fix: Filter completions in a set longer than 1000 characters
        bad_indices = []
        for i, completion in enumerate(completions):
            if len(completion) > 1000:
                bad_indices.append(i % num_prompts)
        old_size = len(completions)
        completions = [
            completion
            for i, completion in enumerate(completions)
            if i % num_prompts not in bad_indices
        ]
        new_size = len(completions)
        print(
            f"Loaded {new_size} completions (filtered {old_size - new_size} from {old_size} total)."
        )

        train_dataset = Dataset.from_dict({"completion": completions})

        # Use $R$ to select the top 1 of the $n$ completions for each prompt ($d$ total).
        pipe = pipeline(
            "text-classification",
            model=self.reward_model,
            tokenizer=self.reward_tokenizer,
            device=self.reward_model.device,
        )
        scored_completions: List[Dict[str, Union[str, float]]] = []
        print("Scoring completions...")
        for row, completion in zip(
            tqdm(
                pipe(
                    KeyDataset(train_dataset, "completion"),
                    batch_size=batch_size,
                    max_length=512,
                ),
                total=len(train_dataset),
            ),
            completions,
        ):
            scored_completions.append({"score": row["score"], "completion": completion})

        print(
            f'Average reward: {np.mean([float(row["score"]) for row in scored_completions])}'
        )

        torch.save(
            scored_completions,
            os.path.join(self.output_dir, "scored_completions.pt"),
        )

    def filter_completions(self) -> Any:
        """
        Select the top 1 of the $n$ completions for each prompt ($d$ total)
        """
        scored_completions = torch.load(
            os.path.join(self.output_dir, "scored_completions.pt")
        )
        num_prompts: int = len(scored_completions) // self.completions_per_prompt

        # Group the completions for the same prompt together into a list of lists.
        # E.g. ['a1', 'b1', 'c1', 'a2', 'b2', 'c2'] -> [['a1', 'a2'], ['b1', 'b2'], ['c1', 'c2']]
        grouped_scored_completions: List[List[Dict[str, Union[str, float]]]] = [
            [] for _ in range(num_prompts)
        ]
        for i, scored_completion in enumerate(scored_completions):
            grouped_scored_completions[i % num_prompts].append(scored_completion)

        # Filter for the best completion for each group
        filtered_completions: List[Dict[str, Union[str, float]]] = []
        for group in grouped_scored_completions:
            filtered_completions.append(max(group, key=lambda x: x["score"]))

        torch.save(
            filtered_completions,
            os.path.join(self.output_dir, "filtered_completions.pt"),
        )
        return scored_completions, filtered_completions

    def tune_model(self, training_args: TrainingArguments) -> None:
        """
        Fine-tune $M$ on the $d$ best-of-$n$ completions.
        Evaluate the loss and average reward during training.
        """
        self.training_args = training_args
        filtered_completions = torch.load(
            os.path.join(self.output_dir, "filtered_completions.pt")
        )
        print(f"Loaded {len(filtered_completions)} filtered completions")

        train_dataset = Dataset.from_dict({"completion": filtered_completions})
        eval_dataset = Dataset.from_dict({"prompt": self.test_prompts})
        self.eval_dataset = eval_dataset

        print("Pre-processing datasets...")
        logging.enable_progress_bar()
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

        train_dataset_processed = train_dataset.map(
            lambda examples: self.language_tokenizer(
                [example["completion"] for example in examples["completion"]],
                truncation=True,
            ),
            batched=True,
        )
        test_dataset_processed = eval_dataset.map(
            lambda examples: self.language_tokenizer(
                list(examples["prompt"]),
                truncation=True,
            ),
            batched=True,
        )

        print("Beginning training...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.language_tokenizer, mlm=False
        )
        self.language_model.train()
        trainer = Trainer(
            model=self.language_model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataset_processed,
            eval_dataset=test_dataset_processed,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def compute_metrics(self, _: EvalPrediction) -> Dict[str, float]:
        """
        Compute the average reward of new completions on the test prompts.

        We ignore the predictions and labels because we need to generate full
        completions for the test prompts, which is not possible with the
        Trainer API.
        """
        pipe = pipeline(
            "text-generation",
            model=self.language_model,
            tokenizer=self.language_tokenizer,
            device=self.language_model.device,
        )

        completions: List[str] = []
        for out in pipe(
            KeyDataset(self.eval_dataset, "prompt"),
            batch_size=self.training_args.eval_batch_size,
            max_new_tokens=256,
            # temperature=self.temperature,
            pad_token_id=self.language_tokenizer.pad_token_id,
            early_stopping=True,
            do_sample=False,
        ):
            completion = out[0]["generated_text"]
            # Filter out everything including and after the second "\n\nHuman:"
            # sequence in case the model repeats the prompt
            completion = "\n\nHuman:".join(completion.split("\n\nHuman:")[:2])
            completion = "\n\nAssistant:".join(completion.split("\n\nAssistant:")[:2])
            completions.append(completion)

        # Now evaluate the completions with the reward model
        completions_dataset = Dataset.from_dict({"completion": completions})

        # Use $R$ to select the top 1 of the $n$ completions for each prompt ($d$ total).
        pipe = pipeline(
            "text-classification",
            model=self.reward_model,
            tokenizer=self.reward_tokenizer,
            device=self.reward_model.device,
        )
        scores: List[float] = []
        print("Scoring completions...")
        for row, completion in zip(
            pipe(
                KeyDataset(completions_dataset, "completion"),
                batch_size=self.training_args.eval_batch_size,
                max_length=512,
            ),
            completions,
        ):
            scores.append(row["score"])

        average_reward = float(np.mean(scores))
        return {"average_reward": average_reward}


# class IterativeBestOfNTrainer(SuperHFTrainer):
#     """
#     The most basic form of Super HF: filtering completions by the reward model
#     and fine-tuning the language model on the filtered completions.

#     Iteratively, in a loop, we:
#         1. Sample $p$ prompts from the training set without replacement.
#         2. Use $M$ to generate $n$ completions for each prompt ($p*completions_per_prompt$ total).
#         3. Use $R$ to select the top 1 of the $n$ completions for each prompt ($p$ total).
#         4. Fine-tune $M$ on the $p$ best-of-$n$ completions.
#         5. Store the fine-tuning loss and average reward across the $p$ best-of-$n$ completions.
#     """

#     def __init__(
#         self,
#         language_model: GenerationMixin,
#         reward_model: nn.Module,
#         language_tokenizer: Any,
#         reward_tokenizer: Any,
#         train_prompts: List[str],
#         test_prompts: List[str],
#         temperature: float = 1.0,
#         completions_per_prompt: int = 2,
#     ) -> None:
#         super().__init__(
#             language_model,
#             reward_model,
#             language_tokenizer,
#             reward_tokenizer,
#             train_prompts,
#             test_prompts,
#         )
#         self.temperature = temperature
#         self.completions_per_prompt = completions_per_prompt

#     def train(self) -> None:
#         """
#         Main training and evaluation loop.
#         """
#         raise NotImplementedError
