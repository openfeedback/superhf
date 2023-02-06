"""
Trainers for finetuning models according to SuperHF (maximizing the scores
from a reward model without the use of reinforcement learning).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from torch import nn
import tqdm
from transformers import (
    # Trainer,
    TrainingArguments,
    # DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    # TextGenerationPipeline,
)
from torchtyping import TensorType


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
        1. Use $M$ to generate $n$ completions for each of $d$ training dataset prompts
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
        temperature: float = 1.0,
        completions_per_prompt: int = 4,
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

    def train(self, batch_size: int = 8) -> None:
        """
        Main training and evaluation loop.
        """
        # Set up tokenizer
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

        # Duplicate each of the prompts $n$ times.
        prompts = self.train_prompts * self.completions_per_prompt

        # Use $M$ to generate $n$ completions for each of $d$ training dataset prompts
        # ($d*completions_per_prompt$ total). Iterate in groups of batch_size.
        # TODO try a pipeline
        # pipeline = TextGenerationPipeline(self.language_model, self.language_tokenizer)

        completions = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i : i + batch_size]
            tokenized = self.language_tokenizer(
                batch, padding=True, return_tensors="pt"
            ).to(self.language_model.device)

            generated_tokens: TensorType = self.language_model.generate(
                inputs=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_length=256,
                num_beams=1,  # TODO try removing this
                temperature=self.temperature,
                early_stopping=True,
                pad_token_id=self.language_tokenizer.pad_token_id,
            )

            # Convert to a list
            generated_tokens = generated_tokens.tolist()

            # Decode the tokens
            completions.extend(
                self.language_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
            )

        # Use $R$ to select the top 1 of the $n$ completions for each prompt ($d$ total).
        # Fine-tune $M$ on the $d$ best-of-$n$ completions.
        # Evaluate the loss and average reward during training.


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
