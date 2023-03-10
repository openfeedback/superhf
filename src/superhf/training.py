"""
Trainers for finetuning models according to SuperHF (maximizing the scores
from a reward model with expert iteration using supervised learning).
"""

from dataclasses import dataclass, field
import re
from typing import Callable, Optional, Union


import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator, find_executable_batch_size
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
    BatchEncoding,
    PreTrainedModel,
    LogitsProcessorList,
)
from torchtyping import TensorType

from superhf import constants
from superhf.data import ListDataset
from superhf.filtering import CompletionFilterBase
from superhf.metrics import SuperHFMetrics, report_metrics_print
from superhf.utils import print_gpu_utilization, separate_prompt_from_completion


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
            "help": (
                "Number of completions to generate with the current "
                "policy before filtering and fine-tuning."
            )
        },
    )
    max_new_tokens: int = 256
    max_length_rm: int = 1024
    logits_processors: Optional[LogitsProcessorList] = None
    conversation_prompt: str = ""  # the prompt to be prepended to all prompts

    # Batching to avoid OOMs
    minibatch_size_generating: int = 64
    minibatch_size_scoring: int = 64
    minibatch_size_finetuning: int = 64

    # Training
    learning_rate: float = 1e-5
    inverse_loss_penalty: float = 0.0
    mixed_precision: str = "no"

    # Dataset settings
    prompt_delimiter: str = constants.PROMPT_DELIMITER


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
        language_model: PreTrainedModel,
        reward_model: PreTrainedModel,
        language_tokenizer: PreTrainedTokenizerBase,
        reward_tokenizer: PreTrainedTokenizerBase,
        completion_filter: CompletionFilterBase,
        training_args: SuperHFTrainingArguments,
        report_metrics: Optional[
            Union[
                Callable[[SuperHFMetrics], None], list[Callable[[SuperHFMetrics], None]]
            ]
        ] = None,
    ) -> None:
        self.language_model = language_model
        self.reward_model = reward_model
        self.language_tokenizer = language_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.completion_filter = completion_filter
        self.training_args = training_args
        if report_metrics is None:
            report_metrics = [report_metrics_print]
        elif not isinstance(report_metrics, list):
            report_metrics = [report_metrics]
        self.report_metrics = report_metrics

        # Add padding tokens if they are not already there
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token
            print("Added pad token to language tokenizer.")
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
            print("Added pad token to reward tokenizer.")

        # Reward model is always in eval mode
        self.reward_model.eval()

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
            enumerate(prompts_dataloader),
            total=len(prompts_dataloader),
            desc="Superbatch",
        ):
            tqdm.write(
                f"Before generation, on superbatch_index {superbatch_index} ", end=""
            )
            print_gpu_utilization()
            # Generate completions for each prompt in the superbatch
            completions_encoded = find_executable_batch_size(
                self.generate_completions,
                self.training_args.minibatch_size_generating,
            )(superbatch_prompts)

            tqdm.write("Before scoring ", end="")
            print_gpu_utilization()
            # Score the completions
            completions, scores = find_executable_batch_size(
                self.score_completions,
                self.training_args.minibatch_size_scoring,
            )(completions_encoded)

            tqdm.write("Before filtering ", end="")
            print_gpu_utilization()
            # Filter the completions
            filtered_completions, filtered_scores = self.completion_filter.filter(
                completions, scores
            )

            # Fine-tune the language model on the filtered completions
            average_loss = find_executable_batch_size(
                self.finetune_language_model,
                self.training_args.minibatch_size_finetuning,
            )(filtered_completions)

            # Optionally report metrics
            metrics = SuperHFMetrics(
                superbatch_index=superbatch_index,
                superbatch_count=len(prompts_dataloader),
                completions=completions,
                filtered_completions=filtered_completions,
                scores=scores,
                filtered_scores=filtered_scores,
                average_loss=average_loss,
                scheduler_lr=self.training_args.learning_rate,  # TODO get this from the scheduler
            )
            if self.report_metrics is not None:
                for report_metrics_function in self.report_metrics:
                    report_metrics_function(metrics)

            # Optionally, save the model
            # self.save_model()

    def collate_fn_lm_completions(self, batch: list[str]) -> BatchEncoding:
        """
        Collate function for the language model completions DataLoader.

        Prepends the constitution to each prompt. By default this is the empty string.
        """
        constitution = self.training_args.conversation_prompt
        batch = [constitution + prompt for prompt in batch]
        return self.language_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_args.max_length_rm,
        )

    def generate_completions(
        self,
        minibatch_size: int,
        superbatch_prompts: list[str],
        # accelerator: Accelerator,
    ) -> list[TensorType["batch", "seq_len"]]:
        """
        Generate completions for the prompts in the superbatch.

        Args:
            minibatch_size: The minibatch size to use for generation.
            superbatch_prompts: The prompts in the superbatch.
        """
        self.training_args.minibatch_size_generating = minibatch_size

        tqdm.write(f"Trying generation with batch size {minibatch_size}")
        print_gpu_utilization()

        completion_dataloader = DataLoader(
            ListDataset(superbatch_prompts),
            batch_size=minibatch_size,
            collate_fn=self.collate_fn_lm_completions,
        )

        completions: list[TensorType["batch", "seq_len"]] = []
        with torch.no_grad():
            for minibatch in tqdm(completion_dataloader, desc="Generation"):
                encodings = minibatch
                encodings.to(self.language_model.device)
                completions.extend(
                    self.language_model.generate(
                        **encodings,
                        max_length=self.training_args.max_new_tokens * 2,
                        max_new_tokens=self.training_args.max_new_tokens,
                        temperature=self.training_args.temperature,
                        top_p=self.training_args.top_p,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.language_tokenizer.pad_token_id,
                        logits_processor=self.training_args.logits_processors,
                    ).to("cpu")
                )
        # completions_gathered: list[str] = accelerator.gather(
        #     completions
        # )  # TODO Unclear whether this is needed?
        return completions

    def collate_fn_rm(
        self, batch: list[TensorType["batch", "seq_len"]]
    ) -> tuple[list[str], BatchEncoding]:
        """
        Collate function for the reward model's dataloader.

        Takes encoded completions from the language model, decodes them, encodes them for the
        reward model, and returns both the decoded completion text and re-encoded completions.
        """
        completions = self.language_tokenizer.batch_decode(
            batch, skip_special_tokens=True
        )

        # Remove completions after any extra "\n\nHuman:", "\n\nA:", "\n\nH:", or similar.
        # This is to prevent the model from learning to generate additional turns of conversation.
        prompts_and_completions = [
            separate_prompt_from_completion(completion) for completion in completions
        ]
        completions = [
            prompt + re.split(constants.PROMPT_DELIMITER_REGEX, completion)[0]
            for prompt, completion in prompts_and_completions
        ]

        return (
            completions,
            self.reward_tokenizer(
                completions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.training_args.max_length_rm,
            ).to(self.reward_model.device),
        )

    def score_completions(
        self,
        minibatch_size: int,
        completions_encoded: list[TensorType["batch", "seq_len"]],
    ) -> tuple[list[str], list[TensorType[1]]]:
        """
        Score the completions.

        Returns a tuple of the decoded completions and the scores.

        If using accelerate for this step, will need to update collate_fn_rm to not set device.
        """
        self.training_args.minibatch_size_scoring = minibatch_size
        all_completions: list[str] = []
        all_scores: list[TensorType[1]] = []

        score_dataloader = DataLoader(
            ListDataset(completions_encoded),
            batch_size=minibatch_size,
            collate_fn=self.collate_fn_rm,
        )

        with torch.no_grad():
            iteration = 0
            for minibatch in tqdm(score_dataloader, desc="Scoring"):
                iteration += 1
                completions, completion_encodings = minibatch
                scores = self.reward_model(**completion_encodings)
                all_completions.extend(completions)
                all_scores.extend(scores.logits.flatten().tolist())
        return all_completions, all_scores

    def collate_fn_lm_finetuning(self, batch: list[str]) -> BatchEncoding:
        """
        Collate function for the language model fine-tuning DataLoader.
        """
        encodings = self.language_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_args.max_length_rm,
        )
        encodings["labels"] = encodings["input_ids"].detach().clone()

        # Extract the prompt (the part before and including the first "\n\nAssistant:")
        # We only need the first example because of left-padding (the delimiter is aligned)
        delimiter = self.training_args.prompt_delimiter
        prompt = batch[0].split(delimiter)[0] + delimiter
        prompt_token_length = len(self.language_tokenizer(prompt).input_ids)

        # Set labels to -100 for tokens that should be ignored (non-completion part of the prompt)
        encodings["labels"][:, :prompt_token_length] = -100

        return encodings

    def finetune_language_model(
        self,
        minibatch_size: int,
        filtered_completions: list[str],
        # accelerator: Accelerator,
    ) -> float:
        """
        Fine-tune the language model on the completions.

        Returns the average loss for metrics.
        """
        # pylint: disable=too-many-locals

        accelerator = Accelerator(mixed_precision=self.training_args.mixed_precision)

        tqdm.write(f"Trying finetuning with batch size {minibatch_size}")
        print_gpu_utilization()
        self.training_args.minibatch_size_finetuning = minibatch_size

        finetuning_dataloader = DataLoader(
            ListDataset(filtered_completions),
            batch_size=minibatch_size,
            collate_fn=self.collate_fn_lm_finetuning,
        )

        # Initialize the optimizer
        # TODO: Shoulld this be moved to outside this function?
        optimizer = torch.optim.AdamW(
            self.language_model.parameters(), lr=self.training_args.learning_rate
        )
        self.language_model, optimizer, finetuning_dataloader = accelerator.prepare(
            self.language_model, optimizer, finetuning_dataloader
        )

        tqdm.write("After accelerator prepare, ", end="")
        print_gpu_utilization()
        sum_loss = 0
        self.language_model.train()
        for minibatch in tqdm(finetuning_dataloader, desc="Fine-tuning"):
            optimizer.zero_grad()
            outputs = self.language_model(**minibatch)
            if outputs.loss is None:
                raise ValueError("Loss is None on the outputs")

            loss = outputs.loss

            # Inverse loss penalty to regularize away from low-entropy states
            if self.training_args.inverse_loss_penalty > 0:
                loss = loss + self.training_args.inverse_loss_penalty / loss

            sum_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
        optimizer.zero_grad()

        return sum_loss / len(finetuning_dataloader)

    def save_model(self) -> None:
        """
        Save the model.
        """
        raise NotImplementedError
