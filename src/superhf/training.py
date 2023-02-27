"""
Trainers for finetuning models according to SuperHF (maximizing the scores
from a reward model with expert iteration using supervised learning).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator, find_executable_batch_size
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
    BatchEncoding,
    PreTrainedModel,
)
from torchtyping import TensorType

from superhf.data import ListDataset
from superhf.filtering import CompletionFilterBase
from superhf.metrics import SuperHFMetrics, report_metrics_print
from superhf.utils import print_gpu_utilization


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
    max_length_lm: int = 256
    max_length_rm: int = 1024

    # Batching to avoid OOM
    minibatch_size_initial: int = 64
    minibatch_size_generating: int = 64
    minibatch_size_scoring: int = 64
    minibatch_size_finetuning: int = 64

    # Training
    learning_rate: float = 1e-5
    mixed_precision: str = "no"


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
        # TODO try getting rid of first None in these
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

        # initallilze batch sizes
        (
            self.training_args.minibatch_size_finetuning,
            self.training_args.minibatch_size_generating,
            self.training_args.minibatch_size_scoring,
        ) = [self.training_args.minibatch_size_initial] * 3
        # Then, iterate over the prompts in superbatches
        for superbatch_index, superbatch_prompts in tqdm(
            enumerate(prompts_dataloader),
            total=len(prompts_dataloader),
            desc="Superbatch",
        ):
            print(f"Before generation, on superbatch_index {superbatch_index} ", end="")
            print_gpu_utilization()
            # Generate completions for each prompt in the superbatch
            completions_encoded = find_executable_batch_size(
                self.generate_completions,
                self.training_args.minibatch_size_generating,
            )(superbatch_prompts)

            print("Before scoring ", end="")
            print_gpu_utilization()
            # Score the completions
            completions, scores = find_executable_batch_size(
                self.score_completions,
                self.training_args.minibatch_size_scoring,
            )(completions_encoded)

            print("Before filtering ", end="")
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
                scheduler_lr=5e-5,  # TODO get this from the scheduler
            )
            if self.report_metrics is not None:
                for report_metrics_function in self.report_metrics:
                    report_metrics_function(metrics)

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
        ).to(self.language_model.device)

    def generate_completions(
        self, minibatch_size: int, superbatch_prompts: list[str]
    ) -> list[TensorType["batch", "seq_len"]]:
        """Generate completions for the prompts in the superbatch."""
        self.training_args.minibatch_size_generating = minibatch_size

        sampler = None
        if torch.cuda.device_count() > 1:
            self.language_model = torch.nn.DataParallel(self.language_model).module
            sampler = DistributedSampler(ListDataset(superbatch_prompts))

        completion_dataloader = DataLoader(
            ListDataset(superbatch_prompts),
            batch_size=minibatch_size,
            collate_fn=self.collate_fn_lm,
            sampler=sampler,
        )
        completions: list[str] = []
        with torch.no_grad():
            for minibatch in tqdm(completion_dataloader, desc="Generation"):
                encodings = minibatch
                completions.extend(
                    self.language_model.generate(
                        **encodings,
                        max_length=self.training_args.max_length_lm,
                        temperature=self.training_args.temperature,
                        top_p=self.training_args.top_p,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.language_tokenizer.pad_token_id,
                    ).to("cpu")
                )
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

        # TODO remove completions after any extra "\n\nHuman:", "\n\nA:", "\n\nH:", or similar.

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
            for minibatch in score_dataloader:
                completions, completion_encodings = minibatch
                scores = self.reward_model(**completion_encodings)
                all_completions.extend(completions)
                all_scores.extend(scores.logits.flatten().tolist())
        return all_completions, all_scores

    def finetune_language_model(
        self, minibatch_size: int, filtered_completions: list[str]
    ) -> float:
        """
        Fine-tune the language model on the completions.

        Returns the average loss for metrics.
        """
        print(f"Trying with batch size {minibatch_size}")
        print_gpu_utilization()
        self.training_args.minibatch_size_finetuning = minibatch_size

        loss_function = torch.nn.CrossEntropyLoss()

        finetuning_dataloader = DataLoader(
            ListDataset(filtered_completions),
            batch_size=minibatch_size,
            collate_fn=self.collate_fn_lm,
        )

        # Initialize the accelerator
        accelerator = Accelerator(mixed_precision="fp16")

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            self.language_model.parameters(), lr=self.training_args.learning_rate
        )

        self.language_model, optimizer, finetuning_dataloader = accelerator.prepare(
            self.language_model, optimizer, finetuning_dataloader
        )
        print("After accelerator prepare, memory usage is: ", end="")
        print_gpu_utilization()
        average_loss = 0
        self.language_model.train()
        for minibatch in tqdm(finetuning_dataloader, desc="Fine-tuning"):
            encodings = minibatch  # Encodings have keys dict_keys(['input_ids', 'attention_mask'])
            # input_ids have shape [completion_filter_top_k, seq_len]
            targets_flat = encodings["input_ids"].view(
                -1
            )  # TODO: Do I need to shift the targets or logits?

            outputs = self.language_model(**encodings)
            # outputs contains odict_keys(['logits', 'past_key_values'])
            logits_flat = outputs.logits.view(
                -1, outputs.logits.shape[-1]
            )  # [completion_filter_top_k * seq_len, |V|]

            # outputs.logits have shape [completion_filter_top_k, seq_len, |V|]

            # Mask out positions with padding
            padding_mask_flat = encodings["attention_mask"].view(-1)
            targets_flat = targets_flat[padding_mask_flat]
            logits_flat = logits_flat[padding_mask_flat]

            loss = loss_function(logits_flat, targets_flat)  # is a scalar on gpu device
            # assert loss.item() > 0.0, f"Loss is {loss.item()}, which is not positive."
            #  TODO add this assertion
            average_loss += loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        return average_loss / len(
            finetuning_dataloader
        )  # TODO: Figure out the correct denominator

    def save_model(self) -> None:
        """
        Save the model.
        """
        raise NotImplementedError
