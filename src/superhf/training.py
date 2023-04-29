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
    get_scheduler,
)
from torchtyping import TensorType
from peft import PeftModel

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
    top_p: float = 0.95
    superbatch_size: int = field(
        default=32,
        metadata={
            "help": (
                "Number of completions to generate with the current "
                "policy before filtering and fine-tuning."
            )
        },
    )
    prompt_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of prompts to generate, score, and filter before "
                "fine-tuning (0 for all the prompts). Used to blend between "
                "iterative and single-pass training."
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
    inverse_loss_penalty: float = 0.0
    mixed_precision: str = "no"
    dtype: torch.dtype = torch.float32
    learning_rate: float = 1e-5
    scheduler_name: str = "linear"
    scheduler_warmup_steps: int = 0
    kl_coefficient: float = 0.0
    validation_interval: int = 0

    # Dataset settings
    prompt_delimiter: str = constants.PROMPT_DELIMITER

    # Reward shaping
    length_penalty: float = 0.0

    # Reward model settings
    reward_model_is_steamshp: bool = False

    # Push to hub (set to 0 to disable)
    hub_repo_id: Optional[str] = None
    push_to_hub_interval: int = 0


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

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        language_model: PreTrainedModel,
        reward_model_train: PreTrainedModel,
        reward_model_val: Optional[PreTrainedModel],
        language_tokenizer: PreTrainedTokenizerBase,
        reward_tokenizer_train: PreTrainedTokenizerBase,
        reward_tokenizer_val: Optional[PreTrainedTokenizerBase],
        completion_filter: CompletionFilterBase,
        training_args: SuperHFTrainingArguments,
        report_metrics: Optional[
            Union[
                Callable[[SuperHFMetrics], None], list[Callable[[SuperHFMetrics], None]]
            ]
        ] = None,
    ) -> None:
        self.language_model = language_model
        self.reward_model_train = reward_model_train
        self.reward_model_val = reward_model_val
        self.language_tokenizer = language_tokenizer
        self.reward_tokenizer_train = reward_tokenizer_train
        self.reward_tokenizer_val = reward_tokenizer_val
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
        if self.reward_tokenizer_train.pad_token is None:
            self.reward_tokenizer_train.pad_token = (
                self.reward_tokenizer_train.eos_token
            )
            print("Added pad token to reward tokenizer.")

        # Reward model is always in eval mode
        self.reward_model_train.eval()

        # Initialize the accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.training_args.mixed_precision
        )

        # Prepare with accelerator
        self.language_model = self.accelerator.prepare(self.language_model)

        # Lazy-init optimizer and scheduler
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None

        # Check that we're using a LoRA model if using a KL loss term
        if self.training_args.kl_coefficient > 0:
            assert isinstance(
                self.language_model, PeftModel
            ), "KL loss term only supported for LoRA models."
            self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def train(self, prompts: list[str]) -> None:
        """
        Main training and evaluation loop.
        """
        # pylint: disable=too-many-locals

        # FIXME hack
        # self.training_args.prompt_accumulation_steps = 16

        # First, put all the prompts into a Dataset and DataLoader
        prompt_batch_size = self.training_args.prompt_accumulation_steps
        if prompt_batch_size == 0:
            prompt_batch_size = len(prompts)
        prompts_dataloader = DataLoader(
            ListDataset(prompts),
            batch_size=prompt_batch_size,
        )
        num_superbatches = len(prompts_dataloader)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.language_model.parameters(), lr=self.training_args.learning_rate
        )
        self.scheduler = get_scheduler(
            self.training_args.scheduler_name,
            self.optimizer,
            num_warmup_steps=self.training_args.scheduler_warmup_steps,
            num_training_steps=num_superbatches,
        )
        self.optimizer, self.scheduler = self.accelerator.prepare(
            self.optimizer, self.scheduler
        )
        assert self.scheduler is not None

        # Then, iterate over group of prompts in superbatches
        for superbatch_index, superbatch_prompts in tqdm(
            enumerate(prompts_dataloader),
            total=num_superbatches,
            desc="Superbatch",
        ):
            tqdm.write(
                f"Before generation, on superbatch_index {superbatch_index} ", end=""
            )
            print_gpu_utilization()
            # Generate completions for each prompt in the superbatch
            completions_raw = find_executable_batch_size(
                self.generate_completions,
                self.training_args.minibatch_size_generating,
            )(superbatch_prompts)

            tqdm.write("Before scoring ", end="")
            print_gpu_utilization()
            # Score the completions
            try:
                (
                    scores_train,
                    completions_trimmed,
                    completion_lengths,
                ) = find_executable_batch_size(
                    self.score_completions_train,
                    self.training_args.minibatch_size_scoring,
                )(
                    completions_raw
                )

                if (
                    superbatch_index % self.training_args.validation_interval == 0
                    and self.reward_model_val is not None
                    and self.reward_tokenizer_val is not None
                ):
                    # Score the completions with the validation reward model
                    scores_val = self.score_completions_val(completions_trimmed)
                else:
                    scores_val = []
            except (IndexError, KeyError) as exc:
                print("Error during scoring completions:")
                print(exc)
                print("completions_raw:")
                print(completions_raw)
                continue

            tqdm.write("Before filtering ", end="")
            print_gpu_utilization()

            # Filter the completions
            (
                filtered_scores,
                filtered_completions,
                filtered_completion_lengths,
            ) = self.filter_completions(
                len(superbatch_prompts),
                scores_train,
                completions_trimmed,
                completion_lengths,
            )

            # Fine-tune the language model on the filtered completions
            average_loss, average_kl_div = find_executable_batch_size(
                self.finetune_language_model,
                self.training_args.minibatch_size_finetuning,
            )(filtered_completions)

            # Optionally report metrics
            metrics = SuperHFMetrics(
                superbatch_index=superbatch_index,
                superbatch_count=num_superbatches,
                completions=completions_trimmed,
                filtered_completions=filtered_completions,
                scores_train=scores_train,
                scores_val=scores_val,
                filtered_scores=filtered_scores,
                average_loss=average_loss,
                average_kl_div=average_kl_div,
                scheduler_lr=self.scheduler.get_last_lr()[0],
                completion_lengths=completion_lengths,
                filtered_completion_lengths=filtered_completion_lengths,
            )
            if self.report_metrics is not None:
                for report_metrics_function in self.report_metrics:
                    report_metrics_function(metrics)

            # Optionally, save the model
            # self.save_model()

            # Optionally, push the model to the hub
            self.consider_pushing_to_hub(superbatch_index, num_superbatches)

    def consider_pushing_to_hub(self, superbatch_index: int, num_prompts: int) -> None:
        """Pushes the model to the hub if it's appropriate to do so."""
        if (  # pylint: disable=too-many-boolean-expressions
            # User must specify a hub repo
            self.training_args.hub_repo_id is not None
            and self.training_args.hub_repo_id != ""
            # User must specify a push interval
            and self.training_args.push_to_hub_interval > 0
            # Don't push on the first superbatch
            and superbatch_index > 0
            and (
                # every N superbatches
                superbatch_index % self.training_args.push_to_hub_interval == 0
                # last superbatch
                or superbatch_index == num_prompts - 1
            )
        ):
            tqdm.write("Pushing model and tokenizer to the Hub!")
            tqdm.write(
                str(
                    self.language_model.push_to_hub(
                        repo_id=self.training_args.hub_repo_id,
                        commit_message=(
                            f"Upload model from superbatch {superbatch_index}"
                        ),
                    )
                )
            )
            tqdm.write(
                str(
                    self.language_tokenizer.push_to_hub(
                        repo_id=self.training_args.hub_repo_id,
                        commit_message=(
                            f"Upload tokenizer from superbatch {superbatch_index}"
                        ),
                    )
                )
            )

    def collate_fn_lm_completions(self, batch: list[str]) -> BatchEncoding:
        """
        Collate function for the language model completions DataLoader.

        Prepends the system prompt to each prompt. By default this is the empty string.
        """
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
    ) -> list[str]:
        """
        Generate completions for the prompts in the superbatch.

        Args:
            minibatch_size: The minibatch size to use for generation.
            superbatch_prompts: The prompts in the superbatch.
        """
        self.training_args.minibatch_size_generating = minibatch_size

        tqdm.write(f"Trying generation with batch size {minibatch_size}")
        print_gpu_utilization()

        # Duplicate each prompt superbatch_size numbers time with system prompt
        system_prompt = self.training_args.conversation_prompt
        prompt_batch_duplicated = [
            system_prompt + prompt
            for prompt in superbatch_prompts
            for _ in range(self.training_args.superbatch_size)
        ]

        completion_dataloader = DataLoader(
            ListDataset(prompt_batch_duplicated),
            batch_size=minibatch_size,
            collate_fn=self.collate_fn_lm_completions,
            pin_memory=True,
        )

        completions_encoded: list[TensorType["batch", "seq_len"]] = []
        with torch.no_grad():
            for minibatch in tqdm(
                completion_dataloader,
                desc="Generation",
                total=len(completion_dataloader),
            ):
                encodings = minibatch
                # with torch.cuda.amp.autocast(dtype=self.training_args.dtype):  # type: ignore
                encodings.to(self.language_model.device)
                outputs = self.language_model.generate(  # type: ignore
                    **encodings,
                    max_new_tokens=self.training_args.max_new_tokens,
                    temperature=self.training_args.temperature,
                    top_p=self.training_args.top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.language_tokenizer.pad_token_id,
                    logits_processor=self.training_args.logits_processors,
                )
                completions_encoded.extend(outputs.to("cpu"))
        # completions_gathered: list[str] = accelerator.gather(
        #     completions
        # )  # TODO Unclear whether this is needed?
        completions_text: list[str] = self.language_tokenizer.batch_decode(
            completions_encoded, skip_special_tokens=True
        )
        return completions_text

    def collate_fn_rm_train(
        self, completions: list[str]
    ) -> tuple[list[str], BatchEncoding, list[int]]:
        """
        Collate function for the reward model's dataloader.

        Takes encoded completions from the language model, decodes them, encodes them for the
        reward model, and returns both the decoded completion text and re-encoded completions.
        """

        # Remove completions after any extra "\n\nHuman:", "\n\nA:", "\n\nH:", or similar.
        # This is to prevent the model from learning to generate additional turns of conversation.
        prompts_and_completions = [
            separate_prompt_from_completion(completion) for completion in completions
        ]
        completions_for_lm: list[str] = []
        completions_for_rm: list[str] = []
        completion_lengths: list[int] = []
        for prompt, completion in prompts_and_completions:
            stripped_completion = re.split(
                constants.PROMPT_DELIMITER_REGEX_MEDIUM, completion, maxsplit=1
            )[0].strip()
            completion_lengths.append(len(stripped_completion))
            joined_completion_normal = prompt + " " + stripped_completion
            completions_for_lm.append(joined_completion_normal)
            if self.training_args.reward_model_is_steamshp:
                # Handle the weird SteamSHP format
                prompt_only = prompt.split(constants.HUMAN_DELIMITER)[1].split(
                    constants.PROMPT_DELIMITER
                )[0]
                joined_completion_shp = (
                    f"POST:{prompt_only}\n\n"
                    f" RESPONSE A: {stripped_completion}\n\n RESPONSE B: .\n\n Which"
                    " response is better? RESPONSE"
                )
                completions_for_rm.append(joined_completion_shp)
            else:
                # Concat normally
                completions_for_rm.append(joined_completion_normal)

        return (
            completions_for_lm,
            self.reward_tokenizer_train(
                completions_for_rm,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.training_args.max_length_rm,
            ).to(self.reward_model_train.device),
            completion_lengths,
        )

    def collate_fn_rm_val(self, completions: list[str]) -> BatchEncoding:
        """
        Collate function for the reward model's dataloader.

        Simply tokenizes the completions for the val reward model.
        """
        assert self.reward_model_val is not None
        assert self.reward_tokenizer_val is not None
        return self.reward_tokenizer_val(
            completions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_args.max_length_rm,
        ).to(self.reward_model_val.device)

    def score_completions_train(
        self,
        minibatch_size: int,
        completions_encoded: list[TensorType["batch", "seq_len"]],
    ) -> tuple[list[float], list[str], list[int]]:
        """
        Score the completions.

        Returns a tuple of the scores and the lengths of just the LM-generated completion parts.

        If using accelerate for this step, will need to update collate_fn_rm to not set device.
        """
        self.training_args.minibatch_size_scoring = minibatch_size
        all_scores: list[float] = []
        all_completions_trimmed: list[str] = []
        all_completion_lengths: list[int] = []

        score_dataloader = DataLoader(
            ListDataset(completions_encoded),
            batch_size=minibatch_size,
            collate_fn=self.collate_fn_rm_train,
        )

        with torch.no_grad():
            iteration = 0
            for minibatch in tqdm(score_dataloader, desc="Scoring"):
                iteration += 1
                (
                    completions_trimmed,
                    completion_encodings,
                    completion_lengths,
                ) = minibatch
                if self.training_args.reward_model_is_steamshp:
                    # Handle the weird SteamSHP format
                    outputs = self.reward_model_train.generate(
                        **completion_encodings,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=1,
                    )
                    # index 71 corresponds to the token for 'A'
                    scores = (
                        torch.softmax(outputs.scores[0], dim=1)[:, 71]  # type: ignore
                        .flatten()
                        .cpu()
                    )
                else:
                    scores = self.reward_model_train(**completion_encodings)
                    if not isinstance(scores, torch.Tensor):
                        # Handle SequenceClassifierOutput
                        scores = scores.logits
                    scores = scores.flatten().cpu()  # type: ignore
                if self.training_args.length_penalty != 0.0:
                    # Add -length_penalty * char_length to penalize long completions.
                    scores -= self.training_args.length_penalty * torch.log(
                        torch.tensor(completion_lengths)
                    )
                all_scores.extend(scores.tolist())
                all_completions_trimmed.extend(completions_trimmed)
                all_completion_lengths.extend(completion_lengths)
        return all_scores, all_completions_trimmed, all_completion_lengths

    def score_completions_val(
        self,
        trimmed_completions: list[str],
    ) -> list[float]:
        """Use the validation reward mode to score the trimmed completions."""
        assert self.reward_model_val is not None

        all_scores: list[float] = []
        score_dataloader = DataLoader(
            ListDataset(trimmed_completions),
            batch_size=self.training_args.minibatch_size_scoring,
            collate_fn=self.collate_fn_rm_val,
        )
        with torch.no_grad():
            for minibatch in score_dataloader:
                completion_encodings = minibatch
                scores = self.reward_model_val(**completion_encodings)
                if not isinstance(scores, torch.Tensor):
                    # Handle SequenceClassifierOutput
                    scores = scores.logits
                scores = scores.flatten().cpu()
                all_scores.extend(scores.tolist())
        return all_scores

    def filter_completions(
        self,
        num_distinct_prompts: int,
        scores: list[float],
        completions_trimmed: list[str],
        completion_lengths: list[int],
    ) -> tuple[list[float], list[str], list[int]]:
        """Filter the completions and their lengths based on the scores."""
        # pylint: disable=too-many-locals

        filtered_scores: list[float] = []
        filtered_completions: list[str] = []
        filtered_completion_lengths: list[int] = []
        for i in range(num_distinct_prompts):
            start = i * self.training_args.superbatch_size
            end = (i + 1) * self.training_args.superbatch_size

            # Remove any 0-length completions
            (scores_i, completions_trimmed_i, completion_lengths_i) = zip(
                *[
                    (score, completion, length)
                    for score, completion, length in zip(
                        scores[start:end],
                        completions_trimmed[start:end],
                        completion_lengths[start:end],
                    )
                    if length > 0
                ]
            )

            # Filter the completions
            (
                filtered_scores_i,
                (filtered_completions_i, filtered_completion_lengths_i),
            ) = self.completion_filter.filter(
                scores_i,  # type: ignore
                completions_trimmed_i,  # type: ignore
                completion_lengths_i,  # type: ignore
            )
            filtered_scores += filtered_scores_i
            filtered_completions += filtered_completions_i
            filtered_completion_lengths += filtered_completion_lengths_i
        return filtered_scores, filtered_completions, filtered_completion_lengths

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
        encodings["labels"] = encodings["input_ids"].detach().clone()  # type: ignore

        # Extract the prompt (the part before and including the first "\n\nAssistant:")
        delimiter = self.training_args.prompt_delimiter
        prompts = [example.split(delimiter)[0] + delimiter for example in batch]
        prompt_token_lengths = [
            len(tokenized) for tokenized in self.language_tokenizer(prompts).input_ids
        ]

        # Set labels to -100 for tokens that should be ignored (non-completion part of the prompt)
        for i, length in enumerate(prompt_token_lengths):
            encodings["labels"][i, :length] = -100

        return encodings

    def finetune_language_model(
        self,
        minibatch_size: int,
        filtered_completions: list[str],
    ) -> tuple[float, float]:
        """
        Fine-tune the language model on the completions.

        Returns the average loss and KL divergence (or 0) for metrics.
        """
        # pylint: disable=too-many-locals

        assert self.optimizer is not None
        assert self.scheduler is not None

        tqdm.write(f"Trying finetuning with batch size {minibatch_size}")
        print_gpu_utilization()
        self.training_args.minibatch_size_finetuning = minibatch_size

        finetuning_dataloader = DataLoader(
            ListDataset(filtered_completions),
            batch_size=minibatch_size,
            collate_fn=self.collate_fn_lm_finetuning,
        )

        finetuning_dataloader = self.accelerator.prepare(finetuning_dataloader)

        tqdm.write("After accelerator prepare, ", end="")
        print_gpu_utilization()
        sum_loss = 0
        num_invalid_losses = 0
        self.language_model.train()

        sum_kl_divergence = 0
        for minibatch in tqdm(finetuning_dataloader, desc="Fine-tuning"):
            self.optimizer.zero_grad()
            outputs = self.language_model(**minibatch)
            if outputs.loss is None:
                raise ValueError("Loss is None on the outputs")

            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                num_invalid_losses += 1
                continue

            # Inverse loss penalty to regularize away from low-entropy states
            if self.training_args.inverse_loss_penalty > 0:
                loss = loss + self.training_args.inverse_loss_penalty / loss

            # KL divergence penalty
            if self.training_args.kl_coefficient > 0:
                # Calculate the log probabilities of the generated tokens
                logp_online_model = torch.log_softmax(outputs.logits, dim=2)

                # Disable LoRA adapters (required, otherwise high memory)
                with self.language_model.disable_adapter(), torch.no_grad():  # type: ignore
                    # Get the log probabilities from the original model
                    try:
                        logp_original_model = self.language_model(**minibatch)
                        logp_original_model = torch.log_softmax(
                            logp_original_model.logits, dim=2
                        )
                    except Exception as exc:
                        # Hack to fix https://github.com/huggingface/peft/issues/367 until merged.
                        # Manually fix the peft context adapter not re-enabling the adapter.
                        self.language_model.base_model.enable_adapter_layers()
                        raise exc

                # Truncate each to just the part that was generated (after where labels == -100)
                # We have to iterate over each fine-tune example because the lengths may differ
                for i in range(logp_online_model.shape[0]):
                    mask = minibatch["labels"][i] != -100
                    logp_online_model_i = logp_online_model[i, mask, :]
                    logp_original_model_i = logp_original_model[i, mask, :]

                    # Compute the KL divergence
                    kl_divergence = self.kl_loss(
                        logp_online_model_i, logp_original_model_i
                    )
                    # Clamp KL to be positive to avoid negative KL gaming
                    kl_divergence = torch.maximum(kl_divergence, torch.tensor(0.0))
                    sum_kl_divergence += kl_divergence.item()
                    loss = loss + self.training_args.kl_coefficient * kl_divergence

            sum_loss += loss.item()
            self.accelerator.backward(loss)
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        if num_invalid_losses > 0:
            tqdm.write(
                f"WARNING: {num_invalid_losses} minibatches had nan, inf, or negative"
                " loss."
            )

        num_valid_losses = len(finetuning_dataloader) - num_invalid_losses
        return sum_loss / num_valid_losses if num_valid_losses > 0 else 0, (
            sum_kl_divergence / num_valid_losses if num_valid_losses > 0 else 0
        )

    def save_model(self) -> None:
        """
        Save the model.
        """
        raise NotImplementedError
