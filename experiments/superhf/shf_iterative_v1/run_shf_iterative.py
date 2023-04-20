"""
Client code showing how to call the training loop for the iterative version of the SuperHF model.
"""

import argparse
import random
import os
import yaml

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
import torch
import wandb

from superhf.data import get_superhf_prompts
from superhf.filtering import CompletionFilterTopK
from superhf.metrics import (
    initialize_metrics_wandb,
    report_metrics_wandb,
    report_metrics_print,
    delay_metrics,
)
from superhf.mocking import MockLanguageModel, MockRewardModel
from superhf.training import SuperHFTrainingArguments, SuperHFTrainer
from superhf.utils import set_seed, print_gpu_utilization

WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "superhf-v3"


def main(argparse_args: argparse.Namespace) -> None:
    """
    Instantiate and train the SuperHF model.
    """
    # pylint: disable=too-many-locals

    # Configure device and seed
    device = torch.device(
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    )
    set_seed(66)

    # Initialize Weights and Biases run
    wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        notes=argparse_args.notes,
        save_code=True,
        config=argparse_args.config,
    )
    language_model_name = wandb.config.language_model_name
    reward_model_name = wandb.config.reward_model_name

    # Get the prompt dataset
    prompts: list[str] = []
    for dataset_name in wandb.config.prompt_dataset_names:
        prompts.extend(get_superhf_prompts(dataset_name))
    random.shuffle(prompts)

    # Duplicate each prompt so that each superbatch is the same prompt if desired
    if wandb.config.same_prompt_per_superbatch:
        new_prompts = []
        for prompt in prompts:
            new_prompts.extend([prompt] * wandb.config.superbatch_size)
        prompts = new_prompts

    # Filter out prompts that are too long
    old_prompt_count = len(prompts)
    prompts = [
        prompt
        for prompt in prompts
        if len(prompt) < wandb.config.max_prompt_char_length
    ]
    print(
        f"Filtered {old_prompt_count - len(prompts)} prompts over "
        f"{wandb.config.max_prompt_char_length} chars."
    )

    # Only load the first section of prompts
    if wandb.config.debug_max_prompts != 0:
        prompts = prompts[: wandb.config.debug_max_prompts]

    print(f"Loaded {len(prompts)} prompts.")
    print_gpu_utilization()
    print("Instantiating models...")
    # Instantiate our language and reward models and tokenizers
    dtype = (
        torch.float16
        if wandb.config.mixed_precision == "fp16"
        else torch.bfloat16
        if wandb.config.mixed_precision == "bf16"
        else torch.float32
    )
    language_model = (
        MockLanguageModel()
        if language_model_name == "mock"
        else AutoModelForCausalLM.from_pretrained(
            language_model_name, torch_dtype=dtype
        ).to(device)
    )
    print(f"Instantiated language model: {language_model_name}")
    print_gpu_utilization()
    reward_model = (
        MockRewardModel()
        if reward_model_name == "mock"
        else (
            AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(
                device
            )
            if "SteamSHP-flan-t5" not in reward_model_name
            else AutoModelForSeq2SeqLM.from_pretrained(reward_model_name).to(device)
        )
    )
    print(f"Instantiated reward model: {reward_model_name}")
    print_gpu_utilization()

    language_tokenizer_name = (
        "gpt2"
        if wandb.config.language_model_name == "mock"
        else wandb.config.language_model_name
    )
    language_tokenizer = AutoTokenizer.from_pretrained(
        language_tokenizer_name, padding_side="left"
    )
    reward_tokenizer_name = (
        "gpt2"
        if wandb.config.reward_model_name == "mock"
        else wandb.config.reward_model_name
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_tokenizer_name)
    print("Instantiated tokenizers.")
    print_gpu_utilization()

    # Set our training arguments
    print("Setting up trainer...")
    logits_processors = LogitsProcessorList()
    if wandb.config.no_repeat_ngram_size > 0:
        logits_processors.append(
            NoRepeatNGramLogitsProcessor(wandb.config.no_repeat_ngram_size)
        )
    if wandb.config.repetition_penalty > 1.0:
        logits_processors.append(
            RepetitionPenaltyLogitsProcessor(wandb.config.repetition_penalty)
        )
    training_args = SuperHFTrainingArguments(
        temperature=wandb.config.temperature,
        top_p=wandb.config.top_p,
        superbatch_size=wandb.config.superbatch_size,
        max_new_tokens=wandb.config.max_new_tokens,
        max_length_rm=wandb.config.max_length_rm,
        minibatch_size_generating=wandb.config.minibatch_size_generating,
        minibatch_size_scoring=wandb.config.minibatch_size_scoring,
        minibatch_size_finetuning=wandb.config.minibatch_size_finetuning,
        mixed_precision=wandb.config.mixed_precision,
        conversation_prompt=wandb.config.conversation_prompt,
        logits_processors=logits_processors,
        learning_rate=wandb.config.learning_rate,
        scheduler_name=wandb.config.scheduler_name,
        scheduler_warmup_steps=wandb.config.scheduler_warmup_steps,
        inverse_loss_penalty=wandb.config.inverse_loss_penalty,
        reward_model_is_steamshp=("SteamSHP" in wandb.config.reward_model_name),
        length_penalty=wandb.config.length_penalty,
        hub_repo_id=wandb.config.hub_repo_id,
        push_to_hub_interval=wandb.config.push_to_hub_interval,
    )
    completion_filter_top_k = wandb.config.completion_filter_top_k
    completion_filter = CompletionFilterTopK(completion_filter_top_k)
    metrics_functions = (
        [report_metrics_wandb, report_metrics_print, delay_metrics]
        if wandb.config.language_model_name == "mock"
        else [report_metrics_wandb, report_metrics_print]
    )

    # Instantiate our trainer
    trainer = SuperHFTrainer(
        language_model=language_model,  # type: ignore
        reward_model=reward_model,  # type: ignore
        language_tokenizer=language_tokenizer,
        reward_tokenizer=reward_tokenizer,
        completion_filter=completion_filter,
        training_args=training_args,
        report_metrics=metrics_functions,
    )

    # Initialize metrics
    print("Initializing metrics...")
    initialize_metrics_wandb()  # Defines the run metrics
    wandb.watch(language_model, log="all")

    # Run training
    wandb.alert(title="Beginning SuperHF run", text="Beginning SuperHF run...")
    trainer.train(prompts)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        # Get file path relative to this file
        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
        help="The name of the Weights and Biases config to use.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Notes to add to the Weights and Biases run.",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="",
        help=(
            "If specified, path to a yaml file to use to for a sweep. See"
            " ../sweeps/params.yaml"
        ),
    )
    args = parser.parse_args()

    if args.sweep != "":
        # Run sweeps
        with open(args.sweep, encoding="utf-8") as f:
            sweep_params = yaml.load(f, Loader=yaml.FullLoader)
        wandb.agent(
            sweep_params["id"],
            function=lambda: main(args),
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            count=sweep_params["count"],
        )
    else:
        main(args)
