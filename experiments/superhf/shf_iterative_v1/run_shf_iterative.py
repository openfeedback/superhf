"""
Client code showing how to call the training loop for the iterative version of the SuperHF model.
"""

import argparse
import random
import os
from typing import Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    LlamaTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
import torch
import yaml
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
from superhf.utils import set_seed, print_memory_utilization
from reward_modelling.reward_model import RewardModel

WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "superhf-v4"


def main(argparse_args: argparse.Namespace, extra_args: list[str]) -> None:
    """
    Instantiate and train the SuperHF model.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches

    print_memory_utilization()

    # Attempt to fix too many open files issue on SLURM
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Enable tf32 training if supported
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("Enabling tf32 training.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize Weights and Biases run
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        notes=argparse_args.notes,
        save_code=True,
        config=argparse_args.config,
    )
    assert run is not None

    # Configure seed
    set_seed(wandb.config.seed)

    # Process any extra arguments, converting values to appropriate types
    extra_args_dict = {}
    for arg in extra_args:
        value: Any
        key, value = arg.split("=")
        if value == "True":
            value = True
        elif value == "False":
            value = False
        elif "." in value:
            value = float(value)
        elif value.isdigit():
            value = int(value)
        extra_args_dict[key] = value
    wandb.config.update(extra_args_dict, allow_val_change=True)

    # Get the prompt dataset
    prompts: list[str] = []
    num_datasets = len(wandb.config.prompt_dataset_names)
    num_prompts_per_dataset = (
        wandb.config.num_prompts // num_datasets if wandb.config.num_prompts > 0 else 0
    )
    for dataset_name in wandb.config.prompt_dataset_names:
        new_prompts = get_superhf_prompts(
            dataset_name, max_length_chars=wandb.config.max_prompt_char_length
        )
        total_loaded = len(new_prompts)
        if num_prompts_per_dataset != 0:
            new_prompts = new_prompts[:num_prompts_per_dataset]
        total_filtered = len(new_prompts)
        print(f"Loaded {total_filtered}/{total_loaded} prompts from {dataset_name}.")
        prompts.extend(new_prompts)
    random.shuffle(prompts)

    print(f"Loaded {len(prompts)} total prompts.")
    print_memory_utilization()
    print("Instantiating models...")

    # Instantiate our language and reward models and tokenizers
    language_model_name = wandb.config.language_model_name
    reward_model_train_name = wandb.config.reward_model_train_name
    reward_model_val_name = wandb.config.reward_model_val_name
    reward_tokenizer_train_name = wandb.config.reward_model_train_name
    reward_tokenizer_val_name = wandb.config.reward_model_val_name

    language_model = load_language_model(language_model_name)
    print_memory_utilization()

    reward_model_train = load_reward_model(reward_model_train_name)
    print_memory_utilization()

    if reward_model_val_name != "" and reward_model_val_name.lower() != "none":
        reward_model_val = load_reward_model(reward_model_val_name)
        print_memory_utilization()
    else:
        print("No validation reward model specified.")
        reward_model_val = None

    language_tokenizer = load_language_tokenizer(language_model_name)
    reward_tokenizer_train = load_reward_tokenizer(reward_tokenizer_train_name)
    if reward_model_val_name != "" and reward_model_val_name.lower() != "none":
        reward_tokenizer_val = load_reward_tokenizer(reward_tokenizer_val_name)
    else:
        reward_tokenizer_val = None
    print_memory_utilization()

    # # Check for unix before compiling models
    # if os.name == "posix":
    #     print("Compiling models...")
    #     print(type(language_model))
    #     language_model = torch.compile(language_model)
    #     print(type(language_model))
    #     reward_model_train = torch.compile(reward_model_train)
    #     reward_model_val = torch.compile(reward_model_val)
    #     print("Compiled models.")
    #     print_gpu_utilization()

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
    dtype = (
        torch.float16
        if wandb.config.mixed_precision == "fp16"
        else torch.bfloat16 if wandb.config.mixed_precision == "bf16" else torch.float32
    )
    training_args = SuperHFTrainingArguments(
        seed=wandb.config.seed,
        temperature=wandb.config.temperature,
        top_p=wandb.config.top_p,
        superbatch_size=wandb.config.superbatch_size,
        prompt_accumulation_steps=wandb.config.prompt_accumulation_steps,
        max_new_tokens=wandb.config.max_new_tokens,
        max_length_rm=wandb.config.max_length_rm,
        minibatch_size_generating=wandb.config.minibatch_size_generating,
        minibatch_size_scoring=wandb.config.minibatch_size_scoring,
        minibatch_size_finetuning=wandb.config.minibatch_size_finetuning,
        mixed_precision=wandb.config.mixed_precision,
        dtype=dtype,
        conversation_prompt=wandb.config.conversation_prompt,
        logits_processors=logits_processors,
        learning_rate=wandb.config.learning_rate,
        scheduler_name=wandb.config.scheduler_name,
        scheduler_warmup_steps=wandb.config.scheduler_warmup_steps,
        inverse_loss_penalty=wandb.config.inverse_loss_penalty,
        kl_coefficient=wandb.config.kl_coefficient,
        validation_interval=wandb.config.validation_interval,
        max_exception_count=wandb.config.max_exception_count,
        reward_model_is_steamshp=("SteamSHP" in reward_model_train_name),
        length_penalty=wandb.config.length_penalty,
        hub_repo_id=wandb.config.hub_repo_id,
        push_to_hub_interval=wandb.config.push_to_hub_interval,
        push_to_hub_additional_indices=wandb.config.push_to_hub_additional_indices,
        sweep_param_name=wandb.config.sweep_param_name,
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
        language_model=language_model,
        reward_model_train=reward_model_train,
        reward_model_val=reward_model_val,
        language_tokenizer=language_tokenizer,
        reward_tokenizer_train=reward_tokenizer_train,
        reward_tokenizer_val=reward_tokenizer_val,
        completion_filter=completion_filter,
        training_args=training_args,
        report_metrics=metrics_functions,
    )

    # Initialize metrics
    print("Initializing metrics...")
    initialize_metrics_wandb()  # Defines the run metrics
    # wandb.watch(language_model, log="all")

    # Run training
    wandb.alert(title="Beginning SuperHF run", text="Beginning SuperHF run...")
    trainer.train(prompts)

    # Explicit finish to avoid wandb hanging
    wandb.alert(
        title="FINISHED SuperHF run!", text="FINISHED SuperHF run! <@WGPFRK13K>"
    )
    wandb.finish()


def load_language_model(language_model_name: str) -> PreTrainedModel:
    """Load the language model."""
    language_model = (
        MockLanguageModel()
        if language_model_name == "mock"
        else AutoModelForCausalLM.from_pretrained(
            language_model_name,
            low_cpu_mem_usage=True,
        )
    )
    if wandb.config.lora_r != 0 and wandb.config.lora_alpha != 0:
        # Set up low-rank adapters (LoRA)
        lora_config = LoraConfig(
            r=wandb.config.lora_r,
            lora_alpha=wandb.config.lora_alpha,
            lora_dropout=wandb.config.lora_dropout,
            target_modules=wandb.config.lora_target_modules,
            task_type="CAUSAL_LM",
            fan_in_fan_out=False,
        )
        language_model = get_peft_model(language_model, lora_config)
        language_model.print_trainable_parameters()

    print(f"Instantiated language model: {language_model_name}")
    return language_model


def load_reward_model(reward_model_name: str) -> PreTrainedModel:
    """Load the reward model."""
    if reward_model_name == "mock":
        reward_model_train = MockRewardModel()
    elif "rm_combined" in reward_model_name or "oliversssf2" in reward_model_name:
        reward_model_train = RewardModel.from_pretrained(
            reward_model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,  # Force half for these large RMs
        )
    elif "SteamSHP-flan-t5" in reward_model_name:
        reward_model_train = AutoModelForSeq2SeqLM.from_pretrained(reward_model_name)
    else:
        reward_model_train = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )

    print(f"Instantiated reward model: {reward_model_name}")
    return reward_model_train


def load_language_tokenizer(language_model_name: str) -> PreTrainedTokenizerBase:
    """Load the language model tokenizer."""
    language_tokenizer_name = (
        "gpt2" if language_model_name == "mock" else language_model_name
    )
    if "llama" in language_tokenizer_name or "alpaca" in language_tokenizer_name:
        # Fix for misnamed class in the NLP Cluster's Alpaca tokenizer config
        language_tokenizer = LlamaTokenizer.from_pretrained(
            language_tokenizer_name, padding_side="left"
        )
    else:
        language_tokenizer = AutoTokenizer.from_pretrained(
            language_tokenizer_name, padding_side="left"
        )

    print(f"Instantiated language tokenizer {language_tokenizer_name}")
    return language_tokenizer


def load_reward_tokenizer(reward_tokenizer_name: str) -> PreTrainedTokenizerBase:
    """Load the reward model's tokenizer."""
    if reward_tokenizer_name == "mock":
        reward_tokenizer_name = "gpt2"
    elif (
        "rm_combined" in reward_tokenizer_name or "oliversssf2" in reward_tokenizer_name
    ):
        reward_tokenizer_name = "EleutherAI/gpt-neo-1.3B"

    if "llama" in reward_tokenizer_name or "alpaca" in reward_tokenizer_name:
        # Fix for misnamed class in the NLP Cluster's Alpaca tokenizer config
        reward_tokenizer_train = LlamaTokenizer.from_pretrained(reward_tokenizer_name)
    else:
        reward_tokenizer_train = AutoTokenizer.from_pretrained(reward_tokenizer_name)

    print(f"Instantiated reward tokenizer {reward_tokenizer_name}")
    return reward_tokenizer_train


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
    known_args, unknown_args = parser.parse_known_args()

    if known_args.sweep != "":
        # Run sweeps
        with open(known_args.sweep, mode="r", encoding="utf-8") as f:
            sweep_params = yaml.load(f, Loader=yaml.FullLoader)
        wandb.agent(
            sweep_params["id"],
            function=lambda: main(known_args, unknown_args),
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            count=sweep_params["count"],
        )
    else:
        main(known_args, unknown_args)
