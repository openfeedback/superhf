"""
Instead of an online method like RLHF or SuperHF, simply take all the chosen responses from the
train dataset and do supervised fine-tuning on them.
"""

import argparse
import random

# from typing import Dict, List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import get_peft_model, LoraConfig
import wandb

from superhf.data import get_superhf_prompts
from superhf.utils import print_memory_utilization, set_seed

# from reward_modelling.reward_model import RewardModel


WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "superhf-v3"


def main() -> None:
    """Main execution of the script."""

    # Initialization
    print_memory_utilization()

    # Attempt to fix too many open files issue on SLURM
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Configure seed
    set_seed(66)

    # Enable tf32 training if supported
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("Enabling tf32 training.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train a model on the preferences dataset."
    )
    parser.add_argument(
        "--model_name", type=str, default="/juice5/scr5/nlp/llama_model/alpaca_7b"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[
            "anthropic-helpful-base",
            "anthropic-harmless-base",
        ],
    )
    parser.add_argument("--num_prompts", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size_initial", type=int, default=8)
    parser.add_argument("--scheduler_warmup_steps", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=["q_proj", "v_proj"]
    )
    parser.add_argument("--hub_repo_id", type=str, default="sft-on-preferences")
    parser.add_argument("--push_interval", type=int, default=128)

    # Initialize wandb
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        save_code=True,
        config=parser.parse_args(),
    )
    assert run is not None

    # Load dataset
    prompts: list[str] = []
    for dataset_name in wandb.config.datasets:
        prompts.extend(get_superhf_prompts(dataset_name, load_whole_completion=True))
    random.shuffle(prompts)

    # Only load the first section of prompts
    if wandb.config.num_prompts != 0:
        prompts = prompts[: wandb.config.num_prompts]

    print(f"Loaded {len(prompts)} prompts.")
    print_memory_utilization()
    print("Instantiating models...")

    # Load model and tokenizer

    # Initializer trainer

    # Train model

    # Finish run
    wandb.alert(
        title="FINISHED SFT-on-preferences run!",
        text="FINISHED SFT-on-preferences run! <@WGPFRK13K>",
    )
    run.finish()


def load_language_model(language_model_name: str) -> PreTrainedModel:
    """Load the language model."""
    language_model = AutoModelForCausalLM.from_pretrained(
        language_model_name,
        low_cpu_mem_usage=True,
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


if __name__ == "__main__":
    main()
