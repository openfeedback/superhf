"""
Instead of an online method like RLHF or SuperHF, simply take all the chosen responses from the
train dataset and do supervised fine-tuning on them.
"""

import argparse
import random
from typing import Any

from datasets import Dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
import wandb

from superhf.data import get_superhf_prompts
from superhf.utils import print_memory_utilization, set_seed


WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "superhf-v3"


def main() -> None:
    """Main execution of the script."""

    # pylint: disable=too-many-statements

    # Initialization
    print_memory_utilization()
    torch.multiprocessing.set_sharing_strategy("file_system")
    set_seed(66)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--scheduler_warmup_steps", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=["q_proj", "v_proj"]
    )
    parser.add_argument("--hub_repo_id", type=str, default="sft-on-preferences")
    parser.add_argument("--push_interval", type=int, default=128, help="In examples.")

    # Initialize wandb and hub api
    hf_api = HfApi()
    hf_api.whoami()
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        save_code=True,
        config=parser.parse_args(),
    )
    assert run is not None

    # Load dataset
    examples: list[str] = []
    for dataset_name in wandb.config.datasets:
        examples.extend(get_superhf_prompts(dataset_name, load_whole_completion=True))
    random.shuffle(examples)

    # Only load the first section of prompts
    if wandb.config.num_examples != 0:
        examples = examples[: wandb.config.num_examples]

    print(f"Loaded {len(examples)} prompts.")
    print_memory_utilization()
    print("Instantiating models...")

    # Load model and tokenizer
    language_model = load_language_model(wandb.config.model_name).to(device)
    language_model.train()
    language_tokenizer = load_language_tokenizer(wandb.config.model_name)

    # Tokenize the dataset
    print("Tokenizing dataset...")

    def tokenize_function(examples: dict[str, list[str]]) -> Any:
        return language_tokenizer(
            examples["text"], padding="max_length", truncation=True
        )

    dataset = Dataset.from_dict({"text": examples})
    dataset = dataset.map(tokenize_function, batched=True)
    # dataset = language_tokenizer(examples, padding="max_length", truncation=True)

    # Initializer trainer
    save_steps = wandb.config.push_interval / wandb.config.batch_size
    assert save_steps.is_integer(), "Push interval must be divisible by batch size."
    save_steps = int(save_steps)
    training_args = TrainingArguments(
        output_dir="./sft_training_output/",
        num_train_epochs=1,
        per_device_train_batch_size=wandb.config.batch_size,
        bf16=wandb.config.mixed_precision == "bf16",
        report_to=["wandb"],
        lr_scheduler_type="cosine",
        learning_rate=wandb.config.lr,
        warmup_steps=wandb.config.scheduler_warmup_steps,
        save_steps=save_steps,
    )

    class PushModelCallback(TrainerCallback):
        """Callback for pushing to the hub during training."""

        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            _: TrainerControl,
            **kwargs: Any,
        ) -> None:
            """Push the model to the hub."""
            print_memory_utilization()
            push_to_hub(
                hf_api,
                language_model,
                state.global_step * args.train_batch_size,
            )

    class CustomTrainer(Trainer):
        """Custom trainer for pushing LoRA adapters to the hub."""

        def _save_checkpoint(self, *_: Any, **__: Any) -> None:
            """Turn off saving the full model to file."""

    trainer = CustomTrainer(
        model=language_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=language_tokenizer,
        data_collator=DataCollatorForLanguageModeling(language_tokenizer, mlm=False),
        callbacks=[PushModelCallback],
    )

    # Push the tokenizer first since it won't change
    language_tokenizer.push_to_hub(
        repo_id=wandb.config.hub_repo_id,
        commit_message="Upload tokenizer",
    )

    # Train model
    print("Training model...")
    trainer.train()

    # Finish run
    wandb.alert(
        title="FINISHED SFT-on-preferences run!",
        text="FINISHED SFT-on-preferences run! <@WGPFRK13K>",
    )
    run.finish()


def push_to_hub(
    hf_api: HfApi,
    model: PreTrainedModel,
    num_examples: int,
) -> None:
    """Push the model to the hub."""
    assert hf_api is not None

    tqdm.write("🚀 Pushing model and tokenizer to the Hub!")
    repo_name = wandb.config.hub_repo_id
    if "debug" in repo_name:
        tqdm.write(repo_name + " (not actually pushed due to 'debug' in name)")
    else:
        tqdm.write(
            str(
                model.push_to_hub(
                    repo_id=repo_name,
                    commit_message=f"Upload model from {num_examples} examples",
                )
            )
        )
        tqdm.write(str())
        # Create a new branch with the superbatch index as the name
        hf_username = hf_api.whoami()["name"]
        repo_id = hf_username + "/" + repo_name
        branch = f"step-{num_examples:04}"
        try:
            hf_api.create_branch(repo_id=repo_id, branch=branch)
        except HfHubHTTPError:
            # Delete the branch first
            hf_api.delete_branch(repo_id=repo_id, branch=branch)
            hf_api.create_branch(repo_id=repo_id, branch=branch)


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


def load_language_tokenizer(language_tokenizer_name: str) -> PreTrainedTokenizerBase:
    """Load the language model tokenizer."""
    if "llama" in language_tokenizer_name or "alpaca" in language_tokenizer_name:
        # Fix for misnamed class in the NLP Cluster's Alpaca tokenizer config
        language_tokenizer = LlamaTokenizer.from_pretrained(language_tokenizer_name)
    else:
        language_tokenizer = AutoTokenizer.from_pretrained(language_tokenizer_name)

    if language_tokenizer.pad_token is None:
        language_tokenizer.pad_token = language_tokenizer.eos_token
    print(f"Instantiated language tokenizer {language_tokenizer_name}")
    return language_tokenizer


if __name__ == "__main__":
    main()
