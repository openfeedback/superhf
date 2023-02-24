"""
Client code showing how to call the training loop for the iterative version of the SuperHF model.
"""

import argparse
import random
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # AutoModelForSequenceClassification,
)
import torch
import wandb

from superhf.data import get_superhf_prompts
from superhf.filtering import CompletionFilterTopK
from superhf.metrics import (
    initialize_metrics_wandb,
    report_metrics_wandb,
    report_metrics_print,
)
from superhf.mocking import MockRewardModel
from superhf.training import SuperHFTrainingArguments, SuperHFTrainer
from superhf.utils import set_seed, print_gpu_utilization


def main() -> None:
    """
    Instantiate and train the SuperHF model.
    """

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        # Get file path relative to this file
        default=os.path.join(os.path.dirname(__file__), "configs", "gpt-neo-1.3B.yaml"),
        help="The name of the Weights and Biases config to use.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Notes to add to the Weights and Biases run.",
    )
    args = parser.parse_args()

    # Configure device and seed
    device = torch.device(
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    )
    set_seed(66)

    # Initialize Weights and Biases run
    wandb.init(
        project="shf-iterative-v1",
        notes=args.notes,
        save_code=True,
        config=args.config,
    )

    # Get the prompt dataset
    prompts = get_superhf_prompts("anthropic-red-team")
    random.shuffle(prompts)

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

    # Testing: only load the first section of prompts
    if wandb.config.DEBUG_MAX_PROMPTS != 0:
        prompts = prompts[: wandb.config.DEBUG_MAX_PROMPTS]

    print(f"Loaded {len(prompts)} prompts.")

    # Instantiate our language and reward models and tokenizers
    language_model = AutoModelForCausalLM.from_pretrained(
        wandb.config.LANGUAGE_MODEL_NAME
    ).to(device)
    reward_model = (
        MockRewardModel()
    )  # AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)
    language_tokenizer = AutoTokenizer.from_pretrained(
        wandb.config.LANGUAGE_MODEL_NAME, padding_side="left"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(wandb.config.REWARD_MODEL_NAME)
    print_gpu_utilization()

    # Set our training arguments
    training_args = SuperHFTrainingArguments()
    completion_filter_top_k = 8
    completion_filter = CompletionFilterTopK(completion_filter_top_k)

    # Instantiate our trainer
    trainer = SuperHFTrainer(
        language_model=language_model,
        reward_model=reward_model,
        language_tokenizer=language_tokenizer,
        reward_tokenizer=reward_tokenizer,
        completion_filter=completion_filter,
        training_args=training_args,
        report_metrics=[report_metrics_wandb, report_metrics_print],
    )

    # Initialize metrics
    initialize_metrics_wandb()  # Defines the run metrics
    # wandb.watch(language_model, log="all")

    # Run training
    wandb.alert(title="Beginning SuperHF run", text="Beginning SuperHF run...")
    trainer.train(prompts)


if __name__ == "__main__":
    main()
