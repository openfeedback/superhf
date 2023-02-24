"""
Client code showing how to call the training loop for the iterative version of the SuperHF model.
"""

import random
from typing import Any

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


# TODO use argparse and wandb config for these instead
LANGUAGE_MODEL_NAME = "eleutherai/gpt-neo-125M"  # "eleutherai/gpt-neo-1.3B"
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-base"
DEBUG_MAX_PROMPTS = 1000
MAX_PROMPT_CHAR_LENGTH = 1024


def main() -> None:
    """
    Instantiate and train the SuperHF model.
    """

    device = torch.device(
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    )

    set_seed(66)

    # Get the prompt dataset
    prompts = get_superhf_prompts("anthropic-red-team")
    random.shuffle(prompts)

    # Filter out prompts that are too long
    old_prompt_count = len(prompts)
    prompts = [prompt for prompt in prompts if len(prompt) < MAX_PROMPT_CHAR_LENGTH]
    print(
        f"Filtered {old_prompt_count - len(prompts)} prompts over {MAX_PROMPT_CHAR_LENGTH} chars."
    )

    # Testing: only load the first section of prompts
    if DEBUG_MAX_PROMPTS != 0:
        prompts = prompts[:DEBUG_MAX_PROMPTS]

    print(f"Loaded {len(prompts)} prompts.")

    # Instantiate our language and reward models and tokenizers
    language_model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_NAME).to(
        device
    )
    reward_model = (
        MockRewardModel()
    )  # AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)
    language_tokenizer = AutoTokenizer.from_pretrained(
        LANGUAGE_MODEL_NAME, padding_side="left"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
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

    # Begin our experiment
    config: dict[str, Any] = {}
    wandb.init(
        project="shf-iterative-v1",
        notes="First test run",
        config=config,
        save_code=True,
    )
    initialize_metrics_wandb()  # Defines the run metrics
    # wandb.watch(language_model, log="all")

    # Run training
    wandb.alert(title="Beginning SuperHF run", text="Beginning SuperHF run...")
    trainer.train(prompts)


if __name__ == "__main__":
    main()
