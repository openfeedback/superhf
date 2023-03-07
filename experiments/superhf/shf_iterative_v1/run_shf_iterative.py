"""
Client code showing how to call the training loop for the iterative version of the SuperHF model.
"""

import argparse
import random
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
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


def main() -> None:
    """
    Instantiate and train the SuperHF model.
    """
    # pylint: disable=too-many-locals

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
        entity="stanfordaialignment",
        project="shf-iterative-v1",
        notes=args.notes,
        save_code=True,
        config=args.config,
    )
    language_model_name = wandb.config.language_model_name
    reward_model_name = wandb.config.reward_model_name

    if language_model_name == "mock" and reward_model_name == "mock":
        # Create a mock dataset of prompts
        prompts = [
            f"{i}\n\nHuman: ...\n\nAssistant: Sphinx of black quartz, judge my vow."
            for i in range(50000)
        ]
    else:
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
    if wandb.config.debug_max_prompts != 0:
        prompts = prompts[: wandb.config.debug_max_prompts]

    print(f"Loaded {len(prompts)} prompts.")
    print_gpu_utilization()
    print("Instantiating models...")
    # Instantiate our language and reward models and tokenizers
    language_model = (
        MockLanguageModel()
        if language_model_name == "mock"
        else AutoModelForCausalLM.from_pretrained(language_model_name).to(device)
    )
    reward_model = (
        MockRewardModel()
        if reward_model_name == "mock"
        else AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(
            device
        )
    )
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
    print_gpu_utilization()

    # Set our training arguments
    print("Setting up trainer...")
    logits_processors = LogitsProcessorList([NoRepeatNGramLogitsProcessor(6)])
    training_args = SuperHFTrainingArguments(
        temperature=wandb.config.temperature,
        top_p=wandb.config.top_p,
        superbatch_size=wandb.config.superbatch_size,
        max_length_lm=wandb.config.max_length_lm,
        max_length_rm=wandb.config.max_length_rm,
        minibatch_size_generating=wandb.config.minibatch_size_generating,
        minibatch_size_scoring=wandb.config.minibatch_size_scoring,
        minibatch_size_finetuning=wandb.config.minibatch_size_finetuning,
        mixed_precision=wandb.config.mixed_precision,
        constitution_prompt=wandb.config.constitution_prompt,
        logits_processors=logits_processors,
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
        reward_model=reward_model,
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
    main()
