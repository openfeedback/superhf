"""
Executable script for running the SuperHF single-pass best-of-N finetuning experiment.
Counterpart to the notebook.

There are two modes. One mode uses the assistant to generate completions and the reward model
to evaluate the completions. The other mode uses the completions to finetune the assistant on
the best of n completions as scored by the reward model.

Example Usage:
    python shf_best_of_n_v2.py -v v2 -d --model_name eleutherai/gpt-neo-1.3B
"""

import os
import platform
import random
import sys
import argparse
from typing import List, Dict, Union

import wandb
import torch
import transformers  # type: ignore
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from matplotlib import pyplot as plt  # type: ignore
from superhf.data import get_superhf_prompts  # type: ignore
from superhf.finetuning import SinglePassBestOfNTrainer  # type: ignore

LANGUAGE_MODEL_NAME = "eleutherai/gpt-neo-1.3B"
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-base"
DATASET_NAME = "anthropic-red-team"
NUM_TRAIN_EXAMPLES = 8000
NUM_TEST_EXAMPLES = 100
RANDOM_SEED = 66
SHUTDOWN_AFTER_RUN = True
MAX_EXAMPLE_LENGTH = 36


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="The version of the experiment. Used to create the output directory.",
    )
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="Specify this flag to generate completions instead of finetuning the assistant.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Specify this flag to run the experiment in debug mode. In this mode we use"
        " 1,000 prompts for training, and only load 124 prompts for completing.",
    )
    parser.add_argument(
        "--language_model_name",
        type=str,
        default=LANGUAGE_MODEL_NAME,
        help="The name of the language model to finetune.",
    )
    parser.add_argument(
        "--reward_model_name",
        type=str,
        default=REWARD_MODEL_NAME,
        help="The name of the reward model to use for scoring completions.",
    )
    parser.add_argument(
        "--max_completion_new_tokens",
        type=int,
        default=64,
        help="The maximum number of new tokens to generate for each completion.",
    )
    parser.add_argument(
        "--completion_batch_size",
        type=int,
        default=8,
        help="The batch size to use when generating completions.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        help="Leave blank if not on SC to use the output dir. If on SC, this is the directory"
        " to save checkpoints to underneath scr*/. Should use a"
        " unique name that won't clash with other people.",
    )
    parser.add_argument(
        "--score_batch_size",
        type=int,
        default=8,
        help="The batch size to use when scoring completions with the reward model.",
    )
    return parser.parse_args()


def plot_word_counts(dataset: list[str], output_dir) -> None:
    """
    Plot a histogram of the word counts.
    """
    plt.hist([len(example.split()) for example in dataset], bins=50, log=True)
    plt.title("Histogram of word counts")
    plt.xlabel("Word count")
    plt.ylabel("Frequency (log scale)")
    plt.savefig(os.path.join(output_dir, "word_counts.png"))


def prepare_completions_dataset(
    dataset: list[str], max_example_length: int
) -> list[str]:
    """
    Prepare the dataset by filtering out completioins that are too long.
    """
    filtered_dataset = [
        example for example in dataset if len(example.split()) <= max_example_length
    ]
    print(f"Filtered out {len(dataset) - len(filtered_dataset)} prompts.")
    return filtered_dataset


def split_dataset(dataset: list[str]) -> tuple[list[str], list[str]]:
    """
    Split the dataset into train and test sets.
    """
    random.shuffle(dataset)
    # Cut down the dataset for time
    # filtered_dataset = filtered_dataset[:NUM_TRAIN_EXAMPLES + NUM_TEST_EXAMPLES]
    train_dataset = dataset[NUM_TEST_EXAMPLES:]
    test_dataset = dataset[:NUM_TEST_EXAMPLES]
    print(
        f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples."
    )

    # Print some examples
    print("Test examples:")
    for example in test_dataset[:3]:
        print(example)

    return train_dataset, test_dataset


def print_statistics(
    all_completions: List[Dict[str, Union[str, float]]],
    filtered_completions: List[Dict[str, Union[str, float]]],
    output_dir: str,
) -> None:
    """Print some statistics and plot a histogram of the scores of the completions."""
    scores_all = [completion["score"] for completion in all_completions]
    scores_filtered = [completion["score"] for completion in filtered_completions]
    print(f"Number of completions: {len(all_completions)}")
    print(f"Number of filtered completions: {len(filtered_completions)}")

    mean_score_all, std_score_all = (
        torch.tensor(scores_all).mean(),
        torch.tensor(scores_all).std(),
    )
    mean_score_filtered, std_score_filtered = (
        torch.tensor(scores_filtered).mean(),
        torch.tensor(scores_filtered).std(),
    )

    print(f"Mean score of all completions: {mean_score_all:.3f} ± {std_score_all:.3f}")
    print(
        f"Mean score of filtered completions: {mean_score_filtered:.3f} ± {std_score_filtered:.3f}"
    )

    # Graph a plot of the scores of the all and filtered completions
    plt.hist(
        [completion["score"] for completion in all_completions],
        bins=50,
        log=True,
        alpha=0.5,
        label="All completions",
    )
    # Smaller width per bin
    plt.hist(
        [completion["score"] for completion in filtered_completions],
        bins=50,
        log=True,
        alpha=0.5,
        label="Filtered completions",
        color="red",
        width=0.015,
    )
    plt.title("Histogram of completion scores")
    plt.xlabel("Completion score")
    plt.ylabel("Frequency (log scale)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "completion_scores.png"))


def check_node() -> str:
    """
    Check if we are on the sail compute cluster. If so, return a scratch directory to
    write checkpoints to and logs for wandb. If not, return None.
    """
    if not os.path.exists("/sailhome"):
        print("Not on sail compute cluster.")
        return ""
    # print machine name
    machine_name = platform.node().split(".")[0]
    print("We are running on node: ", machine_name)

    # print available scratch directories
    print(" ".join(os.listdir(f"/{machine_name}")))

    # get a random scratch directory
    scratch_dir = "scr0"
    print("Using scratch directory: ", scratch_dir)
    return scratch_dir


# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def main() -> None:
    """
    Main function that runs the experiment.
    """
    args = parse_args()
    print(args)

    scratch_dir = check_node()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login()

    # Initialize random seeds for everything
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    transformers.enable_full_determinism(RANDOM_SEED)

    language_model = AutoModelForCausalLM.from_pretrained(args.language_model_name).to(
        device
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name
    ).to(device)

    language_tokenizer = AutoTokenizer.from_pretrained(
        args.language_model_name, padding_side="left"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)

    train_dataset, test_dataset = None, None

    models = {"language_model": language_model, "reward_model": reward_model}
    tokenizers = {
        "language_tokenizer": language_tokenizer,
        "reward_tokenizer": reward_tokenizer,
    }
    data = {"train_prompts": train_dataset, "test_prompts": test_dataset}

    trainer = SinglePassBestOfNTrainer(
        models,
        tokenizers,
        data,
        output_dir=args.version,
        debug=args.debug,
    )
    # Load a list of prompts
    dataset = get_superhf_prompts(DATASET_NAME)
    if args.debug:
        dataset = dataset[:1000]
        print(
            "Running in debug mode, and now the length of the dataset is: ",
            len(dataset),
        )

    plot_word_counts(dataset, args.version)

    dataset = prepare_completions_dataset(
        dataset, max_example_length=MAX_EXAMPLE_LENGTH
    )

    train_dataset, test_dataset = split_dataset(dataset)
    trainer.test_prompts = test_dataset

    if args.generate:
        trainer.train_prompts = train_dataset
        print("Generating completions...")
        trainer.generate_completions(
            batch_size=args.completion_batch_size,
            max_new_tokens=args.max_completion_new_tokens,
        )
        wandb.init()
        wandb.alert(
            title=f"Done generating completions for {args.language_model_name}",
            text="Done generating completions.",
        )
        print("Done generating completions, so we exit the script.")
        trainer.score_completions(batch_size=args.score_batch_size)
        all_completions, filtered_completions = trainer.filter_completions()

        print_statistics(all_completions, filtered_completions, trainer.output_dir)
        sys.exit(0)
    else:
        print("Not generating completions, so we continue with the training code.")

    config = {
        "language_model_name": args.language_model_name,
        "reward_model_name": args.reward_model_name,
        "dataset_name": DATASET_NAME,
        "num_train_examples": -1 if not train_dataset else len(train_dataset),
        "num_test_examples": -1 if not test_dataset else len(test_dataset),
        "num_train_epochs": 2,
        "lr": 1.0e-7,
        "lr_scheduler_type": "constant",
        "warmup_steps": 1024,
        "weight_decay": 0.01,
        "train_batch_size": 2,
        "eval_batch_size": 4,
        "prepare_completions_dataset_max_example_length": MAX_EXAMPLE_LENGTH,
    }

    wandb.init(
        project="shf-single-pass",
        notes="lr 1e-7, constant schedule, 2 epochs, neo-1.3B",
        config=config,
        save_code=True,
        name=f"shf_single_pass_{args.version}",
    )
    wandb.config.update(args)
    wandb.watch(language_model, log="all")
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("average_reward", summary="max")
    wandb.define_metric("average_completion_length", summary="last")
    wandb.alert(title="Experiment Status", text="Beginning run...")

    checkpoint_dir = os.path.join(
        scratch_dir,
        args.checkpoint_dir,
        "checkpoints",
        wandb.config.version,
    )
    if args.checkpoint_dir:
        checkpoint_dir = os.path.join(
            args.output_dir,
            "checkpoints",
            wandb.config.version,
        )
    else:
        print(
            f"No checkpoint directory specified, so we use the default: {checkpoint_dir}"
        )

    print("Checkpoint directory: ", checkpoint_dir)
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        num_train_epochs=wandb.config.num_train_epochs,
        per_device_train_batch_size=wandb.config.train_batch_size,
        per_device_eval_batch_size=wandb.config.eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=2048,
        logging_steps=2048,
        save_steps=2048,
        # fp16=True,
        load_best_model_at_end=True,
        report_to="wandb",
        disable_tqdm=False,
        log_level="warning",
        learning_rate=wandb.config.lr,
        lr_scheduler_type=wandb.config.lr_scheduler_type,
        warmup_steps=wandb.config.warmup_steps,
        weight_decay=wandb.config.weight_decay,
    )

    trainer.tune_model(training_args)
    wandb.finish()

    wandb.alert(title="Experiment Status", text="Finished run!")
    if SHUTDOWN_AFTER_RUN:
        os.system("shutdown /h")


if __name__ == "__main__":
    main()
