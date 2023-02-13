"""
Executable script for running the SuperHF single-pass best-of-N finetuning experiment.
Counterpart to the notebook.
"""

import os
import random

import torch
import wandb
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from matplotlib import pyplot as plt  # type: ignore
from superhf.data import get_superhf_prompts  # type: ignore
from superhf.finetuning import SinglePassBestOfNTrainer  # type: ignore

LANGUAGE_MODEL_NAME = "eleutherai/gpt-neo-125M"
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-base"
DATASET_NAME = "anthropic-red-team"
NUM_TEST_EXAMPLES = 100
TUNING_INTERVAL = 100
RANDOM_SEED = 66
OUTPUT_DIR = "v2.00"
SMALL_DEBUGGING_SAMPLE = 1000  # Set to None to use the full dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "Must be using a GPU."

# Initialize random seeds for everything
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
transformers.enable_full_determinism(RANDOM_SEED)

language_model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_NAME).to(device)
reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME).to(
    device
)

language_tokenizer = AutoTokenizer.from_pretrained(
    LANGUAGE_MODEL_NAME, padding_side="left"
)
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)

# Load a list of prompts
dataset = get_superhf_prompts(DATASET_NAME)[:SMALL_DEBUGGING_SAMPLE]

# Plot a histogram of the word counts
plt.hist([len(example.split()) for example in dataset], bins=50, log=True)
plt.title("Histogram of word counts")
plt.xlabel("Word count")
plt.ylabel("Frequency (log scale)")
plt.savefig(os.path.join(OUTPUT_DIR, "word_counts.png"))

# Cut off some of the long examples
prev_dataset_length = len(dataset)
filtered_dataset = [example for example in dataset if len(example.split()) < 100]
print(f"Removed {prev_dataset_length - len(filtered_dataset)} examples.")

# Split it into a number of test examples and all the rest for training
random.shuffle(filtered_dataset)
train_dataset = filtered_dataset[NUM_TEST_EXAMPLES:]
test_dataset = filtered_dataset[:NUM_TEST_EXAMPLES]

# Randomize it
print(
    f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples."
)

# Print some examples
print("Test examples:")
for example in test_dataset[:3]:
    print(example)

trainer = SinglePassBestOfNTrainer(
    language_model,
    reward_model,
    language_tokenizer,
    reward_tokenizer,
    train_dataset,
    test_dataset,
    output_dir=OUTPUT_DIR,
)

all_completions, filtered_completions = trainer.filter_completions()

# Print some statistics
scores_all = [completion["score"] for completion in all_completions]
scores_filtered = [completion["score"] for completion in filtered_completions]
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
plt.savefig(os.path.join(trainer.output_dir, "completion_scores.png"))

SHUTDOWN_AFTER_RUN = False
config = {
    "version": "v1.6",
    "language_model_name": LANGUAGE_MODEL_NAME,
    "reward_model_name": REWARD_MODEL_NAME,
    "dataset_name": DATASET_NAME,
    "num_train_examples": len(train_dataset),
    "num_test_examples": len(test_dataset),
    "lr": 5e-7,
    "lr_scheduler_type": "constant",
    "warmup_steps": 1024,
    "weight_decay": 0.01,
    "train_batch_size": 4,
    "eval_batch_size": 4,
}

wandb.init(
    project="shf-single-pass",
    notes="half lr",
    config=config,
    save_code=True,
    name=f"shf_single_pass_{config['version']}",
)
# wandb.run.name = f"shf_single_pass_{config['version']}"
wandb.watch(language_model, log="all")
wandb.define_metric("train/loss", summary="min")
wandb.define_metric("average_reward", summary="max")
wandb.define_metric("average_completion_length", summary="last")

training_args = TrainingArguments(
    output_dir=os.path.join(trainer.output_dir, "checkpoints", wandb.config.version),
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=wandb.config.train_batch_size,
    per_device_eval_batch_size=wandb.config.eval_batch_size,
    evaluation_strategy="steps",
    eval_steps=512,
    logging_steps=512,
    save_steps=1024,
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

# Optional: Hibernate the computer
if SHUTDOWN_AFTER_RUN:
    os.system("shutdown /h")
