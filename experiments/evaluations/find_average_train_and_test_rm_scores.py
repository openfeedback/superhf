"""Find the average of the train and test rm scores to compare."""

import json
import os

import numpy as np

TRAIN_FOLDER = "experiments/evaluations/train_scores"
TEST_FOLDER = "experiments/evaluations/test_scores"

MODELS = [
    "alpaca_7b",
    "llama-7b",
    "sft-on-preferences-v1",
    "rlhf-v3-lr-5.0e-6-batch-16@gold-run",
    "shf-7b-default",
    # "pythia-6.9B-deduped",
]


def main() -> None:
    """Main function."""
    for folder in [TRAIN_FOLDER, TEST_FOLDER]:
        all_scores = []
        for model in MODELS:
            with open(os.path.join(folder, f"{model}.json"), encoding="utf-8") as file:
                scores_file = json.load(file)
            for scores in scores_file.values():
                all_scores.extend(scores)

        print(f"{folder} average: {np.mean(all_scores).round(2)}")
        print(f"{folder} std: {np.std(all_scores).round(2)}")


if __name__ == "__main__":
    main()
