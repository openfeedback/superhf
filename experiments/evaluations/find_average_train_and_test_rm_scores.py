"""
Generate a table of average RM scores for each model on each dataset.
"""

import json
import os
from tabulate import tabulate

import numpy as np

TRAIN_FOLDER = "experiments/evaluations/train_scores"
TEST_FOLDER = "experiments/evaluations/test_scores"

MODELS = [
    "alpaca_7b",
    # "llama-7b",
    # "sft-on-preferences-v1",
    "rlhf-v3-lr-5.0e-6-batch-16@gold-run",
    "test-save-alpaca@model-2048-prompts-batch-size-8",
    "shf-7b-gold-v1@step-0064",
    "shf-7b-gold-v1@step-8192",
    # "shf-7b-default",
    # "pythia-6.9B-deduped",
]


def reformat_folder_name(name):
    """
    Given a folder name in format experiments/evaluations/train_scores
    reformat to just 'test' or 'train'
    """
    return name.split(os.path.sep, maxsplit=2)[2].split("_")[0]


def main() -> None:
    """Main function."""
    for folder in [TRAIN_FOLDER, TEST_FOLDER]:
        table_data = []
        headers = [
            "Language Model",
            f"{reformat_folder_name(folder)} dataset",
            "Average",
            "STD",
            "Median",
            "Min",
            "Max",
        ]

        for model in MODELS:
            model_scores = []
            scores_per_dataset = {}
            with open(os.path.join(folder, f"{model}.json"), encoding="utf-8") as file:
                scores_file = json.load(file)
            for dataset in scores_file:
                if dataset not in scores_per_dataset:
                    scores_per_dataset[dataset] = []
                scores_per_dataset[dataset].extend(scores_file[dataset])
                model_scores.extend(scores_file[dataset])
            if "test-save-alpaca" in model:
                model = "rlhf-gold-v1"
            elif "rlhf-v3-lr-5.0e-6-batch-16@gold-run" == model:
                model = "rlhf-gold-final-v3"
            table_data.append(
                [
                    model,
                    "All",
                    np.mean(model_scores).round(2),
                    np.std(model_scores).round(2),
                    np.median(model_scores).round(2),
                    np.min(model_scores).round(2),
                    np.max(model_scores).round(2),
                ]
            )

            for dataset, scores in scores_per_dataset.items():
                table_data.append(
                    [
                        model,
                        dataset,
                        np.mean(scores).round(2),
                        np.std(scores).round(2),
                        np.median(scores).round(2),
                        np.min(scores).round(2),
                        np.max(scores).round(2),
                    ]
                )

        print(tabulate(table_data, headers, tablefmt="grid"))
        print("---------")

        if not os.path.isdir("experiments/evaluations/tables"):
            os.mkdir("experiments/evaluations/tables")
        # save results in latex to a file in tables called folder.txt
        with open(
            os.path.join(
                "experiments/evaluations/tables",
                f"{folder.split(os.path.sep, maxsplit=2)[2]}.txt",
            ),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(tabulate(table_data, headers, tablefmt="latex"))


if __name__ == "__main__":
    main()
