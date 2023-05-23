"""
Makes graphs of the scores per dataset for each model.

Usage:
    python experiments/evaluations/graph_per_dataset_scores.py

Turn ERROR_BARS on or off for error bars
"""

import json
import os
import plotly.graph_objects as go

import numpy as np

TRAIN_FOLDER = "experiments/evaluations/train_scores"
TEST_FOLDER = "experiments/evaluations/test_scores"
ERROR_BARS = False

MODELS = [
    "alpaca_7b",
    # "llama-7b",
    # "sft-on-preferences-v1",
    "rlhf-v3-lr-5.0e-6-batch-16@gold-run",
    "test-save-alpaca@model-2048-prompts-batch-size-8",
    "shf-7b-gold-v1",
    "shf-7b-gold-v1@step-0064",
    "shf-7b-gold-v1@step-1024",
    "shf-7b-gold-v1@step-8192",
    "shf-pythia-12B@v3",
    "pythia-12B-deduped",
    # "shf-7b-default",
    # "pythia-6.9B-deduped",
]


def reformat_folder_name(name):
    """
    Given a folder name in format experiments/evaluations/train_scores
    reformat to just 'test' or 'train'
    """
    return name.split(os.path.sep, maxsplit=2)[2].split("_")[0]


def create_bar_plot(folder):
    """
    Given a folder of scores, create a bar plot of the mean scores per dataset
    """
    data = []
    datasets = []
    for model in MODELS:
        model_scores = []
        scores_per_dataset = {}
        with open(os.path.join(folder, f"{model}.json"), encoding="utf-8") as file:
            scores_file = json.load(file)
        for dataset in scores_file:
            if dataset not in datasets:
                datasets.append(dataset)
            if dataset not in scores_per_dataset:
                scores_per_dataset[dataset] = []
            scores_per_dataset[dataset].extend(scores_file[dataset])
            model_scores.extend(scores_file[dataset])
        if "test-save-alpaca" in model:
            model = "rlhf-gold-v1"
        elif "rlhf-v3-lr-5.0e-6-batch-16@gold-run" == model:
            model = "rlhf-gold-final-v3"
        mean_scores = [
            (
                np.mean(scores_per_dataset[dataset]).round(2)
                if dataset in scores_per_dataset
                else 0
            )
            for dataset in datasets
        ]
        if ERROR_BARS:
            mean_stds = [
                (
                    np.std(scores_per_dataset[dataset]).round(2)
                    if dataset in scores_per_dataset
                    else 0
                )
                for dataset in datasets
            ]
            data.append(
                go.Bar(
                    name=model,
                    x=datasets,
                    y=mean_scores,
                    error_y={"type": "data", "array": mean_stds},
                )
            )
        else:
            data.append(go.Bar(name=model, x=datasets, y=mean_scores))

    fig = go.Figure(data=data)
    fig.update_layout(
        barmode="group",
        xaxis_title="Dataset",
        yaxis_title="Mean Score",
        title=f"Comparison of Mean Scores by Model ({reformat_folder_name(folder)} RM)",
    )
    fig.show()


def main() -> None:
    """Main function."""
    create_bar_plot(TRAIN_FOLDER)
    create_bar_plot(TEST_FOLDER)


if __name__ == "__main__":
    main()
