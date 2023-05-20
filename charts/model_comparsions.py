"""Held out test rewards across models."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    get_test_scores,
    initialize_plot,
    save_plot,
    model_type_to_palette_color,
)

OUTPUT_FILE = "./charts/models/model_comparisons.png"

# SCORE_BIAS = 0.6
SCORE_BIAS = 0.0
Y_MIN = -0.6


def main() -> None:
    """Main function."""

    # Initialize
    initialize_plot()

    # Define the model names
    model_names = [
        "llama-7b.json",
        "rlhf-v4-llama.json",
        "shf-llama-7b.json",
        "alpaca_7b.json",
        "sft-on-preferences-v1.json",
        # "test-save-alpaca@model-2048-prompts-batch-size-8.json",
        "rlhf-v3-lr-5.0e-6-batch-16@gold-run.json",
        "shf-7b-default.json",
        # "shf-7b-gold-v1.json",
        "gpt-4_2023-05-13_completions_output.json",
    ]

    # Create an empty list for x-axis labels
    x_labels = [
        "LLaMA",
        "RLHF\n(LLaMA)",
        "SuperHF\n(LLaMA)",
        "Alpaca",
        "FTP\n(Alpaca)",
        "RLHF\n(Alpaca)",
        "SuperHF\n(Alpaca)",
        "GPT-4",
    ]

    # Calculate plot values for data
    all_data = []
    all_mean_scores = []
    for model_name in model_names:
        # TODO refactor this into chart_utils.py
        file_path = f"./experiments/evaluations/test_scores/{model_name}"
        scores = get_test_scores(file_path)

        mean_score = np.mean(scores)
        all_mean_scores.append(mean_score)
        print(f"{model_name} mean score: {mean_score:.2f}")

        # Add the data
        for score in scores:
            # score = score - Y_MIN
            all_data.append([model_name, score + SCORE_BIAS])

    dataframe = pd.DataFrame(all_data, columns=["Model", "Score"])

    # Create the plot
    errorbar = "ci"
    plt.rcParams["lines.markersize"] = 1
    palette = [model_type_to_palette_color(model_name) for model_name in x_labels]
    plot = sns.barplot(
        data=dataframe,
        x="Model",
        y="Score",
        capsize=0.1,
        errorbar=errorbar,
        palette=palette,
        # bottom=Y_MIN,
        # color=model_type_to_palette_color("superhf"),
    )
    # Disable the fill for negative average scores
    for patch in plot.patches:
        if patch.get_height() < 0.0:
            patch.set_color("#fff0")

    sns.barplot(
        x=x_labels,
        y=[Y_MIN if score > 0.0 else score - Y_MIN for score in all_mean_scores],
        bottom=[Y_MIN if score < 0.0 else 0.0 for score in all_mean_scores],
        palette=palette,
    )

    # Horizontal line at Alpaca
    plt.axhline(
        y=all_mean_scores[3], color="black", linestyle="--", label="Alpaca Mean"
    )
    plt.legend()

    # Set labels and title
    plt.xlabel("Model Type")
    plt.ylabel("Test Score")
    plt.title("Held-Out Test Scores")

    # Set x-ticks
    plt.xticks(range(len(x_labels)), x_labels)

    # Print the plot
    # plt.show()

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
