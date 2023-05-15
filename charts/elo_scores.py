"""
Plot elo scores as for each model type.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    initialize_plot,
    save_plot,
    MODEL_NAME_MAPPING,
)

INPUT_FILE = "./eval_results/gpt4_qualitative/elo_scores.json"
OUTPUT_FILE = "./charts/models/elo_scores.png"


def main() -> None:
    """Main function."""

    order = [
        "LLaMA",
        "Alpaca",
        "SFT",
        "RLHF",
        "SuperHF",
        "GPT-3.5",
        "GPT-4",
    ]

    # Initialize
    initialize_plot()

    # Load the data
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Create a dataframe
    elo_data = []
    for model_name, elo_scores in data.items():
        for score in elo_scores:
            elo_data.append([MODEL_NAME_MAPPING[model_name], score])

    dataframe = pd.DataFrame(elo_data, columns=["Model", "Score"])

    # Create the plot
    plt.rcParams["lines.markersize"] = 1
    sns.barplot(
        data=dataframe,
        x="Model",
        y="Score",
        capsize=0.1,
        errorbar="sd",
        order=order,
        # label="Pythia Base",
    )

    # Set the y-axis limits
    plt.ylim(1150, 1850)

    # Set labels and title
    plt.xlabel("Model Type")
    plt.ylabel("Elo Score")
    plt.title("Elo Scores (From GPT-4's Preferences)")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
