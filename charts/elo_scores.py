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
    QUALITATIVE_MODEL_ORDER,
)

QUALITATIVE_MODEL_ORDER = QUALITATIVE_MODEL_ORDER[:-2]

INPUT_FILE = "./eval_results/gpt4_qualitative/elo_scores.json"
OUTPUT_FILE = "./charts/models/elo_scores.png"
# OUTPUT_FILE = "./charts/models/elo_scores_all.png"


def main() -> None:
    """Main function."""

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

    dataframe = pd.DataFrame(elo_data, columns=["Model", "Elo"])

    # Print out the average elo for each model
    for model_name in QUALITATIVE_MODEL_ORDER:
        model_scores = dataframe[dataframe["Model"] == model_name]["Elo"]
        print(f"{model_name}: {model_scores.mean():.2f}")

    # Create the plot
    plt.rcParams["lines.markersize"] = 1
    sns.barplot(
        data=dataframe,
        x="Model",
        y="Elo",
        capsize=0.1,
        errorbar="ci",
        order=QUALITATIVE_MODEL_ORDER,
        # label="Pythia Base",
    )

    # Horizontal line at 1500 for starting Elo
    plt.axhline(y=1500, color="black", linestyle="--", label="Initial Elo")

    # Add a legend
    plt.legend()

    # Set the y-axis limits
    # plt.ylim(1200, 1550)
    plt.ylim(1300, 1630)
    # plt.ylim(1200, 1800)

    # Set labels and title
    plt.xlabel("Model Type")
    plt.ylabel("Elo Score")
    plt.title("Elo Scores (From GPT-4's Preferences)")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
