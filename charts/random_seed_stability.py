"""
Plot test as a function of the prompt accumulation.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    get_test_scores,
    initialize_plot,
    model_type_to_palette_color,
    save_plot,
    LLAMA_TEST_REWARD,
)

OUTPUT_FILE = "./charts/models/random_seed_stability.png"
TEST_SCORES_DIRECTORY = "./experiments/evaluations/test_scores/"


def main() -> None:
    """Main function."""

    # Initialize
    initialize_plot()
    plt.rcParams["lines.markersize"] = 1

    # Define the model sizes and their corresponding file names
    model_types_to_test_names = {"SuperHF": "shf-v4-llama-seed-"}

    # Find all the files in the test scores directory
    shf_data = []
    for model_type, file_name_template in model_types_to_test_names.items():
        for file_name in os.listdir(TEST_SCORES_DIRECTORY):
            if file_name.startswith(file_name_template):
                file_path = os.path.join(TEST_SCORES_DIRECTORY, file_name)
                scores = get_test_scores(file_path)
                labeled_scores = [
                    [model_type, score - LLAMA_TEST_REWARD] for score in scores
                ]
                shf_data.extend(labeled_scores)

    dataframe = pd.DataFrame(shf_data, columns=["Model", "Score"])

    # Create the plot
    sns.barplot(
        data=dataframe,
        x="Model",
        y="Score",
        errorbar="ci",
        capsize=0.1,
        # marker="",
        # label="Pythia Base",
        palette=[model_type_to_palette_color(m) for m in dataframe["Model"].unique()],
        # color=model_type_to_palette_color("superhf"),
    )

    # Set labels and title
    plt.xlabel("Model")
    plt.ylabel("Test Score (Improvement over LLaMA)")
    plt.title("Stability Across 20 Random Seeds (2500 Train Prompts)")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
