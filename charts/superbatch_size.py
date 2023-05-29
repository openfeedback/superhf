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
)

OUTPUT_FILE = "./charts/ablations/superbatch_size.png"
TEST_SCORES_DIRECTORY = "./experiments/evaluations/test_scores/"
TRAIN_SCORES_DIRECTORY = "./experiments/evaluations/train_scores/"


def main() -> None:
    """Main function."""

    # Initialize
    initialize_plot()

    # Define the model sizes and their corresponding file names
    file_name_template = "shf-7b-sbs-"

    # Find all the files in the test scores directory
    train_data: list[list[str, float]] = []
    test_data: list[list[str, float]] = []
    for directory, data_list in [
        (TEST_SCORES_DIRECTORY, test_data),
        (TRAIN_SCORES_DIRECTORY, train_data),
    ]:
        for file_name in os.listdir(directory):
            if file_name.startswith(file_name_template):
                superbatch_size = int(
                    file_name.split(file_name_template)[1].split(".")[0]
                )
                scores = get_test_scores(os.path.join(directory, file_name))
                labeled_scores = [[superbatch_size, score] for score in scores]
                data_list.extend(labeled_scores)

    train_dataframe = pd.DataFrame(train_data, columns=["Superbatch Size", "Score"])
    test_dataframe = pd.DataFrame(test_data, columns=["Superbatch Size", "Score"])

    # Create the plot
    sns.lineplot(
        data=train_dataframe,
        x="Superbatch Size",
        y="Score",
        errorbar="ci",
        # marker="",
        linestyle="--",
        label="Train",
        color=model_type_to_palette_color("pretrained"),
    )
    sns.lineplot(
        data=test_dataframe,
        x="Superbatch Size",
        y="Score",
        errorbar="ci",
        # marker="",
        linestyle="-",
        label="Test",
        color=model_type_to_palette_color("superhf"),
    )

    # Set x-axis to log
    plt.xscale("log")

    # Disable markers
    # plt.rcParams["lines.marker"] = None

    # More x-ticks
    ticks = [2**i for i in range(0, 8)]
    plt.xticks(ticks, ticks)

    # Bounds
    # plt.xlim(ticks[0], ticks[-1])

    # Set labels and title
    plt.xlabel("Superbatch Size (Best-of-N)")
    plt.ylabel("Score")
    plt.title("SuperHF Superbatch Size Ablation")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
