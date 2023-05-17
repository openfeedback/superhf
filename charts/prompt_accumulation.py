"""
Plot scores as a function of the prompt accumulation.
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

OUTPUT_FILE = "./charts/ablations/prompt_accumulation.png"
TEST_SCORES_DIRECTORY = "./experiments/evaluations/test_scores/"


def main() -> None:
    """Main function."""

    # Initialize
    initialize_plot()

    # Define the model sizes and their corresponding file names
    file_name_template = "shf-7b-accum-"

    # Find all the files in the test scores directory
    accums_and_file_paths = []
    for file_name in os.listdir(TEST_SCORES_DIRECTORY):
        if file_name.startswith(file_name_template):
            accum = int(file_name.split(file_name_template)[1].split(".")[0])
            accums_and_file_paths.append(
                (accum, os.path.join(TEST_SCORES_DIRECTORY, file_name))
            )

    # Calculate plot values for data
    shf_data = []
    for accum, file_path in accums_and_file_paths:
        # Handle 0, representing it as a break to the right of the graph
        if accum == 0:
            accum = 0.3
        scores = get_test_scores(file_path)
        labeled_scores = [[accum, score] for score in scores]
        shf_data.extend(labeled_scores)

    dataframe = pd.DataFrame(shf_data, columns=["Prompt Accumulation", "Score"])

    # Create the plot
    sns.lineplot(
        data=dataframe,
        x="Prompt Accumulation",
        y="Score",
        errorbar="se",
        # marker="",
        # label="Pythia Base",
        color=model_type_to_palette_color("superhf"),
    )

    # Set x-axis to log
    plt.xscale("log")

    # Disable markers
    # plt.rcParams["lines.marker"] = None

    # More x-ticks
    # ticks = [2**i for i in range(3, 14)]
    # plt.xticks(ticks, ticks)

    # Bounds
    # plt.xlim(ticks[0], ticks[-1])

    # Set labels and title
    plt.xlabel("Prompt Accumulation (Iterativeness)")
    plt.ylabel("Test Score")
    plt.title("SuperHF Prompt Accumulation Ablation")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
