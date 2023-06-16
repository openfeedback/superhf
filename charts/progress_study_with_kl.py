"""
Plot scores as a function of the gold training checkpoint.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chart_utils import (
    get_test_scores,
    initialize_plot,
    # model_type_to_palette_color,
    save_plot,
)

OUTPUT_FILE = "./charts/ablations/progress_study_with_kl.png"
TEST_SCORES_DIRECTORY = "./experiments/evaluations/test_scores/"
START_STEP = 32
END_STEP = 10000


def main() -> None:
    """Main function."""

    # Initialize
    initialize_plot()

    # Define the models and their corresponding file names
    for model_name, file_name_template in [
        ("KL-Coefficient = 0.0", "shf-v5-llama-gold-kl-0.0@"),
        ("KL-Coefficient = 0.35", "shf-v5-llama-gold-kl-0.35@"),
    ]:
        # Find all the files in the test scores directory
        file_paths_and_steps = []
        for file_name in os.listdir(TEST_SCORES_DIRECTORY):
            if file_name.startswith(file_name_template):
                step = int(file_name.split("@step-")[1].split(".")[0])
                if step < START_STEP:
                    continue
                file_paths_and_steps.append(
                    (step, os.path.join(TEST_SCORES_DIRECTORY, file_name))
                )

        # Calculate plot values for data
        shf_data = []
        for step, file_path in file_paths_and_steps:
            scores = get_test_scores(file_path)
            labeled_scores = [[step, score] for score in scores]
            shf_data.extend(labeled_scores)

        dataframe = pd.DataFrame(shf_data, columns=["Step", "Score"])

        # Wider figure
        plt.rcParams["figure.figsize"] = (12, 5)
        plt.tight_layout()

        # Create the plot
        sns.lineplot(
            data=dataframe,
            x="Step",
            y="Score",
            errorbar="ci",
            marker="",
            label=model_name,
            # color=model_type_to_palette_color("superhf"),
        )

    # Set x-axis to log
    plt.xscale("log")

    # Disable markers
    # plt.rcParams["lines.marker"] = None

    # More x-ticks
    ticks = [
        2**i for i in range(int(np.log2(START_STEP)), int(np.log2(END_STEP)) + 1)
    ]
    plt.xticks(ticks, ticks)

    # Bounds
    plt.xlim(START_STEP, END_STEP)

    # Set labels and title
    plt.xlabel("Training Step")
    plt.ylabel("Test Reward")
    plt.title("SuperHF Training Progress Study")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
