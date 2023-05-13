"""
Plot scores as a function of model size.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from chart_utils import (
    # bootstrapped_stdev,
    create_file_dir_if_not_exists,
    flatten_2d_vector,
    load_json,
    set_plot_style,
)
from superhf.utils import set_seed

OUTPUT_FILE = "./charts/shf_ablations/modelsize.png"


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    # Initialize
    set_seed(66)
    set_plot_style()

    # Define the model sizes and their corresponding file names
    file_names = [
        # "shf-pythia-70m.json",
        "shf-pythia-160m.json",
        "shf-pythia-410m.json",
        "shf-pythia-1b.json",
        "shf-pythia-1.4b.json",
        "shf-pythia-2.8b.json",
        "shf-pythia-6.9b.json",
    ]
    parameters = [1.6e8, 4.1e8, 1e9, 1.4e9, 2.8e9, 6.9e9]

    # Initialize lists to hold the data
    mean_scores = []
    all_scores = []
    # stdevs = []

    # Calculate plot values for data
    for file_name in file_names:
        file_path = f"./experiments/evaluations/test_scores/{file_name}"
        file_data = load_json(file_path)
        scores = (
            file_data["anthropic-red-team"]
            + file_data["anthropic-helpful-base"]
            + file_data["anthropic-harmless-base"]
            + file_data["openai/webgpt_comparisons"]
        )
        # Unwrap scores from 2D array
        scores = flatten_2d_vector(scores)

        # Calculate the mean score
        mean_score = np.average(scores)

        # Calculate the standard deviation of the scores
        # stdev = bootstrapped_stdev(scores)

        # Add the data
        all_scores.append(scores)
        mean_scores.append(mean_score)
        # stdevs.append(stdev)

    # Convert data types
    # data["model_size"] = data["model_size"].astype(int)
    # data["score"] = data["score"].astype(float)

    # Initialize arrays to hold the mean scores and the confidence intervals
    # mean_scores = []
    # confidence_intervals = []

    # # For each model size, calculate the mean score and the 95% confidence interval
    # for model_size in model_sizes:
    #     scores = data[data["model_size"] == model_size]["score"]
    #     mean_score = scores.mean()
    #     confidence_interval = sem(scores) * t.ppf((1 + 0.95) / 2.0, scores.count() - 1)
    #     mean_scores.append(mean_score)
    #     confidence_intervals.append(confidence_interval)

    # Create the plot
    sns.lineplot(x=parameters, y=mean_scores, label="SuperHF")

    # Set x-axis to log
    plt.xscale("log")

    # More x-ticks
    plt.xticks(
        [1e8, 3e8, 1e9, 3e9, 1e10],
        ["100M", "300M", "1B", "3B", "10B"],
    )

    # Set labels and title
    plt.xlabel("Parameters")
    plt.ylabel("Average test accuracy")
    plt.title("SuperHF scaling (Pythia model suite)")

    # Display grid
    plt.grid(True)

    # Save the plot
    create_file_dir_if_not_exists(OUTPUT_FILE)
    plt.savefig(OUTPUT_FILE)


if __name__ == "__main__":
    main()
