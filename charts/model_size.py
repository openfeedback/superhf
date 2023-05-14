"""
Plot scores as a function of model size.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    create_file_dir_if_not_exists,
    flatten_2d_vector,
    load_json,
    set_plot_style,
    model_type_to_palette_color,
)
from superhf.utils import set_seed

OUTPUT_FILE = "./charts/shf_ablations/model_size.png"


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    # Initialize
    set_seed(66)
    set_plot_style()

    # Define the model sizes and their corresponding file names
    file_names = [
        "shf-pythia-70m.json",
        "shf-pythia-160m.json",
        "shf-pythia-410m.json",
        "shf-pythia-1b.json",
        "shf-pythia-1.4b.json",
        "shf-pythia-2.8b.json",
        "shf-pythia-6.9b.json",
        "shf-pythia-12b.json",
    ]
    parameters = [7e7, 1.6e8, 4.1e8, 1e9, 1.4e9, 2.8e9, 6.9e9, 1.2e10]

    # Calculate plot values for data
    all_data = []
    for parameter, file_name in zip(parameters, file_names):
        # TODO refactor this into chart_utils.py
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

        # Add the data
        for score in scores:
            all_data.append([parameter, score])

    dataframe = pd.DataFrame(all_data, columns=["Parameter", "Score"])

    # Create the plot
    errorbar = "ci"
    sns.lineplot(
        data=dataframe,
        x="Parameter",
        y="Score",
        errorbar=errorbar,
        color=model_type_to_palette_color("superhf"),
    )
    # sns.lineplot(x=parameters, y=mean_scores, label="SuperHF")

    # Set x-axis to log
    plt.xscale("log")

    # More x-ticks
    # plt.xticks(
    #     [1e8, 3e8, 1e9, 3e9, 1e10],
    #     ["100M", "300M", "1B", "3B", "10B"],
    # )
    plt.xticks(parameters, ["70M", "160M", "410M", "1B", "1.4B", "2.8B", "6.9B", "12B"])

    # Set labels and title
    plt.xlabel("Parameters")
    plt.ylabel("Test score")
    plt.title("SuperHF scaling (Pythia model suite)")

    # Save the plot
    create_file_dir_if_not_exists(OUTPUT_FILE)
    plt.savefig(OUTPUT_FILE)


if __name__ == "__main__":
    main()
