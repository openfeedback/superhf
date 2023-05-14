"""
Plot scores as a function of model size.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    get_test_scores,
    initialize_plot,
    model_type_to_palette_color,
    save_plot,
)

OUTPUT_FILE = "./charts/shf_ablations/model_size.png"


def main() -> None:
    """Main function."""

    # Initialize
    initialize_plot()

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
        file_path = f"./experiments/evaluations/test_scores/{file_name}"
        scores = get_test_scores(file_path)
        named_scores = [[parameter, score] for score in scores]
        all_data.extend(named_scores)

    dataframe = pd.DataFrame(all_data, columns=["Parameter", "Score"])

    # Create the plot
    sns.lineplot(
        data=dataframe,
        x="Parameter",
        y="Score",
        errorbar="ci",
        color=model_type_to_palette_color("superhf"),
    )

    # Set x-axis to log
    plt.xscale("log")

    # More x-ticks
    plt.xticks(parameters, ["70M", "160M", "410M", "1B", "1.4B", "2.8B", "6.9B", "12B"])

    # Set labels and title
    plt.xlabel("Parameters")
    plt.ylabel("Test score")
    plt.title("SuperHF scaling (Pythia model suite)")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
