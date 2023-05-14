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

OUTPUT_FILE = "./charts/models/model_size.png"


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
        "pythia-70m-deduped.json",
        "pythia-160m-deduped.json",
        "pythia-410m-deduped.json",
        "pythia-1b-deduped.json",
        "pythia-1.4b-deduped.json",
        "pythia-2.8b-deduped.json",
        "pythia-6.9b-deduped.json",
        "pythia-12b-deduped.json",
    ]
    parameters = [7e7, 1.6e8, 4.1e8, 1e9, 1.4e9, 2.8e9, 6.9e9, 1.2e10]

    # Calculate plot values for data
    shf_data = []
    pretrained_data = []
    for parameter, file_name in zip(parameters * 2, file_names):
        file_path = f"./experiments/evaluations/test_scores/{file_name}"
        scores = get_test_scores(file_path)
        named_scores = [[parameter, score] for score in scores]
        if "shf" in file_name:
            shf_data.extend(named_scores)
        else:
            pretrained_data.extend(named_scores)

    dataframe_shf = pd.DataFrame(shf_data, columns=["Parameter", "Score"])
    dataframe_pretrained = pd.DataFrame(pretrained_data, columns=["Parameter", "Score"])

    # Create the plot
    sns.lineplot(
        data=dataframe_pretrained,
        x="Parameter",
        y="Score",
        errorbar="ci",
        label="Pythia Base",
        color=model_type_to_palette_color("pretrained"),
    )
    sns.lineplot(
        data=dataframe_shf,
        x="Parameter",
        y="Score",
        errorbar="ci",
        label="SuperHF",
        color=model_type_to_palette_color("superhf"),
    )

    # Legend
    plt.legend(loc="upper left")

    # Set x-axis to log
    plt.xscale("log")

    # More x-ticks
    plt.xticks(parameters, ["70M", "160M", "410M", "1B", "1.4B", "2.8B", "6.9B", "12B"])

    # Set labels and title
    plt.xlabel("Parameters")
    plt.ylabel("Test Score")
    plt.title("SuperHF Scaling (Pythia Model Suite)")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
