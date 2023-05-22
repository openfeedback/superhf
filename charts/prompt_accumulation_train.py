"""
Plot train scores as a function of the prompt accumulation.
"""

import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    initialize_plot,
    model_type_to_palette_color,
    save_plot,
    normalize_train_scores,
)

INPUT_FILE = "./charts/data/accum_train_scores.csv"
OUTPUT_FILE = "./charts/ablations/prompt_accumulation_train.png"

# 0 doesn't update over training, 3 had a weird training bug due to not power of 2
DENYLIST = [0, 3]


def main() -> None:
    """Main function."""

    # Initialize
    initialize_plot()

    # Load the data
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        data = list(reader)
    assert len(data) > 0

    dataframe = pd.DataFrame(data)

    # Drop columns with __ (MIN and MAX)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("__")]

    # Change columns names from "SHF-Accum-v2-{N} - score_train_avg" to "N"
    dataframe.columns = ["Step"] + [
        int(column.split("-")[3].split(" ")[0]) for column in dataframe.columns[1:]
    ]

    # Change from columns as accum values and rows as steps to just (accum, list of scores)
    dataframe = dataframe.drop(columns=["Step"])
    new_data = {accum: dataframe[accum] for accum in dataframe.columns}

    # Drop empty strings, convert strings to floats, put in a dataframe
    accums_to_scores = []
    for accum, scores in new_data.items():
        if accum in DENYLIST:
            continue
        scores = [float(score) for score in scores if score != ""]
        scores = normalize_train_scores(scores)
        for score in scores:
            accums_to_scores.append([accum, score])
    assert len(accums_to_scores) > 0

    dataframe = pd.DataFrame(
        accums_to_scores, columns=["Prompt Accumulation", "Train Score"]
    )

    # Create the plot
    sns.lineplot(
        data=dataframe,
        x="Prompt Accumulation",
        y="Train Score",
        errorbar="ci",
        # marker="",
        # label="Pythia Base",
        color=model_type_to_palette_color("superhf"),
    )

    # Set x-axis to log
    plt.xscale("log")

    # Disable markers
    # plt.rcParams["lines.marker"] = None

    # More x-ticks
    ticks = sorted([key for key in new_data.keys() if int(key) not in DENYLIST])
    plt.xticks(ticks, ticks)

    # Bounds
    # plt.xlim(ticks[0], ticks[-1])

    # Set labels and title
    plt.xlabel("Prompt Accumulation (Iterativeness)")
    plt.ylabel("Train Score")
    plt.title("SuperHF Prompt Accumulation Ablation")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
