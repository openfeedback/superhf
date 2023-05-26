"""Radar chart of qualitative data but just for the KL sweep and with 4 axes."""

import math
from typing import Any

import jsonlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    initialize_plot,
    save_plot,
)

OUTPUT_FILE_RADAR = "./charts/qualitative/qualitative_radar_kl_sweep.png"
OUTPUT_FILE_BAR = "./charts/qualitative/qualitative_bar_kl_sweep.png"


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a jsonlines file into a DataFrame.
    """
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    return pd.DataFrame(data)


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    initialize_plot()
    plt.rcParams["lines.marker"] = ""
    plt.figure(figsize=(8, 8))

    # Load data
    named_files = [
        ("Avoidance", "eval_results/gpt4_qualitative/kl_sweep/avoidance.jsonl"),
        ("Bias", "eval_results/gpt4_qualitative/kl_sweep/bias.jsonl"),
        ("Reward\nGaming", "eval_results/gpt4_qualitative/kl_sweep/gaming.jsonl"),
        ("Relevance", "eval_results/gpt4_qualitative/kl_sweep/relevance.jsonl"),
    ]
    tabular_data: dict[str, Any] = {
        "group": [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    }
    # Manually add Test scores
    tabular_data["Test Score"] = [
        1.302,
        1.152,
        0.406,
        0.863,
        0.475,
        0.517,
        0.439,
        0.307,
        0.383,
        0.492,
    ]
    for quality_name, file_path in named_files:
        data = load_data(file_path)
        # Drop rows where the rating isn't a 1-10 number
        data = data[data["rating"].str.match(r"^[1-9]$|^10$")]
        data["rating"] = data["rating"].astype(int)
        # Calculate the average per model
        avg_data = data.groupby("model")["rating"].mean().reset_index()
        # Go from shf-7b-kl-0.15.json to 0.15
        avg_data["model"] = avg_data["model"].str.extract(r"kl-(\d+\.\d+)")
        # Sort by model order
        avg_data = avg_data.sort_values(
            "model", key=lambda x: x.map(dict(zip(tabular_data["group"], range(10))))
        )
        tabular_data[quality_name] = avg_data["rating"].tolist()
    dataframe = pd.DataFrame(tabular_data)

    # Normalize each group to go from 0.2 (min) to 1 (max)
    qualities = ["Test Score", "Avoidance", "Bias", "Reward\nGaming", "Relevance"]
    for quality_name in qualities:
        dataframe[quality_name] = (
            dataframe[quality_name] - dataframe[quality_name].min()
        ) / (dataframe[quality_name].max() - dataframe[quality_name].min())
        dataframe[quality_name] = dataframe[quality_name] * 0.8 + 0.2

    # number of variable
    categories = list(dataframe)[1:]
    num_categories = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(num_categories) * 2 * math.pi for n in range(num_categories)]
    angles += angles[:1]

    # Initialise the spider plot
    axis = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    axis.set_theta_offset(math.pi / 2)
    axis.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    axis.set_rlabel_position(0)
    # plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    plt.ylim(0, 1.0)
    # plt.yticks[0.1,]

    # Move x-axis labels away from the plot
    plt.tick_params(axis="x", pad=15)

    # Choose colors on a scale using a seaborn palette
    colors = sns.color_palette("viridis", len(dataframe["group"]))

    # Plot each individual = each line of the data
    for model_name, color in zip(dataframe["group"], colors):
        values = (
            dataframe[dataframe["group"] == model_name]
            .drop("group", axis=1)
            .values.flatten()
            .tolist()
        )
        values += values[:1]
        line_style = "solid"
        axis.plot(
            angles,
            values,
            linewidth=3,
            linestyle=line_style,
            label=model_name,
            color=color,
        )

    # Add legend
    plt.legend(loc="lower right", bbox_to_anchor=(1.1, 0.1))

    plt.title("Normalized Qualitative Ratings using GPT-4 (KL Sweep)")

    # Save the plot
    save_plot(OUTPUT_FILE_RADAR)

    # Reset the plot
    plt.clf()

    # We'll now calculate the average of the qualitative ratings
    # for each model and plot them as a bar chart

    # Drop the test score since it's not a qualitative rating
    dataframe = dataframe.drop("Test Score", axis=1)
    assert dataframe is not None

    # Group the scores in a list
    # pylint: disable=unsubscriptable-object
    new_data = []
    for quality_name in qualities:
        if quality_name == "Test Score":
            continue
        for kl_value in dataframe["group"]:
            new_data.append(
                (
                    kl_value,
                    dataframe[dataframe["group"] == kl_value][quality_name].values[0],
                )
            )
    dataframe_bar = pd.DataFrame(new_data, columns=["group", "Qualitative Scores"])

    # # Calculate the average of each model
    # dataframe["average"] = dataframe.drop("group", axis=1).mean(axis=1)
    # dataframe["std"] = dataframe.drop("group", axis=1).std(axis=1)

    # Plot the average of each model
    sns.barplot(
        x="group",
        y="Qualitative Scores",
        data=dataframe_bar,
        palette=sns.color_palette("viridis", len(dataframe["group"])),
        errorbar="ci",
        capsize=0.1,
    )

    # Add labels
    plt.xlabel("KL Coefficient")
    plt.ylabel("Average Qualitative Rating")

    # Add a title
    plt.title("Average Qualitative Ratings using GPT-4 (KL Sweep)")

    # Save the plot
    save_plot(OUTPUT_FILE_BAR)


if __name__ == "__main__":
    main()
