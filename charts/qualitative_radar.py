"""Radar chart of qualitative data."""

import math
from typing import Any

import jsonlines
import pandas as pd
import matplotlib.pyplot as plt

from chart_utils import (
    initialize_plot,
    save_plot,
    MODEL_NAME_MAPPING,
    QUALITATIVE_MODEL_ORDER,
)

SHOW_LIMITED_MODELS = True

OUTPUT_FILE = "./charts/qualitative/qualitative_radar_all.png"
if SHOW_LIMITED_MODELS:
    OUTPUT_FILE = "./charts/qualitative/qualitative_radar_limited.png"


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a jsonlines file into a DataFrame.
    """
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    return pd.DataFrame(data)


def main() -> None:
    """Main function."""
    initialize_plot()
    plt.rcParams["lines.marker"] = ""
    plt.figure(figsize=(8, 8))

    # Load data
    named_files = [
        ("Avoidance", "eval_results/gpt4_qualitative/avoidance.jsonl"),
        ("Bias", "eval_results/gpt4_qualitative/bias.jsonl"),
        ("Reward\nGaming", "eval_results/gpt4_qualitative/gaming.jsonl"),
        ("Relevance", "eval_results/gpt4_qualitative/relevance.jsonl"),
    ]
    tabular_data: dict[str, Any] = {"group": QUALITATIVE_MODEL_ORDER}
    # Manually add Elo scores
    tabular_data["Elo Score"] = [
        1220.91,  # LLaMA
        1507.60,  # Alpaca
        1311.50,  # SFT
        1444.27,  # RLHF
        1527.14,  # SuperHF
        1711.37,  # GPT-3.5
        1777.20,  # GPT-4
    ]
    for quality_name, file_path in named_files:
        data = load_data(file_path)
        # Drop rows where the rating isn't a 1-10 number
        data = data[data["rating"].str.match(r"^[1-9]$|^10$")]
        data["rating"] = data["rating"].astype(int)
        # Calculate the average per model
        avg_data = data.groupby("model")["rating"].mean().reset_index()
        # Sort by model order using the mapping
        avg_data["model"] = avg_data["model"].map(MODEL_NAME_MAPPING)
        avg_data = avg_data.sort_values(
            "model", key=lambda x: x.map(dict(zip(QUALITATIVE_MODEL_ORDER, range(7))))
        )
        tabular_data[quality_name] = avg_data["rating"].tolist()
    dataframe = pd.DataFrame(tabular_data)

    if SHOW_LIMITED_MODELS:
        # Drop all rows but Alpaca, RLHF, and SuperHF
        dataframe = dataframe.iloc[[1, 3, 4]]

    # Normalize each group to go from 0.1 (min) to 1 (max)
    qualities = ["Elo Score", "Avoidance", "Bias", "Reward\nGaming", "Relevance"]
    for quality_name in qualities:
        dataframe[quality_name] = (
            dataframe[quality_name] - dataframe[quality_name].min()
        ) / (dataframe[quality_name].max() - dataframe[quality_name].min())
        if SHOW_LIMITED_MODELS:
            dataframe[quality_name] = dataframe[quality_name] * 0.5 + 0.5
        else:
            dataframe[quality_name] = dataframe[quality_name] * 0.9 + 0.1

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

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    for model_name in dataframe["group"]:
        values = (
            dataframe[dataframe["group"] == model_name]
            .drop("group", axis=1)
            .values.flatten()
            .tolist()
        )
        values += values[:1]
        line_style = "solid" if SHOW_LIMITED_MODELS else "dashed"
        axis.plot(angles, values, linewidth=3, linestyle=line_style, label=model_name)
        if SHOW_LIMITED_MODELS:
            axis.fill(angles, values, alpha=0.15)
    # values = datafame.loc[0].drop("group").values.flatten().tolist()
    # values += values[:1]
    # ax.plot(angles, values, linewidth=1, linestyle="solid", label="group A")
    # # ax.fill(angles, values, "b", alpha=0.1)

    # # Ind2
    # values = datafame.loc[1].drop("group").values.flatten().tolist()
    # values += values[:1]
    # ax.plot(angles, values, linewidth=1, linestyle="solid", label="group B")
    # ax.fill(angles, values, "r", alpha=0.1)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 1))
    if SHOW_LIMITED_MODELS:
        plt.legend()

    # Show the graph
    # plt.show()

    plt.title("Normalized Qualitative Ratings using GPT-4")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
