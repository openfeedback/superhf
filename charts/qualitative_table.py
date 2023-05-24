"""Table of qualitative data."""

from typing import Any

import jsonlines
import pandas as pd
from tabulate import tabulate

from chart_utils import (
    MODEL_NAME_MAPPING,
    QUALITATIVE_MODEL_ORDER,
)

SHOW_LIMITED_MODELS = True

OUTPUT_FILE = "./charts/qualitative/qualitative_table_all.tex"


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a jsonlines file into a DataFrame.
    """
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    return pd.DataFrame(data)


def main() -> None:
    """Main function."""

    # Load data
    named_files = [
        ("Avoidance", "eval_results/gpt4_qualitative/avoidance.jsonl"),
        ("Bias", "eval_results/gpt4_qualitative/bias.jsonl"),
        ("Reward Gaming", "eval_results/gpt4_qualitative/gaming.jsonl"),
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
        # Calculate the mean and std per model
        avg_data = data.groupby("model")["rating"].mean().reset_index()
        std_data = data.groupby("model")["rating"].std().reset_index()
        # Sort by model order using the mapping
        avg_data["model"] = avg_data["model"].map(MODEL_NAME_MAPPING)
        avg_data = avg_data.sort_values(
            "model", key=lambda x: x.map(dict(zip(QUALITATIVE_MODEL_ORDER, range(7))))
        )
        std_data["model"] = std_data["model"].map(MODEL_NAME_MAPPING)
        std_data = std_data.sort_values(
            "model", key=lambda x: x.map(dict(zip(QUALITATIVE_MODEL_ORDER, range(7))))
        )
        # Write mean±std for each model
        tabular_data[quality_name] = [
            rf"{avg:.2f}$\pm${std:.2f}"
            for avg, std in zip(avg_data["rating"], std_data["rating"])
        ]
    dataframe = pd.DataFrame(tabular_data)

    # Save the plot
    print(tabulate(dataframe.transpose(), tablefmt="grid").replace(r"\pm", "±"))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(
            tabulate(dataframe, tablefmt="latex_raw", headers=dataframe.columns[1:])
        )


if __name__ == "__main__":
    main()
