"""
Functions to help with creating charts.
"""

import os
import json
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from superhf.utils import set_seed

DEFAULT_COLOR_PALETTE = "colorblind"

MODEL_NAME_MAPPING = {
    "llama-7b.json": "LLaMA",
    "llama-ftp-49516.json": "FeedME",
    "llama-instruct-12379.json": "Instruct",
    "rlhf-fixed-llama-v3-bs-16.json": "RLHF (LLaMA)",
    "rlhf-fixed-llama-instruct-bs-16.json": "RLHF (Instruct)",
    "shf-v4-llama-10000-kl-0.35.json": "SuperHF (LLaMA)",
    "shf-v4-llama-instruct-10k-kl-0.35.json": "SuperHF (Instruct)",
    "alpaca_7b.json": "Alpaca",
}

QUALITATIVE_MODEL_ORDER = [
    "LLaMA",
    "FeedME",
    "Instruct",
    "RLHF (LLaMA)",
    "RLHF (Instruct)",
    "SuperHF (LLaMA)",
    "SuperHF (Instruct)",
    "Alpaca",
]


def spaces_to_newlines(model_name_list: list[str]) -> list[str]:
    """Replace spaces with newlines in a list of model names."""
    return [model_name.replace(" ", "\n") for model_name in model_name_list]


QUALITATIVE_MODEL_ORDER_MULTILINE = spaces_to_newlines(QUALITATIVE_MODEL_ORDER)

LLAMA_TEST_REWARD = -0.51  # TODO get accurate reward
ALPACA_TEST_REWARD = -0.01  # TODO redo with new data


def bootstrapped_stdev(data: list[Any], num_samples: int = 1000) -> Any:
    """
    Bootstrap a stdev by sampling the whole dataset with replacement N times.

    We calculate the average of each sample, then take the stdev of the averages.
    """
    averages = []
    for _ in range(num_samples):
        # Sample the data with replacement
        sample = np.random.choice(data, size=len(data), replace=True)

        # Calculate the average of the sample
        average = np.average(sample)

        # Add the average to the array
        averages.append(average)

    # Calculate the standard deviation of the averages
    stdev = np.std(averages)

    return stdev


def flatten_2d_vector(vector: list[list[Any]]) -> list[Any]:
    """Flatten a 2D list of single-element lists into a 1D list."""
    output = [element[0] for element in vector]
    return output


def load_json(file_path: str) -> dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, encoding="utf-8") as file:
        file_data = json.load(file)
    assert isinstance(file_data, dict)
    return file_data


def create_file_dir_if_not_exists(file_path: str) -> None:
    """Create the directory for a file if it doesn't already exist."""
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def normalize_train_scores(scores: list[float]) -> list[float]:
    """Normalize the train scores to the test score scale."""
    # See experiments/evaluations/find_average_train_and_test_rm_scores.py
    sd_ratio_test_over_train = 1.94 / 2.44
    return [(score - -2.65) * sd_ratio_test_over_train for score in scores]


def get_test_scores(file_path: str) -> list[Any]:
    """Get the test scores from a file."""
    file_data = load_json(file_path)
    scores = (
        file_data["anthropic-red-team"]
        + file_data["anthropic-helpful-base"]
        + file_data["anthropic-harmless-base"]
        + file_data["openai/webgpt_comparisons"]
    )
    # Unwrap scores from 2D array
    output = flatten_2d_vector(scores)

    # Normalize scores (see experiments/evaluations/find_average_train_and_test_rm_scores.py)
    assert "train_scores" in file_path or "test_scores" in file_path
    if "train_scores" in file_path:
        output = normalize_train_scores(output)
    elif "test_scores" in file_path:
        output = [score - -2.22 for score in output]

    # Add the data
    return output


def initialize_plot() -> None:
    """Set default plot styling."""
    # Set seed
    set_seed(66)
    # Default theme
    sns.set_theme(context="paper", font_scale=1.5, style="whitegrid")
    # Figure size
    plt.rcParams["figure.figsize"] = (8, 5)
    # Make title larger
    plt.rcParams["axes.titlesize"] = 16
    # Higher DPI
    plt.rcParams["figure.dpi"] = 300
    # Default marker
    plt.rcParams["lines.marker"] = "o"
    # Default marker size
    plt.rcParams["lines.markersize"] = 8
    # Accessible colors
    sns.set_palette(DEFAULT_COLOR_PALETTE)


def set_plot_style() -> None:
    """Deprecated. Use initialize_plot() instead."""
    raise DeprecationWarning("Use initialize_plot() instead.")


def _get_color_from_palette(index: int) -> Any:
    """Get a color from the default palette."""
    palette = sns.color_palette(DEFAULT_COLOR_PALETTE)
    color = palette[index]
    return color


def model_type_to_palette_color(model_type: str) -> Any:
    """Standardize our use of models types to palette colors."""
    model_type = model_type.lower()
    model_name_to_type = {
        "ftp": "ftp_preferences",
        "feedme": "ftp_preferences",
        "rlhf": "rlhf",
        "superhf": "superhf",
        "shf": "superhf",
        "gpt-4": "gpt-3.5",
        "llama": "pretrained",
        "alpaca": "alpaca",
    }
    for model_name, model_value in model_name_to_type.items():
        if model_name in model_type:
            model_type = model_value
    all_model_types = [
        "pretrained",
        "instruct",
        "ftp_preferences",
        "rlhf",
        "superhf",
        "gpt-3.5",
        "gpt-4",
        "alpaca",
    ]
    assert model_type in all_model_types
    return _get_color_from_palette(all_model_types.index(model_type))


def model_type_to_hatch(model_type: str, num_hatches: int = 3) -> Any:
    """Standardize our hatching"""
    if "Instruct" in model_type or "Alpaca" in model_type:
        return "/" * num_hatches
    return ""


def model_type_to_line_style(model_type: str) -> Any:
    """Standardize our hatching"""
    if "Instruct" in model_type or "Alpaca" in model_type:
        return "--"
    return "-"


def save_plot(file_path: str) -> None:
    """Save a plot to a file."""
    create_file_dir_if_not_exists(file_path)
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
