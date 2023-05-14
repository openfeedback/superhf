"""
Functions to help with creating charts.
"""

import os
import json
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_COLOR_PALETTE = "colorblind"


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


def get_test_scores(file_path: str) -> list[float]:
    """Get the test scores from a file."""
    output = []
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
        output.append(score)
    return output


def set_plot_style() -> None:
    """Set default plot styling."""
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


def _get_color_from_palette(index: int) -> Any:
    """Get a color from the default palette."""
    palette = sns.color_palette(DEFAULT_COLOR_PALETTE)
    color = palette[index]
    return color


def model_type_to_palette_color(model_type: str) -> Any:
    """Standardize our use of models types to palette colors."""
    # TODO change to dict[str, int] and add final run names as options
    all_model_types = [
        "pretrained",
        "instruct",
        "sft_preferences",
        "rlhf",
        "superhf",
    ]
    assert model_type in all_model_types
    return _get_color_from_palette(all_model_types.index(model_type))
