"""Intermediate data for the GPT-4 qualitative evals."""

import json
import os
import random
from collections import defaultdict
import csv
from typing import Any

import jsonlines

from evaluation_utils import create_file_dir_if_not_exists
from superhf.utils import set_seed

OUTPUT_DIR = "./eval_results/gpt4_qualitative"


def load_data(file_path: str) -> list[dict[str, Any]]:
    """
    Load data from a jsonlines file.

    Args:
        file_path (str): Path to the jsonlines file.

    Returns:
        List[Dict]: A list of dictionaries containing the data.
    """
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def calculate_win_rates(data: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """
    Calculate win rates between each model and each other model.

    Args:
        data (List[Dict]): A list of dictionaries containing the data.

    Returns:
        Dict[Tuple[str, str], float]: A dictionary of tuples of model pairs to win rates.
    """
    win_counts: dict[tuple[str, str], float] = defaultdict(int)
    total_counts: dict[tuple[str, str], float] = defaultdict(int)

    for entry in data:
        if entry["rating"] in ["A", "B"]:
            model_a = entry["model_a"]
            model_b = entry["model_b"]
            winner = model_a if entry["rating"] == "A" else model_b
            win_counts[(model_a, model_b)] += winner == model_a
            win_counts[(model_b, model_a)] += winner == model_b
            total_counts[(model_a, model_b)] += 1
            total_counts[(model_b, model_a)] += 1

    win_rates = {pair: win_counts[pair] / total_counts[pair] for pair in total_counts}
    return win_rates


def save_win_rates_to_csv(
    win_rates: dict[tuple[str, str], float], file_path: str
) -> None:
    """
    Save win rates to a CSV file.

    Args:
        win_rates (Dict[Tuple[str, str], float]): A dictionary with keys as tuples of model pairs
            and values as win rates.
        file_path (str): Path to the CSV file.
    """
    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["model_a", "model_b", "win_rate"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pair, win_rate in win_rates.items():
            writer.writerow(
                {"model_a": pair[0], "model_b": pair[1], "win_rate": win_rate}
            )


def elo_update(
    rating_a: float, rating_b: float, result_a: int, k: float = 32
) -> tuple[float, float]:
    """
    Update Elo ratings based on the result of a match.

    Args:
        rating_a (float): Elo rating of player A.
        rating_b (float): Elo rating of player B.
        result_a (int): 1 if player A wins, 0 if player B wins.
        k (float, optional): K-factor for Elo rating update. Defaults to 32.

    Returns:
        Tuple[float, float]: Updated Elo ratings for player A and player B.
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))
    rating_a += k * (result_a - expected_a)
    rating_b += k * ((1 - result_a) - expected_b)
    return rating_a, rating_b


def bootstrap_elo(
    data: list[dict[str, Any]], iterations: int = 1000
) -> dict[str, list[float]]:
    """
    Calculate bootstrapped Elo scores for each model.

    Args:
        data (List[Dict]): A list of dictionaries containing the data.
        iterations (int, optional): Number of iterations for bootstrapping. Defaults to 1000.

    Returns:
        Dict[str, List[float]]: A dictionary of model names to lists of Elo scores.
    """
    models = set(entry["model_a"] for entry in data) | set(
        entry["model_b"] for entry in data
    )
    elo_scores: dict[str, Any] = {model: [] for model in models}

    # Hack to not include GPT-3.5 and GPT-4 in the Elo calculations
    denylist = []
    denylist = [
        "gpt-3.5-turbo_2023-05-13_completions_output.json",
        "gpt-4_2023-05-13_completions_output.json",
    ]

    for _ in range(iterations):
        random.shuffle(data)
        ratings = {model: 1500.0 for model in models}

        for entry in data:
            if entry["rating"] in ["A", "B"]:
                model_a = entry["model_a"]
                model_b = entry["model_b"]
                if model_a in denylist or model_b in denylist:
                    continue
                result_a = entry["rating"] == "A"
                ratings[model_a], ratings[model_b] = elo_update(
                    ratings[model_a], ratings[model_b], result_a
                )

        for model in models:
            elo_scores[model].append(ratings[model])

    return elo_scores


def save_elo_scores_to_file(elo_scores: dict[str, list[float]], file_path: str) -> None:
    """
    Save Elo scores to a file.

    Args:
        elo_scores (Dict[str, List[float]]): A dictionary with keys as model names and values as
            lists of Elo scores.
        file_path (str): Path to the file.
    """
    # Sort the elo scores by model name since dict is unordered
    elo_scores = {model: elo_scores[model] for model in sorted(elo_scores)}
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(elo_scores, file, indent=4)


def main() -> None:
    """Main function."""
    set_seed(66)
    data_file = "eval_results/gpt4_qualitative/preferences.jsonl"
    win_rates_csv = os.path.join(OUTPUT_DIR, "win_rates.csv")
    elo_scores_file = os.path.join(OUTPUT_DIR, "elo_scores.json")
    create_file_dir_if_not_exists(win_rates_csv)

    data = load_data(data_file)
    win_rates = calculate_win_rates(data)
    save_win_rates_to_csv(win_rates, win_rates_csv)
    elo_scores = bootstrap_elo(data)
    save_elo_scores_to_file(elo_scores, elo_scores_file)


if __name__ == "__main__":
    main()
