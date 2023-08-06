"""
Plot test reward along with METEOR score similarity.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from chart_utils import (
    get_test_scores,
    initialize_plot,
    model_type_to_palette_color,
    save_plot,
)
from superhf.utils import bootstrap_meteor_similarity_from_completions

OUTPUT_FILE = "./charts/ablations/progress_study_with_kl_meteor.png"
TEST_SCORES_DIRECTORY = "./experiments/evaluations/test_scores/"
TEST_COMPLETIONS_DIRECTORY = "./experiments/evaluations/test_completions/"
START_STEP = 32
END_STEP = 10000
NUM_BOOTSTRAP_SAMPLES = 32


def get_rewards_and_similarity(file_name: str) -> tuple[list[float], list[float]]:
    """Get the test scores and METEOR similarity scores for a given model."""
    score_file_path = os.path.join(TEST_SCORES_DIRECTORY, file_name)
    completion_file_path = os.path.join(TEST_COMPLETIONS_DIRECTORY, file_name)
    rewards = get_test_scores(score_file_path)
    meteor_scores = bootstrap_meteor_similarity_from_completions(
        completion_file_path, NUM_BOOTSTRAP_SAMPLES
    )
    return rewards, meteor_scores


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    # Initialize
    initialize_plot()
    color_reward = model_type_to_palette_color("SuperHF")
    color_similarity = model_type_to_palette_color("ftp")

    # Compute LLaMA's, Instruct's, and Alpaca's mean reward and similarity
    llama_rewards, llama_meteor_scores = get_rewards_and_similarity("llama-7b.json")
    llama_mean_reward = np.mean(llama_rewards)
    llama_mean_meteor_score = np.mean(llama_meteor_scores)
    print(f"LLaMA Mean Test Reward: {llama_mean_reward:.3f}")
    print(f"LLaMA Mean Meteor Score: {llama_mean_meteor_score:.3f}")

    # Wider figure
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.tight_layout()

    # Initialize axes
    ax1 = plt.gca()
    ax2 = plt.twinx()
    # Color each y axis with the color of the corresponding line
    ax1.tick_params(axis="y", colors=color_reward)
    ax1.set_ylabel("Test Reward →", color=color_reward)
    ax2.tick_params(axis="y", colors=color_similarity)
    ax2.set_ylabel("METEOR Similarity ←", color=color_similarity)

    # Define the models and their corresponding file names
    for model_name, file_name_template, linestyle in [
        ("Test Reward (KL-Coefficient = 0.0)", "shf-v5-llama-gold-kl-0.0@", "--"),
        ("Test Reward (KL-Coefficient = 0.35)", "shf-v5-llama-gold-kl-0.35@", "-"),
    ]:
        # Find all the files in the test scores directory
        steps_and_file_names = []
        for file_name in os.listdir(TEST_SCORES_DIRECTORY):
            if file_name.startswith(file_name_template):
                step = int(file_name.split("@step-")[1].split(".")[0])
                if step < START_STEP:
                    continue
                steps_and_file_names.append((step, file_name))

        # Calculate plot values for data
        reward_data = []
        similarity_data = []
        for step, file_name in tqdm(steps_and_file_names, desc=model_name):
            scores, similarities = get_rewards_and_similarity(file_name)
            labeled_scores = [[step, score] for score in scores]
            labeled_meteor_scores = [[step, score] for score in similarities]
            reward_data.extend(labeled_scores)
            similarity_data.extend(labeled_meteor_scores)

        reward_df = pd.DataFrame(reward_data, columns=["Step", "Score"])
        similarity_df = pd.DataFrame(similarity_data, columns=["Step", "Similarity"])

        # Create the plots
        sns.lineplot(
            data=reward_df,
            x="Step",
            y="Score",
            ax=ax1,
            errorbar="ci",
            marker="",
            label=model_name,
            color=color_reward,
            linestyle=linestyle,
        )
        sns.lineplot(
            data=similarity_df,
            x="Step",
            y="Similarity",
            ax=ax2,
            errorbar="ci",
            marker="",
            # label=model_name,
            color=color_similarity,
            linestyle=linestyle,
        )

    # Set x-axis to log
    plt.xscale("log")

    # More x-ticks
    ticks = [
        2**i for i in range(int(np.log2(START_STEP)), int(np.log2(END_STEP)) + 1)
    ]
    plt.xticks(ticks, ticks)

    # Bounds
    plt.xlim(START_STEP, END_STEP)

    # Set labels and title
    plt.xlabel("Training Step")
    plt.title("SuperHF (LLaMA) Training Progress Study")

    # Turn off the grid for the second y axis
    ax2.grid(False)

    # Plot dashed horizontal lines for llama
    plt.rcParams["lines.marker"] = ""
    ax1.axhline(
        llama_mean_reward, color=color_reward, linestyle=":", label="LLaMA Mean Reward"
    )
    ax2.axhline(
        llama_mean_meteor_score,
        color=color_similarity,
        linestyle=":",
        label="LLaMA Mean Similarity",
    )

    # Add empty lines to ax1 so they show up on the legend
    ax1.plot(
        [],
        [],
        color=color_similarity,
        linestyle="--",
        marker="",
        label="Similarity (KL-Coeffcient = 0.0)",
    )
    ax1.plot(
        [],
        [],
        color=color_similarity,
        linestyle="-",
        marker="",
        label="Similarity (KL-Coeffcient = 0.35)",
    )
    ax1.plot(
        [],
        [],
        color=color_similarity,
        linestyle="--",
        marker="",
        label="LLaMA Mean Similarity",
    )
    ax1.legend()

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
