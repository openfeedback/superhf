"""
Plot test reward along with METEOR score similarity.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from chart_utils import (
    get_test_scores,
    initialize_plot,
    model_type_to_palette_color,
    save_plot,
    # LLAMA_TEST_REWARD,
)
from superhf.utils import bootstrap_meteor_similarity_from_completions

OUTPUT_FILE = "./charts/ablations/kl_meteor_similarity.png"
TEST_COMPLETIONS_DIRECTORY = "./experiments/evaluations/test_completions/"
TEST_SCORES_DIRECTORY = "./experiments/evaluations/test_scores/"
NUM_BOOTSTRAP_SAMPLES = 100


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    # Initialize
    initialize_plot()
    plt.rcParams["lines.markersize"] = 1
    color_reward = model_type_to_palette_color("SuperHF")
    color_similarity = model_type_to_palette_color("ftp")

    # Define the model sizes and their corresponding file names
    model_types_to_test_names = {"SuperHF": "shf-v4-llama-kl-"}

    # Find all the files in the test scores directory
    reward_data = []
    meteor_data = []
    for model_type, file_name_template in tqdm(
        model_types_to_test_names.items(), desc="Models"
    ):
        for file_name in os.listdir(TEST_SCORES_DIRECTORY):
            if file_name.startswith(file_name_template):
                kl_coefficient = file_name.split("-")[-1].split(".json")[0]
                score_file_path = os.path.join(TEST_SCORES_DIRECTORY, file_name)
                scores = get_test_scores(score_file_path)
                labeled_scores = [
                    [model_type, kl_coefficient, score] for score in scores
                ]
                reward_data.extend(labeled_scores)
                completion_file_path = os.path.join(
                    TEST_COMPLETIONS_DIRECTORY, file_name
                )
                meteor_scores = bootstrap_meteor_similarity_from_completions(
                    completion_file_path, NUM_BOOTSTRAP_SAMPLES
                )
                labeled_meteor_scores = [
                    [model_type, kl_coefficient, score] for score in meteor_scores
                ]
                meteor_data.extend(labeled_meteor_scores)

    dataframe_scores = pd.DataFrame(
        reward_data, columns=["Model", "KL-Coefficient", "Test Reward →"]
    )

    # Create the plot
    ax1 = sns.lineplot(
        data=dataframe_scores,
        x="KL-Coefficient",
        y="Test Reward →",
        errorbar="ci",
        # capsize=0.1,
        # marker="",
        # label="Pythia Base",
        # palette=[
        #     model_type_to_palette_color(m) for m in dataframe_scores["Model"].unique()
        # ],
        color=color_reward,
    )

    # Twin
    ax2 = plt.twinx()
    dataframe_meteor = pd.DataFrame(
        meteor_data, columns=["Model", "KL-Coefficient", "Meteor Similarity ←"]
    )
    sns.lineplot(
        data=dataframe_meteor,
        x="KL-Coefficient",
        y="Meteor Similarity ←",
        errorbar="ci",
        # capsize=0.1,
        # marker="",
        # label="Pythia Base",
        # palette=[
        #     model_type_to_palette_color(m) for m in dataframe_meteor["Model"].unique()
        # ],
        color=model_type_to_palette_color("ftp"),
        ax=ax2,
    )

    # Color each y axis with the color of the corresponding line
    # ax1.spines["left"].set_color(color_reward)
    ax1.tick_params(axis="y", colors=color_reward)
    ax1.set_ylabel("Test Reward →", color=color_reward)
    # ax2.spines["right"].set_color(color_similarity)
    ax2.tick_params(axis="y", colors=color_similarity)
    ax2.set_ylabel("Meteor Similarity ←", color=color_similarity)

    # Turn off the grid for the second y axis
    ax2.grid(False)

    # Set labels and title
    plt.xlabel("KL Coefficient")
    # plt.ylabel("Test Score →")
    plt.title("KL Regularization of Model Diversity")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
