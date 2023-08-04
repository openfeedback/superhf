"""
Plot elo scores as for each model type.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chart_utils import (
    initialize_plot,
    save_plot,
    load_json,
    MODEL_NAME_MAPPING,
    QUALITATIVE_MODEL_ORDER,
)


INPUT_FOLDER = "./eval_results/lm_evals"
OUTPUT_FILE = "./charts/downstream/mmlu_safety_comsense_new.png"


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    # Initialize
    initialize_plot()

    # Store all our data in a list of (category, model, average, error)
    all_data: list[tuple[str, str, float, float]] = []

    for category_folder in ["mmlu", "common_sense", "safety"]:
        category_name = category_folder.replace("_", " ").title()
        if category_folder == "mmlu":
            category_name = "MMLU"

        # Load the data
        subfolder_path = os.path.join(INPUT_FOLDER, category_folder)
        num_loaded = 0
        for file_name in os.listdir(subfolder_path):
            if file_name not in MODEL_NAME_MAPPING:
                continue
            model_name = MODEL_NAME_MAPPING[file_name]
            model_data = load_json(os.path.join(subfolder_path, file_name))
            num_loaded += 1
            accuracies = []
            errors = []
            for result in model_data["results"].values():
                if "acc_norm" in result:
                    accuracies.append(result["acc_norm"])
                    errors.append(result["acc_norm_stderr"])
                elif "mc1" in result:
                    # TruthfulQA
                    accuracies.append(result["mc1"])
                    accuracies.append(result["mc2"])
                    errors.append(result["mc1_stderr"])
                    errors.append(result["mc2_stderr"])
                else:
                    accuracies.append(result["acc"])
                    errors.append(result["acc_stderr"])

            average_acc = np.average(accuracies)
            average_err = np.average(errors)
            all_data.append((category_name, model_name, average_acc, average_err))
            print(f"{category_name} {model_name}: {average_acc:.2f}±{average_err:.2f}")
        print(f"Loaded {num_loaded} models for {category_name}")

    # Create a dataframe
    dataframe = pd.DataFrame(
        all_data, columns=["Category", "Model", "Average", "Error"]
    )

    # Reorder the model names by QUALITATIVE_MODEL_ORDER
    dataframe["Model"] = pd.Categorical(dataframe["Model"], QUALITATIVE_MODEL_ORDER)

    # Create the plot
    plt.rcParams["lines.markersize"] = 1  # No markers for bar plots
    barplot = sns.catplot(
        data=dataframe,
        kind="bar",
        x="Category",
        y="Average",
        hue="Model",
        capsize=0.1,
        # yerr=dataframe["Error"],
        # errorbar="ci",
        # order=QUALITATIVE_MODEL_ORDER,
        # label="Pythia Base",
    )

    # Add error bars
    for i, model in enumerate(dataframe["Model"].unique().categories):
        model_data = dataframe[dataframe["Model"] == model]
        barplot.ax.errorbar(
            x=np.arange(len(model_data)) + i * 0.1 - 0.35,  # x positions
            y=model_data["Average"],  # y positions
            yerr=model_data["Error"],  # error values
            fmt="none",  # format for the error bars
            capsize=3,  # cap size for the error bars
            color="black",  # color of the error bars
        )

    # Set labels and title
    plt.xlabel("Evaluation Category")
    # plt.ylabel("Performance Improvement over LLaMA")
    plt.ylabel("Performance")
    plt.title("Downstream Capabilities and Safety Evaluations")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
