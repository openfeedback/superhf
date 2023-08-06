"""METEOR similarity of completions across models."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    initialize_plot_bar,
    save_plot,
    model_type_to_palette_color,
    model_type_to_hatch,
)
from superhf.utils import bootstrap_meteor_similarity_from_completions

OUTPUT_FILE = "./charts/models/model_comparison_meteor.png"

BOOTSTRAP_N = 128


def main() -> None:
    """Main function."""

    # Initialize
    initialize_plot_bar()

    # Define the model names
    model_files_to_names = {
        "llama-7b.json": "LLaMA",
        "llama-ftp-49516.json": "FeedME",
        "llama-instruct-12379.json": "Instruct",
        # "rlhf-v4-llama.json":
        "rlhf-v5-llama@step-00384.json": "RLHF\n(LLaMA)",
        "rlhf-fixed-llama-instruct-bs-16.json": "RLHF\n(Instruct)",
        "shf-v4-llama-10000-kl-0.35.json": "SuperHF\n(LLaMA)",
        # "shf-v5-llama-ftp-24758-rm-0.5.json":
        "shf-v4-llama-instruct-10k-kl-0.35.json": "SuperHF\n(Instruct)",
        "alpaca_7b.json": "Alpaca",
        # "sft-on-preferences-v1.json",
        # "test-save-alpaca@model-2048-prompts-batch-size-8.json",
        # "rlhf-v3-lr-5.0e-6-batch-16@gold-run.json",
        # "shf-7b-default.json",
        # "shf-7b-gold-v1.json",
        # "gpt-4_2023-05-13_completions_output.json",
    }
    model_filenames = list(model_files_to_names.keys())
    x_labels = list(model_files_to_names.values())

    # Calculate plot values for data
    all_data = []
    all_mean_similarities = []
    for model_name in model_filenames:
        file_path = f"./experiments/evaluations/test_completions/{model_name}"
        similarities = bootstrap_meteor_similarity_from_completions(
            file_path, BOOTSTRAP_N
        )

        mean_similarity = np.mean(similarities)
        all_mean_similarities.append(mean_similarity)
        print(f"{model_name} mean similiarity: {mean_similarity:.23f}")

        # Add the data
        for similarity in similarities:
            # score = score - Y_MIN
            all_data.append([model_name, similarity])

    dataframe = pd.DataFrame(all_data, columns=["Model", "Similarity"])

    # Create the plot
    errorbar = "ci"
    palette = [model_type_to_palette_color(model_name) for model_name in x_labels]
    hatch = [model_type_to_hatch(model_name) for model_name in x_labels]
    sns.barplot(
        data=dataframe,
        x="Model",
        y="Similarity",
        capsize=0.1,
        errorbar=errorbar,
        palette=palette,
        hatch=hatch,
        # bottom=Y_MIN,
        # color=model_type_to_palette_color("superhf"),
    )

    # Horizontal line at Alpaca
    plt.axhline(
        y=all_mean_similarities[-1], color="black", linestyle="--", label="Alpaca Mean"
    )
    plt.legend()

    # Set labels and title
    plt.xlabel("Model Type")
    plt.ylabel("METEOR Similarity ←")
    plt.title("METEOR Similarity of Completions Across Models")

    # Set x-ticks
    plt.xticks(range(len(x_labels)), x_labels)

    # Print the plot
    # plt.show()

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
