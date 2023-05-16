"""Graph the reward as a function of KL coefficient."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from chart_utils import (
    initialize_plot,
    model_type_to_palette_color,
    get_test_scores,
    save_plot,
)


OUTPUT_FILE = "./charts/ablations/kl_coefficient.png"


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    # Initialize
    initialize_plot()

    # Define the model names
    kl_coefficients = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # Calculate plot values for data
    all_data = []
    all_data2 = []  # For the second dataset
    for kl_coefficient in kl_coefficients:
        url = f"experiments/evaluations/test_scores/shf-7b-kl-{kl_coefficient}.json"
        scores = get_test_scores(url)

        all_data.extend([(kl_coefficient, score, "SuperHF") for score in scores])

        url2 = (
            f"experiments/evaluations/test_scores/rlhf-v3-kl-sweep-kl-{kl_coefficient}"
            "@sxyq16uf.json"
        )

        scores2 = get_test_scores(url2)
        all_data2.extend([(kl_coefficient, score, "RLHF") for score in scores2])

    dataframe = pd.DataFrame(
        all_data, columns=["KL Coefficients", "Test Reward", "Model Type"]
    )
    dataframe2 = pd.DataFrame(
        all_data2, columns=["KL Coefficients", "Test Reward", "Model Type"]
    )  # For the second dataset

    # Combine both dataframes into one
    combined_dataframe = pd.concat([dataframe, dataframe2])

    # Create the plot
    sns.lineplot(
        data=combined_dataframe,
        x="KL Coefficients",
        y="Test Reward",
        hue="Model Type",
        palette=[
            model_type_to_palette_color(m)
            for m in combined_dataframe["Model Type"].unique()
        ],
        errorbar="ci",
    )

    # Set labels and title
    plt.xlabel("KL Coefficients")
    plt.ylabel("Test Reward")
    plt.title("Test Reward at different KL Coefficients")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
