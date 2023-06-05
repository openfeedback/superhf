"""Graph the reward as a function of KL coefficient."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from chart_utils import (
    initialize_plot,
    model_type_to_palette_color,
    get_test_scores,
    save_plot,
    ALPACA_TEST_REWARD,
    LLAMA_TEST_REWARD,
)


OUTPUT_FILE = "./charts/models/dollar_costs.png"

INPUT_FILE_MAP = {
    "FTP": [
        "./experiments/evaluations/test_scores/llama-ftp-49516.json",
        "./experiments/evaluations/test_scores/llama-ftp-24758.json",
        "./experiments/evaluations/test_scores/llama-ftp-12379.json",
        "./experiments/evaluations/test_scores/llama-ftp-6338.json",
        "./experiments/evaluations/test_scores/llama-ftp-3095.json",
        "./experiments/evaluations/test_scores/llama-ftp-1547.json",
        "./experiments/evaluations/test_scores/llama-ftp-774.json",
    ],
    "Instruct": [
        "./experiments/evaluations/test_scores/llama-instruct-12379.json",
        "./experiments/evaluations/test_scores/llama-instruct-6190.json",
        "./experiments/evaluations/test_scores/llama-instruct-3095.json",
        "./experiments/evaluations/test_scores/llama-instruct-1585.json",
        "./experiments/evaluations/test_scores/llama-instruct-774.json",
        "./experiments/evaluations/test_scores/llama-instruct-387.json",
        "./experiments/evaluations/test_scores/llama-instruct-194.json",
    ],
    "SuperHF (Instruct)": [
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-12379.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-6190.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-3095.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-1585.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-774.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-387.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-194.json",
    ],
    "SuperHF (LLaMA)": ["./experiments/evaluations/test_scores/shf-llama-7b.json"],
}


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    # Initialize
    initialize_plot()

    # Define the x-axis
    dollar_costs = [322.5, 644.58, 1289.58, 2640.83, 5157.92, 10315.83, 20631.67]
    pref_number_to_dollar_cost_index = {
        774: 0,
        1547: 1,
        3095: 2,
        6338: 3,
        12379: 4,
        24758: 5,
        49516: 6,
    }
    instruct_number_to_dollar_cost_index = {
        194: 0,
        387: 1,
        774: 2,
        1585: 3,
        3095: 4,
        6190: 5,
        12379: 6,
    }

    # Calculate plot values for data
    all_data = []
    for model_type, score_files in INPUT_FILE_MAP.items():
        for score_file in score_files:
            scores = get_test_scores(score_file)
            try:
                train_index = int(score_file.rsplit("-", maxsplit=1)[-1].split(".")[0])
                if "instruct" in score_file:
                    dollar_cost = dollar_costs[
                        instruct_number_to_dollar_cost_index[train_index]
                    ]
                else:
                    dollar_cost = dollar_costs[
                        pref_number_to_dollar_cost_index[train_index]
                    ]
            except ValueError:
                dollar_cost = dollar_costs[-1]  # SuperHF

            all_data.extend([(dollar_cost, score, model_type) for score in scores])

            # Print average test score
            print(
                f"SuperHF average test score for model_type={model_type}:"
                f" {sum(scores)/len(scores):.3f}"
            )

    dataframe = pd.DataFrame(all_data, columns=["Cost", "Test Reward", "Model Type"])

    # Create the plot
    sns.lineplot(
        data=dataframe,
        x="Cost",
        y="Test Reward",
        hue="Model Type",
        palette=[
            model_type_to_palette_color(m) for m in dataframe["Model Type"].unique()
        ],
        errorbar="ci",
    )

    # Add llama and alpaca Test (horizontal line at -0.01)
    plt.axhline(
        y=LLAMA_TEST_REWARD,
        color=model_type_to_palette_color("pretrained"),
        linestyle="--",
        label="LLaMA",
        marker="",
    )
    plt.axhline(
        y=ALPACA_TEST_REWARD,
        color=model_type_to_palette_color("instruct"),
        linestyle="--",
        label="Alpaca",
        marker="",
    )
    plt.legend()

    # Logx
    plt.xscale("log")

    # X-ticks for dollar costs (add $ sign)
    plt.xticks(
        dollar_costs,
        [f"${cost:.0f}" for cost in dollar_costs],
    )

    # Set labels and title
    plt.xlabel("Data Price (At $25/hour)")
    plt.ylabel("Test Reward")
    plt.title("Price Efficiency of Training Methods")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
