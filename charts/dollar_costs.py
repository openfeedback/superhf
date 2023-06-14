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
        "./experiments/evaluations/test_scores/llama-ftp-6190.json",
        "./experiments/evaluations/test_scores/llama-ftp-3095.json",
        "./experiments/evaluations/test_scores/llama-ftp-1547.json",
        "./experiments/evaluations/test_scores/llama-ftp-774.json",
    ],
    "Instruct": [
        "./experiments/evaluations/test_scores/llama-instruct-12379.json",
        "./experiments/evaluations/test_scores/llama-instruct-6190.json",
        "./experiments/evaluations/test_scores/llama-instruct-3095.json",
        "./experiments/evaluations/test_scores/llama-instruct-1547.json",
        "./experiments/evaluations/test_scores/llama-instruct-774.json",
        "./experiments/evaluations/test_scores/llama-instruct-387.json",
        "./experiments/evaluations/test_scores/llama-instruct-194.json",
    ],
    "SHF (Instr OG RM)": [
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-12379.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-6190.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-3095.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-1547.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-774.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-387.json",
        "./experiments/evaluations/test_scores/shf-v4-llama-instruct-194.json",
    ],
    # "SHF (LLaMA OG RM)": [
    #     "./experiments/evaluations/test_scores/shf-v4-llama-10000-kl-0.35.json",
    # ],
    "SHF (LLaMA New RMs)": [
        # "./experiments/evaluations/test_scores/shf-v5-llama-base-rm-2.0.json",
        "./experiments/evaluations/test_scores/shf-v5-llama-base-rm-1.0.json",
        "./experiments/evaluations/test_scores/shf-v5-llama-base-rm-0.5.json",
        "./experiments/evaluations/test_scores/shf-v5-llama-base-rm-0.25.json",
        "./experiments/evaluations/test_scores/shf-v5-llama-base-rm-0.125.json",
        "./experiments/evaluations/test_scores/shf-v5-llama-base-rm-0.0625.json",
        "./experiments/evaluations/test_scores/shf-v5-llama-base-rm-0.03125.json",
        "./experiments/evaluations/test_scores/shf-v5-llama-base-rm-0.015625.json",
        # "./experiments/evaluations/test_scores/shf-v4-llama-base-rm-774.json",
        # "./experiments/evaluations/test_scores/shf-v4-llama-base-rm-1547.json",
        # "./experiments/evaluations/test_scores/shf-v4-llama-base-rm-3095.json",
        # "./experiments/evaluations/test_scores/shf-v4-llama-base-rm-6190.json",
        # "./experiments/evaluations/test_scores/shf-v4-llama-base-rm-12379.json",
        # "./experiments/evaluations/test_scores/shf-v4-llama-base-rm-24758.json",
        # "./experiments/evaluations/test_scores/shf-v4-llama-base-rm-49516.json",
        # "./experiments/evaluations/test_scores/shf-llama-7b.json",
    ],
}


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    # Initialize
    initialize_plot()

    # Define the x-axis
    dollar_costs = [
        322.5,
        644.58,
        1289.58,
        2640.83,
        5157.92,
        10315.83,
        20631.67,
        # 41263.33,
    ]
    pref_number_to_dollar_cost_index = {
        774: 0,
        1547: 1,
        3095: 2,
        6190: 3,
        12379: 4,
        24758: 5,
        49516: 6,
        # 99032: 7,
    }
    instruct_number_to_dollar_cost_index = {
        194: 0,
        387: 1,
        774: 2,
        1547: 3,
        3095: 4,
        6190: 5,
        12379: 6,
        # 24758: 7,
    }
    rm_fraction_to_dollar_cost_index = {
        2.0: 7,
        1.0: 6,
        0.5: 5,
        0.25: 4,
        0.125: 3,
        0.0625: 2,
        0.03125: 1,
        0.015625: 0,
    }

    # Calculate plot values for data
    all_data = []
    for model_type, score_files in INPUT_FILE_MAP.items():
        for score_file in score_files:
            scores = get_test_scores(score_file)
            try:
                train_index_str = score_file.rsplit("-", maxsplit=1)[-1].split(".json")[
                    0
                ]
                if "instruct" in score_file:
                    train_index = int(train_index_str)
                    dollar_cost = dollar_costs[
                        instruct_number_to_dollar_cost_index[train_index]
                    ]
                elif "rm" in score_file and "0." in score_file:
                    train_frac = float(train_index_str)
                    dollar_cost = dollar_costs[
                        rm_fraction_to_dollar_cost_index[train_frac]
                    ]
                else:
                    train_index = int(train_index_str)
                    dollar_cost = dollar_costs[
                        pref_number_to_dollar_cost_index[train_index]
                    ]
            except ValueError:
                dollar_cost = dollar_costs[-2]  # SuperHF

            all_data.extend([(dollar_cost, score, model_type) for score in scores])

            # Print average test score
            print(
                f"SuperHF average test score for model_type={model_type}:"
                f" {sum(scores)/len(scores):.3f}"
            )

    dataframe = pd.DataFrame(all_data, columns=["Cost", "Test Reward", "Model Type"])

    # Create the plot
    palette = [model_type_to_palette_color(m) for m in dataframe["Model Type"].unique()]
    palette[-1] = model_type_to_palette_color("gpt-4")
    palette[3] = "#76f"
    sns.lineplot(
        data=dataframe,
        x="Cost",
        y="Test Reward",
        hue="Model Type",
        palette=palette,
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
    # Smaller more transparent legend
    plt.legend(ncol=3, fontsize=12, loc="upper left", framealpha=0.5)

    # Logx
    plt.xscale("log")

    # X-ticks for dollar costs, preference counts, and instruction counts
    all_num_prefs = sorted(pref_number_to_dollar_cost_index.keys())
    all_num_instructs = sorted(instruct_number_to_dollar_cost_index.keys())
    plt.xticks(
        [200] + dollar_costs,
        ["Price\nPrefs\nInstrs"]
        + [
            f"${cost:,.0f}\n{pref:,d}\n{instruct:,d}"
            for cost, pref, instruct in zip(
                dollar_costs, all_num_prefs, all_num_instructs
            )
        ],
    )

    # Set labels and title
    plt.xlabel("Data Cost (At $25/hour, 1 min/preference, 4 min/instruction)")
    plt.ylabel("Test Reward")
    plt.title("Price Efficiency of Training Methods")

    # Save the plot
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
