"""Plot """

import seaborn as sns
import matplotlib.pyplot as plt


from chart_utils import (
    initialize_plot,
    save_plot,
    model_type_to_palette_color,
    QUALITATIVE_MODEL_ORDER,
)

QUALITATIVE_MODEL_ORDER = QUALITATIVE_MODEL_ORDER[:-2]

OUTPUT_FILE = "./charts/models/win_rates.png"


def main() -> None:
    """Main function."""
    # Initialize plot
    initialize_plot()
    plt.rcParams["figure.figsize"] = (3, 5)

    # Data
    other_models = [
        "LLaMA",
        "Alpaca",
        "SFT",
        "RLHF",
        "GPT-3.5",
    ]
    superhf_win_rates = [80.56, 46.81, 76.6, 68.75, 29.27]
    other_win_rates = [100.0 - x for x in superhf_win_rates]

    # Create a stacked bar chart
    plot = sns.barplot(
        y=other_models,
        x=superhf_win_rates,
        color=model_type_to_palette_color("superhf"),
        width=0.75,
    )

    palette = [model_type_to_palette_color(model_name) for model_name in other_models]
    sns.barplot(
        y=other_models,
        x=other_win_rates,
        left=superhf_win_rates,
        palette=palette,
        width=0.75,
        # color=model_type_to_palette_color("other"),
    )

    plot.set_xlabel("Win Rate (GPT-4's Preferences)", fontsize=10)
    plot.set_title("SuperHF Win Rates")

    # x-ticks at every 25% and label as percentage
    plot.set_xticks([0, 25, 50, 75, 100])
    plot.set_xticklabels(["0%", "", "50%", "", "100%"])
    plt.xlim(0, 100)

    # Put the y-labels on the right
    plt.tick_params(axis="y", labelright=True, labelleft=False)

    # Tight layout
    plt.tight_layout()

    # Minimal page margins
    # plt.rcParams["savefig.pad_inches"] = 0

    # Save the chart
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
