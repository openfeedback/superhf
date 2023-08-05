"""Plot """

import seaborn as sns
import matplotlib.pyplot as plt


from chart_utils import (
    initialize_plot,
    save_plot,
    model_type_to_palette_color,
    model_type_to_hatch,
    spaces_to_newlines,
)

# QUALITATIVE_MODEL_ORDER = QUALITATIVE_MODEL_ORDER[:-2]

SHF_NOT_RLHF = False

OUTPUT_FILE = f"./charts/models/win_rates_{'shf' if SHF_NOT_RLHF else 'rlhf'}.png"


def main() -> None:
    """Main function."""
    # Initialize plot
    initialize_plot()
    plt.rcParams["figure.figsize"] = (3, 5)

    # Data from win_rates_table.py
    other_models_to_my_win_rates = (
        {
            "LLaMA": 0.7,
            "FeedME": 0.2857142857142857,
            "Instruct": 0.65,
            "RLHF (LLaMA)": 0.7272727272727273,
            "RLHF (Instruct)": 0.45,
            "SuperHF (LLaMA)": 0.5,
            "Alpaca": 0.6363636363636364,
        }
        if SHF_NOT_RLHF
        else {
            "LLaMA": 0.5652173913043478,
            "FeedME": 0.2857142857142857,
            "Instruct": 0.47619047619047616,
            "RLHF (LLaMA)": 0.48,
            "SuperHF (LLaMA)": 0.631578947368421,
            "SuperHF (Instruct)": 0.55,
            "Alpaca": 0.23809523809523808,
        }
    )
    other_models = spaces_to_newlines(list(other_models_to_my_win_rates.keys()))
    my_win_rates = [rate * 100.0 for rate in other_models_to_my_win_rates.values()]
    other_win_rates = [100.0 - x for x in my_win_rates]

    # Hatch color
    hatch_color = "#ffffff66"

    # Create a stacked bar chart
    plot = sns.barplot(
        y=other_models,
        x=my_win_rates,
        color=model_type_to_palette_color("superhf" if SHF_NOT_RLHF else "rlhf"),
        width=0.75,
        hatch="/",
        edgecolor=hatch_color,
    )

    palette = [model_type_to_palette_color(model_name) for model_name in other_models]
    hatch = [model_type_to_hatch(model_name, 1) for model_name in other_models]
    # Add new lines to labels
    sns.barplot(
        y=other_models,
        x=other_win_rates,
        left=my_win_rates,
        palette=palette,
        width=0.75,
        hatch=hatch,
        edgecolor=hatch_color,
    )

    # Add data labels to the bars
    for i, value in enumerate(my_win_rates):
        horizontal_align = "left" if value < 50 else "right"
        horizontal_adjust = 2 if value < 50 else -2
        plot.text(
            value + horizontal_adjust,
            i,
            f"{value:.1f}%",
            color="white",
            horizontalalignment=horizontal_align,
            verticalalignment="center",
            # fontweight="bold",
            fontsize=10,
        )

    plot.set_xlabel("Win Rate (GPT-4's Preferences)", fontsize=10)
    plot.set_title(
        f"{'SuperHF' if SHF_NOT_RLHF else 'RLHF'} (Instruct)\nGPT-4 Win Rates"
    )

    # x-ticks at every 25% and label as percentage
    plot.set_xticks([0, 25, 50, 75, 100])
    plot.set_xticklabels(["0%", "", "50%", "", "100%"])
    plt.xlim(0, 100)

    # Put the y-labels on the right
    plt.tick_params(axis="y", labelright=True, labelleft=False)

    # Tight layout
    plt.tight_layout()

    # Save the chart
    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
