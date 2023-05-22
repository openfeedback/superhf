"""Plot the downstream evaluations."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# import seaborn as sns
from chart_utils import initialize_plot, model_type_to_palette_color, save_plot

OUTPUT_FILE = "./charts/downstream/mmlu_safety_comsense.png"


def main() -> None:
    """Main function."""

    # pylint: disable=too-many-locals

    # Call the set_plot_style function to set the seaborn theme and other styling parameters
    initialize_plot()

    # Set the width of the bars
    bar_width = 0.15

    # Set the data
    evaluations = ["MMLU", "Common Sense", "Safety"]

    alpaca = (
        np.array([0.34394035913428933, 0.6292318320364014, 0.6626076935240235]) * 100
    )
    llama = np.array([0.3119035803354221, 0.6099863078256774, 0.540685696149524]) * 100
    rlhf = np.array([0.3369854258823811, 0.6300493429774198, 0.6443659103785468]) * 100
    superhf = (
        np.array([0.34461623820506443, 0.6287948494238742, 0.6705171927145258]) * 100
    )

    # Subtract Llama values from others
    alpaca -= llama
    rlhf -= llama
    superhf -= llama
    llama -= llama
    # Add a bit to llama to show a blip
    llama += np.array(0.2)

    # Standard errors
    alpaca_7b_stderr = (
        np.array([0.035145507173984965, 0.011640221046600886, 0.006787403716561199])
        * 100
    )
    llama_7b_stderr = (
        np.array([0.03437496210298489, 0.01170265180130513, 0.006958014676005136]) * 100
    )
    rlhf_stderr = (
        np.array([0.03489073357486305, 0.011616057348211164, 0.00673004486012271]) * 100
    )
    superhf_stderr = (
        np.array([0.035134698183086434, 0.011637326683180068, 0.006792445694112547])
        * 100
    )

    # Set position of bar on X axis
    offset1 = np.arange(len(alpaca))
    offset2 = [x + bar_width for x in offset1]
    offset3 = [x + bar_width for x in offset2]
    offset4 = [x + bar_width for x in offset3]

    plt.bar(
        offset1,
        llama,
        color=model_type_to_palette_color("pretrained"),
        width=bar_width,
        edgecolor="grey",
        label="LLaMA",
        yerr=llama_7b_stderr,
        capsize=3,
    )
    plt.bar(
        offset2,
        alpaca,
        color=model_type_to_palette_color("instruct"),
        width=bar_width,
        edgecolor="grey",
        label="Alpaca",
        yerr=alpaca_7b_stderr,
        capsize=3,
    )
    plt.bar(
        offset3,
        rlhf,
        color=model_type_to_palette_color("RLHF"),
        width=bar_width,
        edgecolor="grey",
        label="RLHF",
        yerr=rlhf_stderr,
        capsize=3,
    )
    plt.bar(
        offset4,
        superhf,
        color=model_type_to_palette_color("SuperHF"),
        width=bar_width,
        edgecolor="grey",
        label="SuperHF",
        yerr=superhf_stderr,
        capsize=3,
    )

    # Add xticks on the middle of the group bars
    plt.xlabel("Evaluations", fontweight="bold")
    plt.ylabel("Accuracy (Relative to LLaMA)", fontweight="bold")
    plt.xticks([r + bar_width * 1.5 for r in range(len(alpaca))], evaluations)

    # Adjust y-axis limits to show negative bars
    # plt.ylim(min(np.min(Alpaca), np.min(RLHF), np.min(SuperHF)) - 10,
    #   max(np.max(Alpaca), np.max(RLHF), np.max(SuperHF)) + 10)
    plt.ylim(0, max(np.max(alpaca), np.max(rlhf), np.max(superhf)) + 1)

    # Create a function to format y-axis as percentages
    fmt = "%.0f%%"
    yticks = mtick.FormatStrFormatter(fmt)
    plt.gca().yaxis.set_major_formatter(yticks)
    plt.title("Downstream Evaluations")
    plt.legend()

    # plt.savefig("plot.png", dpi=300)

    # plt.show()

    save_plot(OUTPUT_FILE)


if __name__ == "__main__":
    main()
