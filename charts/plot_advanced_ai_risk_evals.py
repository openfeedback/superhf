"""
Plot the results of the advanced AI risk evaluations.
"""

import json

import matplotlib.pyplot as plt
import seaborn as sns

from chart_utils import initialize_plot, save_plot

model_names_and_paths = [
    ("LLaMA", "./eval_results/advanced_ai_risk/llama-7b.json"),
    ("Alpaca", "./eval_results/advanced_ai_risk/alpaca_7b.json"),
    # ("SFT", "./eval_results/advanced_ai_risk/sft-on-preferences-v1.json"),
    (
        "RLHF",
        "./eval_results/advanced_ai_risk/rlhf-v3-lr-5.0e-6-batch-16@gold-run.json",
    ),
    ("SuperHF", "./eval_results/advanced_ai_risk/shf-7b-default.json"),
]

# Order
order = [
    "Corrigibility w.r.t a more HHH objective",
    "Corrigibility w.r.t a neutrally HHH objective",
    "Corrigibility w.r.t a less HHH objective",
    "Coordinate with other AIs",
    "Coordinate with newer/older versions",
    "Coordinate with exact copies",
    "Myopia",
    "One-box tendency",
    "Awareness of lack of internet access",
    "Awareness of being a text-only model",
    "Awareness of ability to solve complex text tasks",
    "Awareness of being an AI",
    "Awareness of architecture",
    "Desire for survival",
    "Desire for power",
    "Desire for wealth",
]


def main() -> None:
    """Main function."""
    initialize_plot()
    plt.figure(figsize=(6, 12))

    for model_name, data_path in model_names_and_paths:
        # Load the data
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Extract the data points
        results = data["results"]
        categories = list(results.keys())
        model_percentage_matches_behavior = [
            result["model_percentage_matches_behavior"] for result in results.values()
        ]

        # Sort the results according to the order
        categories, model_percentage_matches_behavior = zip(
            *sorted(
                zip(categories, model_percentage_matches_behavior),
                key=lambda x: order.index(x[0]),
            )
        )

        # Add line breaks to long y-labels
        def add_line_breaks(label: str, max_line_length: int) -> str:
            """Adds line breaks to a label so that it fits on the plot."""
            words = label.split()
            lines = []
            current_line = ""

            for word in words:
                if len(current_line) + len(word) + 1 > max_line_length:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line += " " + word

            lines.append(current_line)
            return "\n".join(lines).strip()

        label_max_line_length = 20
        categories = [
            add_line_breaks(label, label_max_line_length) for label in categories
        ]

        # Create the plot
        sns.scatterplot(
            x=model_percentage_matches_behavior,
            y=categories,
            label=model_name,
            s=100,
            alpha=0.8,
        )

        # Add labels next to each point
        # for i, value in enumerate(model_percentage_matches_behavior):
        #     plt.text(value + 0.01, i - 0.1, f"{value:.3f}", fontsize=12)

    # Add vertical line at 0.5
    plt.axvline(0.5, color="r", linestyle="--", marker="", alpha=0.6)

    # Set axis labels and limits
    plt.xlabel("Model Percentage Matches Behavior", fontsize=14)
    plt.ylabel("Test Type", fontsize=14)
    plt.xlim(0, 1)

    # Add title
    plt.title("Advanced AI Risk Evaluation Results", fontsize=16)

    # Tight layout
    plt.tight_layout()

    # Add left margin to make room for y-labels
    plt.subplots_adjust(left=0.3)

    # Legend
    plt.legend(loc="upper left", fontsize=12)

    # Show the plot
    # plt.show()

    # Save the plot
    save_plot("./charts/models/advanced_ai_risk.png")


if __name__ == "__main__":
    main()
