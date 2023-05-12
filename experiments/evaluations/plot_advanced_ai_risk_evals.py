"""
Plot the results of the advanced AI risk evaluations.
"""

import json
import os

import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    """Main function."""
    # Load the data
    with open(
        "./eval_results/advanced_ai_risk/llama-7b.json", "r", encoding="utf-8"
    ) as file:
        data = json.load(file)

    # Order
    order = [
        "Corrigibilty w.r.t a more HHH objective",
        "Corrigibilty w.r.t a neutrally HHH objective",
        "Corrigibilty w.r.t a less HHH objective",
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
    categories = [add_line_breaks(label, label_max_line_length) for label in categories]

    # Create the plot
    plt.figure(figsize=(6, 12))
    sns.set_style("whitegrid")
    sns.scatterplot(x=model_percentage_matches_behavior, y=categories, s=100, alpha=0.8)

    # Add vertical line at 0.5
    plt.axvline(0.5, color="r", linestyle="--", alpha=0.6)

    # Set axis labels and limits
    plt.xlabel("Model Percentage Matches Behavior", fontsize=14)
    plt.ylabel("Test Type", fontsize=14)
    plt.xlim(0, 1)

    # Add labels next to each point
    for i, value in enumerate(model_percentage_matches_behavior):
        plt.text(value + 0.01, i - 0.1, f"{value:.3f}", fontsize=12)

    # Add title
    plt.title("Advanced AI Risk Evaluation Results", fontsize=16)

    # Tight layout
    plt.tight_layout()

    # Add left margin to make room for y-labels
    plt.subplots_adjust(left=0.3)

    # Show the plot
    # plt.show()

    # Save the plot
    if not os.path.exists("./charts/advanced_ai_risk"):
        os.makedirs("./charts/advanced_ai_risk")
    plt.savefig("./charts/advanced_ai_risk/llama-7b.png")


if __name__ == "__main__":
    main()
