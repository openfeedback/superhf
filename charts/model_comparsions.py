
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    create_file_dir_if_not_exists,
    flatten_2d_vector,
    load_json,
    set_plot_style,
    model_type_to_palette_color,
)
from superhf.utils import set_seed

OUTPUT_FILE = "./charts/shf_ablations/model_comparisons.png"



def main() -> None:
    """Main function."""

    # Initialize
    set_seed(66)
    set_plot_style()

    # Define the model names
    model_names = [
        "alpaca_7b.json",
        "llama-7b.json",
        "sft-on-preferences-v1.json",
        "test-save-alpaca@model-2048-prompts-batch-size-8.json",
        "shf-7b-default.json",
        "shf-7b-gold-v1.json",
        
    ]

    # Create an empty list for x-axis labels
    x_labels = ["Alpaca", "LLaMA", "SFT", "RLHF", "SHF", "SHF-Gold"]

    # Calculate plot values for data
    all_data = []
    for model_name in model_names:
        # TODO refactor this into chart_utils.py
        file_path = f"./experiments/evaluations/test_scores/{model_name}"
        file_data = load_json(file_path)
        scores = (
            file_data["anthropic-red-team"]
            + file_data["anthropic-helpful-base"]
            + file_data["anthropic-harmless-base"]
            + file_data["openai/webgpt_comparisons"]
        )
        # Unwra
        # Unwrap scores from 2D array
        scores = flatten_2d_vector(scores)

        # Add the data
        for score in scores:
            all_data.append([model_name, score])

        

    dataframe = pd.DataFrame(all_data, columns=["Model", "Score"])

    # Create the plot
    errorbar = "ci"
    plot = sns.barplot(
        data=dataframe,
        x="Model",
        y="Score",
        errorbar=errorbar,
        color=model_type_to_palette_color("superhf"),
    )

    # Set labels and title
    plt.xlabel("Model")
    plt.ylabel("Test score")
    plt.title("Model Comparisons")

    # Set x-ticks
    plt.xticks(range(len(x_labels)), x_labels)

    # Print the plot
    plt.show()

    # Save the plot
    create_file_dir_if_not_exists(OUTPUT_FILE)
    plt.savefig(OUTPUT_FILE)

if __name__ == "__main__":
    main()
