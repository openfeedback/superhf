import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main() -> None:
    """Main function."""

    # Initialize
    set_seed(66)
    set_plot_style()

    # Define the model names
    kl_coefficients = [0, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # Calculate plot values for data
    all_data = []
    all_data2 = []  # For the second dataset
    for kl in kl_coefficients:
        url = f"https://raw.githubusercontent.com/openfeedback/superhf/evaluations/experiments/evaluations/test_scores/shf-7b-kl-{kl}.json"
        r = requests.get(url)
        file_data = r.json()
        scores = (
            file_data["anthropic-red-team"]
            + file_data["anthropic-helpful-base"]
            + file_data["anthropic-harmless-base"]
            + file_data["openai/webgpt_comparisons"]
        )
        
        scores = flatten_2d_vector(scores)
        all_data.extend([(kl, score, 'SuperHF') for score in scores])
       
        url2 = f"https://raw.githubusercontent.com/openfeedback/superhf/evaluations/experiments/evaluations/test_scores/rlhf-v3-kl-sweep-kl-{kl}@sxyq16uf.json"
        r2 = requests.get(url2)
        file_data2 = r2.json()
        scores2 = (
            file_data2["anthropic-red-team"]
            + file_data2["anthropic-helpful-base"]
            + file_data2["anthropic-harmless-base"]
            + file_data2["openai/webgpt_comparisons"]
        )
        scores2 = flatten_2d_vector(scores2)
        all_data2.extend([(kl, score, 'RLHF') for score in scores2])



    dataframe = pd.DataFrame(all_data, columns=["KL Coefficients", "Test Reward", "Model Type"])
    dataframe2 = pd.DataFrame(all_data2, columns=["KL Coefficients", "Test Reward", "Model Type"])  # For the second dataset

    # Combine both dataframes into one
    combined_dataframe = pd.concat([dataframe, dataframe2])

    # Create the plot
    plot = sns.lineplot(
        data=combined_dataframe,
        x="KL Coefficients",
        y="Test Reward",
        hue="Model Type",
        palette=[model_type_to_palette_color(m) for m in combined_dataframe['Model Type'].unique()],
        ci=95
    )

    # Set labels and title
    plt.xlabel("KL Coefficients")
    plt.ylabel("Test Reward")
    plt.title("Test Reward at different KL Coefficients")
    
   




# Save the plot
create_file_dir_if_not_exists(OUTPUT_FILE)
plt.savefig(OUTPUT_FILE)

if __name__ == "__main__":
    main()
