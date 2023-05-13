import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the kl coefficients
kl_coefficients = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0]

# Initialize a data frame to hold the scores
data = pd.DataFrame(columns=["kl_coefficient", "score"])

# Retrieve data from each file
for kl in kl_coefficients:
    url = f"https://raw.githubusercontent.com/openfeedback/superhf/evaluations/experiments/evaluations/test_scores/shf-7b-kl-{kl}.json"
    r = requests.get(url)
    file_data = r.json()
    score_lists = file_data["anthropic-red-team"] + file_data["anthropic-helpful-base"] + file_data["openai/webgpt_comparisons"]
    for score_list in score_lists:
        average_score = np.mean(score_list)  # Compute the average score
        data = data.append({"kl_coefficient": kl, "score": average_score}, ignore_index=True)
    print(f"Score list for kl coefficient {kl}: {score_lists}")

# Convert data types
data["kl_coefficient"] = data["kl_coefficient"].astype(float)
data["score"] = data["score"].astype(float)

# Number of bootstrap iterations
n_iterations = 1000

# Initialize a dataframe to hold the bootstrap results
bootstrap_results = pd.DataFrame(columns=["kl_coefficient", "mean", "std"])

# Bootstrapping
for kl in kl_coefficients:
    bootstrap_means = []
    bootstrap_stds = []
    kl_scores = data[data["kl_coefficient"] == kl]["score"]
    for _ in range(n_iterations):
        bootstrap_sample = kl_scores.sample(len(kl_scores), replace=True)
        bootstrap_means.append(bootstrap_sample.mean())
        bootstrap_stds.append(bootstrap_sample.std())
    bootstrap_results = bootstrap_results.append({"kl_coefficient": kl, "mean": np.mean(bootstrap_means), "std": np.mean(bootstrap_stds)}, ignore_index=True)

# Create the plot
fig, ax = plt.subplots()
ax.errorbar(bootstrap_results["kl_coefficient"], bootstrap_results["mean"], yerr=bootstrap_results["std"], fmt='o')

# Set labels and title
ax.set_xlabel('KL Coefficient')
ax.set_ylabel('Average train score')
ax.set_title('Average train score by KL coefficient')

# Display grid
ax.grid(True)

# Show the plot
plt.show()
