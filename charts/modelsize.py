import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

# Define the model sizes and their corresponding file names
file_names = ["shf-pythia-160m.json", "shf-pythia-410m.json", "shf-pythia-1b.json", 
              "shf-pythia-1.4b.json", "shf-pythia-2.8b.json", "shf-pythia-6.9b.json"]
model_sizes = [160, 410, 1000, 1400, 2800, 6900]  # in millions

# Initialize a data frame to hold the scores
data = pd.DataFrame(columns=["model_size", "score"])

# Retrieve data from each file
for model_size, file_name in zip(model_sizes, file_names):
    url = f"https://raw.githubusercontent.com/openfeedback/superhf/evaluations/experiments/evaluations/test_scores/{file_name}"
    r = requests.get(url)
    file_data = r.json()
    scores = file_data["anthropic-red-team"] + file_data["anthropic-helpful-base"] + file_data["openai/webgpt_comparisons"]
    for score_list in scores:
        score = score_list[0]  # Extract the score from the list
        data = data.append({"model_size": model_size, "score": score}, ignore_index=True)

# Convert data types
data["model_size"] = data["model_size"].astype(int)
data["score"] = data["score"].astype(float)

# Initialize arrays to hold the mean scores and the confidence intervals
mean_scores = []
confidence_intervals = []

# For each model size, calculate the mean score and the 95% confidence interval
for model_size in model_sizes:
    scores = data[data["model_size"] == model_size]["score"]
    mean_score = scores.mean()
    confidence_interval = sem(scores) * t.ppf((1 + 0.95) / 2., scores.count() - 1)
    mean_scores.append(mean_score)
    confidence_intervals.append(confidence_interval)

# Create the plot
fig, ax = plt.subplots()
ax.errorbar(model_sizes, mean_scores, yerr=confidence_intervals, fmt='o')

# Set labels and title
ax.set_xlabel('Model size (in millions)')
ax.set_ylabel('Average test accuracy')
ax.set_title('Average test accuracy by model size')

# Display grid
ax.grid(True)

# Show the plot
plt.show()
