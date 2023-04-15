import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import re

class SummarizeFromFeedbackComparisons(Dataset):
    
    def __init__(self, data_dir=None):
        # Load the "comparisons" configuration of the dataset
        data = load_dataset("openai/summarize_from_feedback", "comparisons", data_dir=data_dir)

        # Combine the training and validation splits
        self.combined_data = concatenate_datasets([data["train"], data["validation"]])

        # Extract the necessary information
        self.info = self.combined_data["info"]
        self.summaries = self.combined_data["summaries"]
        self.choice = self.combined_data["choice"]

    def __getitem__(self, idx):
        info = self.info[idx]
        info_values = ' '.join([str(value) for value in info.values() if not str(value).startswith("t3_")])
        info_values = re.sub(r'\[\d+\]', '', info_values)
        summary1 = self.summaries[idx][0]["text"]
        summary2 = self.summaries[idx][1]["text"]
        choice = self.choice[idx]

        winner_response = f"Info: {info_values} | Summary: {summary1}" if choice == 0 else f"Info: {info_values} | Summary: {summary2}"
        loser_response = f"Info: {info_values} | Summary: {summary2}" if choice == 0 else f"Info: {info_values} | Summary: {summary1}"

        return winner_response, loser_response

    def __len__(self):
        return len(self.info)
