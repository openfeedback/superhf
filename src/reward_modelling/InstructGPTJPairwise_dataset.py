import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class CompatibleSyntheticInstructGPTJPairwise(Dataset):
    

    def __init__(self, split="train", data_dir=None):
        data = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise", data_dir=data_dir)[split]
        self.prompts = data["prompt"]
        self.winner_responses = [f"Human: {prompt}\n\nAssistant: {chosen}" for prompt, chosen in zip(data["prompt"], data["chosen"])]
        self.loser_responses = [f"Human: {prompt}\n\nAssistant: {rejected}" for prompt, rejected in zip(data["prompt"], data["rejected"])]

    def __getitem__(self, idx):
        return self.winner_responses[idx], self.loser_responses[idx]

    def __len__(self):
        return len(self.prompts)
