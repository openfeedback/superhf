"""
This file creates a dataset of human preference data.

Datasets are torch.utils.data.Dataset objects that can be indexed to return
a datum as a Tuple[str, str] in the form (winner_response, loser_response).

The PreferenceDataCollator, passed to the RewardModelTrainer, forms batches
by taking as input a list of tuples (winner_response, loser_response) and
returning a dict {"winner": winner_tokenized, "loser": loser_tokenized}.

RewardModelTrainer.compute_loss(model, inputs) accepts this dict as the
`inputs` argument.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class PreferenceDataCollator:
    """
    Forms a batch by using a list of dataset elements as input. These elements
    are of the same type as the elements of train_dataset and eval_dataset.
    """

    def __init__(self, tokenizer, device="cpu", padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.device = device

    def __call__(self, batch):

        winner_responses = []
        loser_responses = []
        for winner, loser in batch:
            winner_responses.append(winner)
            loser_responses.append(loser)

        winner_tokenized = self.tokenizer(
            winner_responses,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        loser_tokenized = self.tokenizer(
            loser_responses,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        return {"winner": winner_tokenized, "loser": loser_tokenized}


class AnthropicHelpfulHarmless(Dataset):
    """
    Human preference data about helpfulness and harmlessness from Anthropic.
        https://huggingface.co/datasets/Anthropic/hh-rlhf
    """

    def __init__(self, split="train"):
        data = load_dataset("Anthropic/hh-rlhf")[split]
        self.winner_responses = data["chosen"]
        self.loser_responses = data["rejected"]

    def __getitem__(self, idx):
        return self.winner_responses[idx], self.loser_responses[idx]

    def __len__(self):
        return len(self.winner_responses)
