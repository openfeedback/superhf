"""
This file implements the loss function for our reward model.
"""

import torch
from torch import nn


class PreferenceLoss(nn.Module):
    """
    This loss function computes the negative-log likelihood (NLL) of correctly
    predicting the winning input sequence out of a winner-loser pair.

    The probability of predicting the winner is defined as:
        sigmoid(reward(winner) - reward(loser))

    `forward`:
        winner: size (batchsize,) tensor of reward scores of winning responses
        loser: size (batchsize,) tensor of reward scores of losing responses
        Return: mean or summed NLL loss.
    """

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.weight = weight  # could be useful for quality feedback
        self.reduction = reduction

    def forward(self, winner, loser):
        weighted_probabilities = nn.Sigmoid(winner - loser)
        if self.weight is not None:
            weighted_probabilities *= self.weight

        loss = -torch.log(weighted_probabilities)
        if self.reduction == "mean":
            return loss.mean()

        return loss.sum()
