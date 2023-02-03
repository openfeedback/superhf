"""
This file implements the loss function for our reward model.
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, winner, loser):
        loss = -F.logsigmoid(winner - loser)
        if self.reduction == "mean":
            return loss.mean()

        return loss.sum()

