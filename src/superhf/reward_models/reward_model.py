"""
This file implements the loss function for our reward model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification


class PreferenceLoss(nn.Module):
    """
    This loss function computes the negative-log likelihood (NLL) of correctly
    predicting the winning input sequence out of a winner-loser pair.

    The probability of predicting the winner is defined as:
        sigmoid(reward(winner) - reward(loser))

    `forward`:
        winner_scores: size (batchsize,) tensor
        loser_scores: size (batchsize,) tensor
        Return: mean NLL loss.
    """

    def forward(
        self,
        winner_scores: torch.Tensor,
        loser_scores: torch.Tensor,
    ) -> float:
        loss = -F.logsigmoid(winner_scores - loser_scores)
        
        return loss.mean()


class RewardModelTrainer(Trainer):
    """
    This Hugging Face trainer is used to fine-tune a pretrained language model
    on the task of outputting a reward score given a prompt-response dialogue,
    giving higher reward to dialogues that would be preferred by humans.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        model : RewardModel for computing loss
        inputs :
            dict with keys 'prompts', 'winners', 'losers' each mapping to
            a tokenized batch of inputs to pass to model.
        Returns
        -------
        loss: model's loss over inputs, computed by the PreferenceLoss class.
        outputs: size (batch, 2) tensor of reward scores, with winner_scores in
            first column and loser_scores in second.

        """

        winner_scores = model(inputs["prompts"], inputs["winners"])
        loser_scores = model(inputs["prompts"], inputs["losers"])

        loss_fct = PreferenceLoss()
        loss = loss_fct(winner_scores, loser_scores)
        outputs = torch.stack([winner_scores, loser_scores], 1)

        return loss, outputs if return_outputs else loss
