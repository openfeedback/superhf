"""
This file implements our reward model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification


# NOTE ON DATSETS
# I'm assuming that we will build a dataset formed as a:
#     torch.utils.data.Dataset({features: ['winner', 'loser']})
# So when drawing a batch `inputs`, we can do inputs['winner'] to get
# the tokenized winning input sequences, ready for model(**inputs['winner']),
# and likewise for 'loser'.
#
# OpenAssistant achieved something equivalent using a few different datasets
# which they must have massaged into this form.


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

    def forward(self, winner_scores: torch.Tensor, loser_scores):
        loss = -F.logsigmoid(winner_scores - loser_scores)
        return loss.mean()


def compute_metrics(eval_prediction):
    """
    Computes decimal accuracy of predictions during model evaluation.
    """

    reward_scores = eval_prediction.prediction
    correct_predictions = (reward_scores[:, 0] - reward_scores[:, 1]) > 0
    return correct_predictions.sum() / correct_predictions.size[0]


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
            tokenized input batch with features ['winner', 'loser'].
        Returns
        -------
        loss: model's loss over inputs, computed by the PreferenceLoss class.
        reward_scores: size (batch, 2) tensor of reward scores,
            with winner_scores in first column and loser_scores in second.

        """

        winner_scores = model(**inputs["winner"])
        loser_scores = model(**inputs["loser"])

        loss_fct = PreferenceLoss()
        loss = loss_fct(winner_scores, loser_scores)
        reward_scores = torch.stack([winner_scores, loser_scores], 1)
        return (loss, reward_scores) if return_outputs else loss

    def prediction_step(model, inputs, prediction_loss_only, ignore_keys=None):
        """
        model : RewardModel for computing loss
        inputs :
            tokenized input batch with features ['winner', 'loser'].
        Returns
        -------
        loss: model's loss over inputs, computed by the PreferenceLoss class.
        reward_scores: size (batch, 2) tensor of reward scores,
            with winner_scores in first column and loser_scores in second.
        """
        loss, reward_scores = self.compute_loss(model, inputs, True)
        return (loss, None) if prediction_loss_only else (loss, reward_scores)


class RewardModel(nn.Module):
    """
    The reward model to be trained from a pretrained language model.
    """

    def __init__(self, model_name, frozen_prefixes=()):
        self.model = AutoModelForSequenceClassification(model_name, num_labels=1)
        for name, param in self.model.base_model.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                param.requires_grad = False
