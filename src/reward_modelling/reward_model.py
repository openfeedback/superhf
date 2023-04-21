"""
This file implements the training of a reward model.

The PreferenceDataCollator, passed to the RewardModelTrainer, forms batches
by taking as input a list of tuples (winner_response, loser_response) and
returning a dict {"winner": winner_tokenized, "loser": loser_tokenized}.

RewardModelTrainer.compute_loss(model, inputs) accepts this dict as the
`inputs` argument.
"""

import datetime

import logging
import os
import sys


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import datasets
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModel,
    PreTrainedModel,
    AutoConfig,
    HfArgumentParser
)

from reward_modelling.configuartion_reward_model import RewardModelConfig


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
    reward_scores = eval_prediction.predictions
    if reward_scores.ndim==1:
        reward_scores = np.expand_dims(reward_scores, axis=1)
    total = reward_scores.shape[0]
    num_correct = ((reward_scores[:, 0] - reward_scores[:, 1]) > 0).sum()
    accuracy = num_correct / total
    return {
        'accuracy': accuracy, 
        'average_pos_score': reward_scores[:,0].mean().item(),
        'average_neg_score': reward_scores[:,1].mean().item(),
        'average_abs_score_diff': np.mean(np.abs(reward_scores[:,0] - reward_scores[:,1]))}


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
        batch_size = inputs['input_ids'].shape[0]

        scores = model(**inputs)

        chosen_scores = scores[:(batch_size//2)]
        rejected_scores = scores[(batch_size//2):]

        outputs = {'scores': torch.stack((chosen_scores, rejected_scores),dim=1).squeeze(-1)}

        loss_fct = PreferenceLoss()
        loss = loss_fct(chosen_scores, rejected_scores)
        return (loss, outputs) if return_outputs else loss


class RewardModel(PreTrainedModel):
    """
    The reward model to be trained from a pretrained language model.
    """

    config_class = RewardModelConfig

    def __init__(self, config, base_model = None, **kwargs):
        super().__init__(config)
        if base_model == None:
            # print(config.base_model_config)
            self.model = AutoModel.from_config(config.base_model_config, **kwargs)
        else:
            self.model = base_model

        if self.model.config.pad_token_id == None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
    

        self.v_head = nn.Linear(self.model.config.hidden_size, 1, bias=False)

    def forward(self, return_loss=True, **inputs):
        out = self.model(**inputs)[0].mean(dim=1)
        out = self.v_head(out)
        return out
    
    def gradient_checkpointing_enable(self):
        if self.model.supports_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module):
        return super()._init_weights(module)
    
    @classmethod
    def from_pretrained_base_model(cls, base_model_name_or_path, **kwargs):
        config = RewardModelConfig.from_pretrained_base_model(base_model_name_or_path)
        base_model = AutoModel.from_pretrained(base_model_name_or_path)
        return cls(config, base_model,**kwargs)



class PreferenceDataCollator:
    """
    Forms a batch by using a list of dataset elements as input. These elements
    are of the same type as the elements of train_dataset and eval_dataset.
    """

    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):

        chosen_responses = []
        rejected_responses = []
        responses = []
        for chosen_response, rejected_response in batch:
            chosen_responses.append(chosen_response)
            rejected_responses.append(rejected_response)
        responses = chosen_responses+rejected_responses
        
        outputs = self.tokenizer(
            responses,
            return_tensors='pt',
            padding=self.padding,
            max_length=self.max_length,
            truncation=True
        )

        return outputs


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()

    # setup logging
    logger = logging.getLogger(__name__)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()

    from preference_datasets import AnthropicHelpfulHarmless
    from InstructGPTJPairwise_dataset import CompatibleSyntheticInstructGPTJPairwise
    from WebGPTComparisons_dataset import WebGPTComparisons
    # device='cpu'
    # model_name = "distilbert-base-uncased"
    model_name = "EleutherAI/gpt-neo-1.3B"
    # model_name = "EleutherAI/gpt-neo-125M"
    # model_name = "facebook/xglm-4.5B"
    # model_name = "facebook/xglm-2.9B"
    # model_name = "t5-3b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512)
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    model = RewardModel.from_pretrained_base_model(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id

    model_output_dir = f"/nlp/scr/fongsu/reward_model_HH/{model_name}-{datetime.datetime.now()}"

    arguments = TrainingArguments(
        output_dir=model_output_dir,
        logging_steps=10,
        per_device_train_batch_size=4,
        # per_device_eval_batch_size=4,
        num_train_epochs=1,
        do_eval=False,
        # evaluation_strategy="steps",
        # eval_steps=200,
        # save_total_limit=5,
        # save_strategy="steps",
        save_strategy="no",
        # save_steps=20,
        # load_best_model_at_end=True,
        learning_rate=1e-5,
        weight_decay=0.001,
        report_to="wandb",
        # bf16=True,
        # gradient_accumulation_steps=16,
        # fp16=True,
        # optim='adafactor',
        # log_level='debug',
        # label_names='label',
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        push_to_hub=True,
        hub_model_id='gptneo-1.3B-rm-instructgpt'
    )

    log_level = arguments.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    # train_dataset = AnthropicHelpfulHarmless("train", data_dir="harmless-base")
    # eval_dataset = AnthropicHelpfulHarmless("test",data_dir="harmless-base")
    # train_dataset = WebGPTComparisons("train")
    # eval_dataset = WebGPTComparisons("test")
    train_dataset = CompatibleSyntheticInstructGPTJPairwise("train")
    # eval_dataset = CompatibleSyntheticInstructGPTJPairwise("test")

    trainer = RewardModelTrainer(
        model=model,
        args=arguments,
        data_collator=PreferenceDataCollator(tokenizer),
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    result = trainer.train()
    trainer.save_model(model_output_dir)
    # torch.save(model.state_dict(), model_output_dir)
    # print(result)
    print("==============END OF TRAINING===================")
