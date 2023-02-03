"""
Testing.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from superhf import skeleton


def main(device):
    """
    This is a simple example of how to use the SuperHF library to finetune a model on a dataset.
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    def hello():
        print("Hello")

    # Load the dataset
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")

    reward_name = "OpenAssistant/reward-model-deberta-v3-base"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(
        reward_name
    ), AutoTokenizer.from_pretrained(reward_name)
    rank_model.to(device)

    # Load the skeleton

    skeleton1 = skeleton
    print(type(skeleton1))
    print("Model loaded")
    print(dataset)
    print(type(tokenizer))


if __name__ == "__main__":
    main("cpu")
