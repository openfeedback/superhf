"""
Functions for getting formatted prompt datasets for SuperHF training.

Imports datasets from other RLHF experiments. Formats questions into lists:

[
    "Human: What's the capital of France?\n\nAssistant:",
    "Human: What's the best way to get to the airport?\n\nAssistant:",
    "Human: How do I hotwire a car?\n\nAssistant:",
    ...
]
"""

# from datasets.load import load_dataset


# def get_superhf_prompts(dataset_name: str, split: str = "train") -> list[str]:
#     """
#     Get a list of prompts from a dataset.

#     Args:
#         dataset_name: The name of the dataset to load.
#         split: The split of the dataset to load.

#     Returns:
#         A list of prompts.
#     """
#     # dataset = load_dataset(dataset_name, split=split)
#     prompts = []
#     # for example in dataset:
#     #     prompts.append(f"Human: TODO\n\nAssistant:")
#     return prompts
