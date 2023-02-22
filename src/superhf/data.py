"""
Functions for getting formatted prompt datasets for SuperHF training.

Imports datasets from other RLHF experiments. Formats questions into lists:

[
    "\n\nHuman: What's the capital of France?\n\nAssistant:",
    "\n\nHuman: What's the best way to get to the airport?\n\nAssistant:",
    "\n\nHuman: How do I hotwire a car?\n\nAssistant:",
    ...
]

Which is the same format as from the Anthropic HH-RLHF dataset.
"""


from datasets.load import load_dataset


def get_superhf_prompts(dataset_name: str) -> list[str]:
    """
    Get a list of prompts from a dataset.

    Args:
        dataset_name: The name of the dataset to load. One of:
            - 'anthropic-red-team'
        split: The split of the dataset to load.

    Returns:
        A list of prompts.
    """
    prompts: list[str] = []

    # Load the appropriate dataset.
    if dataset_name == "anthropic-red-team":
        dataset = load_dataset(
            "Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train"
        )
        prompts = [
            dict(row)["transcript"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
            for row in dataset
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return prompts
