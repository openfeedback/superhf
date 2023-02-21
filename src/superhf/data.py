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

from typing import Iterator, TypeVar

from datasets.load import load_dataset
from torch.utils.data import IterableDataset

T = TypeVar("T")


def get_superhf_prompts(dataset_name: str, split: str = "train") -> list[str]:
    """
    Get a list of prompts from a dataset.

    Args:
        dataset_name: The name of the dataset to load. One of:
            - 'anthropic-red-team'
        split: The split of the dataset to load.

    Returns:
        A list of prompts.
    """
    # Load the appropriate dataset
    if dataset_name == "anthropic-red-team":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="red-team-attempts",
            split=split,
            keep_in_memory=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Convert all the prompts to a list
    prompts: list[str] = [
        dict(row)["transcript"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
        for row in dataset
    ]
    return prompts


class ListDataset(IterableDataset[T]):
    """A Torch dataset that wraps a list of data."""

    def __init__(self, data: list[T]):
        self.data = data

    def __iter__(self) -> Iterator[T]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T:
        return self.data[index]
