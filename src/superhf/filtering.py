"""
Functions for filtering completions in SuperHF training.
"""

from abc import ABC, abstractmethod
from typing import Any


class CompletionFilterBase(ABC):
    """
    Base class for completion filters.
    """

    @abstractmethod
    def filter(
        self, scores: list[float], *data: list[list[Any]]
    ) -> tuple[list[float], list[list[Any]]]:
        """
        Filter the completions by the scores.

        Returns both the scores and other things you want to filter (e.g. the text completions).
        """
        raise NotImplementedError


class CompletionFilterTopK(CompletionFilterBase):
    """
    Filter the completions by the top-k scores.
    """

    def __init__(self, top_k: int) -> None:
        """
        Initialize the filter.
        """
        self.top_k = top_k

    def filter(
        self, scores: list[float], *data: list[Any]
    ) -> tuple[list[float], list[list[Any]]]:
        """
        Filter the completions by the scores.

        Returns both the scores and other things you want to filter (e.g. the text completions).
        """
        # Sort the completions by their scores
        sorted_completions_packed = sorted(
            zip(scores, *data),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        # Filter the completions by the top-k scores
        filtered_completions_packed = sorted_completions_packed[: self.top_k]

        # Return the filtered scores and appropriate data
        unzipped = tuple(zip(*filtered_completions_packed))
        scores = list(unzipped[0])
        filtered_data = [list(x) for x in unzipped[1:]]

        return scores, filtered_data
