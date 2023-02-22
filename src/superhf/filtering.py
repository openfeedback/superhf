"""
Functions for filtering completions in SuperHF training.
"""

from abc import ABC, abstractmethod


class CompletionFilterBase(ABC):
    """
    Base class for completion filters.
    """

    @abstractmethod
    def filter(
        self, completions: list[str], scores: list[float]
    ) -> tuple[list[str], list[float]]:
        """
        Filter the completions by the scores.

        Returns both the completions and the scores.
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
        self, completions: list[str], scores: list[float]
    ) -> tuple[list[str], list[float]]:
        """
        Filter the completions by the top-k scores.

        Returns both the completions and the scores.
        """
        # Sort the completions by their scores
        sorted_completions = sorted(
            zip(completions, scores), key=lambda x: x[1], reverse=True
        )

        # Filter the completions by the top-k scores
        filtered_completions = sorted_completions[: self.top_k]

        # Return the filtered completions and their scores
        return tuple(zip(*filtered_completions))
