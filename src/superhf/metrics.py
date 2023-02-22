"""
Functions for reporting metrics in SuperHF training.
"""

from dataclasses import dataclass

# import wandb
import numpy as np


@dataclass
class SuperHFMetrics:
    """
    Metrics for SuperHF training.
    """

    superbatch_index: int
    superbatch_count: int
    completions: list[str]
    filtered_completions: list[str]
    scores: list[float]
    filtered_scores: list[float]
    average_loss: float


def report_metrics_print(metrics: SuperHFMetrics) -> None:
    """
    Print basic metrics to STD out.
    """
    percent_complete = metrics.superbatch_index / metrics.superbatch_count * 100
    average_completion_length = np.mean([len(c) for c in metrics.completions])
    average_filtered_completion_length = np.mean(
        [len(c) for c in metrics.filtered_completions]
    )
    average_score = np.mean(metrics.scores)
    average_filtered_score = np.mean(metrics.filtered_scores)
    print(
        f"\nSuperbatch {metrics.superbatch_index}/{metrics.superbatch_count} "
        f"({percent_complete:.3f}%): {len(metrics.completions)} completions, "
        f"{len(metrics.filtered_completions)} filtered completions\n"
        f"average completion length {average_completion_length:.3f}, "
        f"average filtered completion length {average_filtered_completion_length:.3f}\n"
        f"average score {average_score:.3f}, average filtered score {average_filtered_score:.3f}, "
        f"average loss {metrics.average_loss:.3f}."
    )


def report_metrics_wandb() -> None:
    """
    Base class for completion filters.
    """
    raise NotImplementedError
