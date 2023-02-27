"""
Functions for reporting metrics in SuperHF training.
"""

from dataclasses import dataclass

import numpy as np
import wandb


@dataclass
class SuperHFMetrics:
    """
    Metrics for SuperHF training.
    """

    # pylint: disable=too-many-instance-attributes

    superbatch_index: int
    superbatch_count: int
    completions: list[str]
    filtered_completions: list[str]
    scores: list[float]
    filtered_scores: list[float]
    average_loss: float
    scheduler_lr: float


### Printing ###


def report_metrics_print(metrics: SuperHFMetrics) -> None:
    """
    Print basic metrics to STD out.
    """
    percent_complete = (metrics.superbatch_index + 1) / metrics.superbatch_count * 100
    average_completion_length = np.mean([len(c) for c in metrics.completions])
    average_filtered_completion_length = np.mean(
        [len(c) for c in metrics.filtered_completions]
    )
    average_score = np.mean(metrics.scores)
    average_filtered_score = np.mean(metrics.filtered_scores)
    print(
        "\nSuperbatch"
        f" {metrics.superbatch_index}/{metrics.superbatch_count} ({percent_complete:.3f}%):"
        f" {len(metrics.completions)} completions,"
        f" {len(metrics.filtered_completions)} filtered completions\naverage completion"
        f" length {average_completion_length:.3f}, average filtered completion length"
        f" {average_filtered_completion_length:.3f}\naverage score {average_score:.3f},"
        f" average filtered score {average_filtered_score:.3f}, average loss"
        f" {metrics.average_loss:.3f}."
    )


### Weights and Biases ###


def initialize_metrics_wandb() -> None:
    """
    Defines metrics for a Weights and Biases run.
    """
    wandb.define_metric("average_loss", summary="min")
    wandb.define_metric("average_score", summary="max")
    wandb.define_metric("average_completion_length", summary="last")


def report_metrics_wandb(metrics: SuperHFMetrics) -> None:
    """
    Report metrics to Weights and Biases.

    Logs the following metrics:
    - Superbatch index and percentage complete
    - Number of completions, number of filtered completions
    - Average completion length, average filtered completion length
    - Average score, average filtered score
    - Average loss
    - Average score histogram
    - Average filtered score histogram
    - Table of completions and scores
    - Table of filtered completions and scores
    - Learning rate
    """
    percent_complete = (metrics.superbatch_index + 1) / metrics.superbatch_count * 100
    average_completion_length = np.mean([len(c) for c in metrics.completions])
    max_completion_length = np.max([len(c) for c in metrics.completions])
    average_filtered_completion_length = np.mean(
        [len(c) for c in metrics.filtered_completions]
    )
    average_score = np.mean(metrics.scores)
    average_filtered_score = np.mean(metrics.filtered_scores)
    wandb.log(
        {
            "superbatch_index": metrics.superbatch_index,
            "superbatch_percent_complete": percent_complete,
            "completion_count": len(metrics.completions),
            "filtered_completion_count": len(metrics.filtered_completions),
            "average_completion_length": average_completion_length,
            "max_completion_length": max_completion_length,
            "average_filtered_completion_length": average_filtered_completion_length,
            "average_score": average_score,
            "average_filtered_score": average_filtered_score,
            "average_loss": metrics.average_loss,
            "average_score_histogram": wandb.Histogram(metrics.scores),
            "average_filtered_score_histogram": wandb.Histogram(
                metrics.filtered_scores
            ),
            "completions": wandb.Table(
                columns=["superbatch", "completion", "score"],
                data=[
                    [metrics.superbatch_index, completion, score]
                    for completion, score in zip(metrics.completions, metrics.scores)
                ],
            ),
            "filtered_completions": wandb.Table(
                columns=["superbatch", "completion", "score"],
                data=[
                    [metrics.superbatch_index, completion, score]
                    for completion, score in zip(
                        metrics.filtered_completions, metrics.filtered_scores
                    )
                ],
            ),
            "scheduler_lr": metrics.scheduler_lr,
        }
    )
