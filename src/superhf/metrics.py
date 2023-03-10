"""
Functions for reporting metrics in SuperHF training.
"""

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import wandb

from superhf.filtering import CompletionFilterTopK
from superhf.utils import separate_prompt_from_completion


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
    - Score average and histogram
    - Filtered score average and histogram
    - Scheduler learning rate
    - Average loss
    - Scheduler learning rate
    - Completions length average and histogram
    - Filtered completions length average and histogram
    - Completions table
    - Filtered completions table
    - Histogram of filtered score if we filtered different top-K numbers
    """
    percent_complete = (metrics.superbatch_index + 1) / metrics.superbatch_count * 100
    completion_lengths = [
        len(separate_prompt_from_completion(completion)[1])
        for completion in metrics.completions
    ]
    filtered_completion_length = [
        len(separate_prompt_from_completion(completion)[1])
        for completion in metrics.filtered_completions
    ]
    average_score = np.mean(metrics.scores)
    average_filtered_score = np.mean(metrics.filtered_scores)

    # Create plot data of average score if we filtered different top-K numbers
    max_top_k_to_explore = 32
    scores_per_top_k: list[list[Any]] = []
    for top_k in range(1, max_top_k_to_explore + 1):
        top_k_filter = CompletionFilterTopK(top_k)
        _, scores = top_k_filter.filter(metrics.completions, metrics.scores)
        mean, variance = np.mean(scores), np.var(scores)
        scores_per_top_k.append([top_k, mean, variance])

    wandb.log(
        {
            "superbatch_index": metrics.superbatch_index,
            "percent_complete": percent_complete,
            "average_score": average_score,
            "score_histogram": wandb.Histogram(metrics.scores),
            "average_filtered_score": average_filtered_score,
            "filtered_score_histogram": wandb.Histogram(metrics.filtered_scores),
            "average_loss": metrics.average_loss,
            "scheduler_lr": metrics.scheduler_lr,
            "average_completion_length": np.mean(completion_lengths),
            "completion_length_histogram": wandb.Histogram(completion_lengths),
            "average_filtered_completion_length": np.mean(filtered_completion_length),
            "filtered_completion_length_histogram": wandb.Histogram(
                filtered_completion_length
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
            "scores_per_top_k": wandb.plot.line(
                wandb.Table(
                    columns=["Top-K", "Score", "Variance"], data=scores_per_top_k
                ),
                "Top-K",
                "Score",
                stroke="Variance",
                title="Scores Per Top-K (Latest)",
            ),
        }
    )


### Delay ###


def delay_metrics(_: SuperHFMetrics) -> None:
    """
    Delay to allow time for logging to reach a server.
    """
    time.sleep(3)
