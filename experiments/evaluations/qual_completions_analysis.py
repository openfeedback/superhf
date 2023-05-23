"""
This is a file for evaluating trends in the LM completions.
"""

import json
import os

import numpy as np

COMPLETION_FOLDER = "experiments/evaluations/test_completions"

MODELS = [
    "alpaca_7b",
    # "llama-7b",
    # "sft-on-preferences-v1",
    "rlhf-v3-lr-5.0e-6-batch-16@gold-run",
    "test-save-alpaca@model-2048-prompts-batch-size-8",
    "shf-7b-gold-v1@step-0064",
    "shf-7b-gold-v1@step-8192",
    # "shf-7b-default",
    # "pythia-6.9B-deduped",
]
