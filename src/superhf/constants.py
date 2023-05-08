"""Constant values."""

import re

# What we assume is the end of the prompt
HUMAN_DELIMITER = "\n\nHuman:"
PROMPT_DELIMITER = "\n\nAssistant:"

# Matches "\n\n{anything}:" for pruning additional generated turns
PROMPT_DELIMITER_REGEX_SIMPLE = re.compile(r"\n\n[^:]+:")

# Matches the above, but also "Human", and "Assistant"
PROMPT_DELIMITER_REGEX_MEDIUM = re.compile(r"\n\n[^:]+:|Human|Assistant")

# Matches the above, but also "\n", double quotes, "-", "Human", and "Assistant"
PROMPT_DELIMITER_REGEX_COMPLEX = re.compile(r"\n\n[^:]+:|\n|\"|-|Human|Assistant")

# Datasets currently supported by SuperHF
SUPPORTED_DATASETS = [
    "anthropic-red-team",
    "openai/webgpt_comparisons",
    "anthropic-harmless-base",
    "anthropic-helpful-base",
    "mock",
]
