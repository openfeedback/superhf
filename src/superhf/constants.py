"""Constant values."""

import re

# What we assume is the end of the prompt
PROMPT_DELIMITER = "\n\nAssistant:"

# Matches "\n\n{anything}:" for pruning additional generated turns
PROMPT_DELIMITER_REGEX_SIMPLE = re.compile(r"\n\n[^:]+:")

# New delimiter: the above, but also "\n", double quotes, "-", "Human", and "Assistant"
PROMPT_DELIMITER_REGEX_COMPLEX = re.compile(r"\n\n[^:]+:|\n|\"|-|Human|Assistant")
