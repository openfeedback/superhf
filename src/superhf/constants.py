"""Constant values."""

import re

# What we assume is the end of the prompt
PROMPT_DELIMITER = "\n\nAssistant:"

# Matches "\n\n{anything}:" for pruning additional generated turns
PROMPT_DELIMITER_REGEX = re.compile(r"\n\n[^:]+:")
