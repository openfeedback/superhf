"""
Utility functions for the prompt_toolkit shell.
"""

from superhf.constants import PROMPT_DELIMITER


def separate_prompt_from_completion(text: str) -> tuple[str, str]:
    """
    Given a completed prompt text, separate the part before and including the
    prompt delimiter from the part after.
    """
    prompt, completion = text.split(PROMPT_DELIMITER, 1)
    prompt += PROMPT_DELIMITER
    return prompt, completion
