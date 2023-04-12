"""
Utility functions for the prompt_toolkit shell.
"""

import constants


def separate_prompt_from_completion(text: str) -> tuple[str, str]:
    """
    Given a completed prompt text, separate the part before and including the
    prompt delimiter from the part after.
    """
    prompt, completion = text.split(constants.PROMPT_DELIMITER, 1)
    prompt += constants.PROMPT_DELIMITER
    return prompt, completion
