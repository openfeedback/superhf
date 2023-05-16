"""Utils for evaluation scripts."""

import os
import re

from tqdm import tqdm

from superhf.constants import PROMPT_DELIMITER, PROMPT_DELIMITER_REGEX_MEDIUM

VERBOSE = False


def separate_prompt_from_completion(text: str) -> tuple[str, str]:
    """
    Given a completed prompt text, separate the part before and including the
    prompt delimiter from the part after.
    """
    prompt, completion = text.split(PROMPT_DELIMITER, 1)
    prompt += PROMPT_DELIMITER
    return prompt, completion


def trim_generations(raw_completions: list[str]) -> list[str]:
    """
    Trim the generated completions to remove extra simulated turns of conversation.

    Return:
        A list of string wtih the prompt and modell response without any extra simulated
            conversation turns.
    """
    original_length = len(raw_completions)
    prompts_and_completions = [
        separate_prompt_from_completion(completion) for completion in raw_completions
    ]
    trimmed_completions: list[str] = []
    model_completion_lengths: list[int] = []
    for prompt, completion in prompts_and_completions:
        if VERBOSE and completion == "":
            tqdm.write("WARNING: Completion is empty.")
        stripped_completion = re.split(
            PROMPT_DELIMITER_REGEX_MEDIUM, completion, maxsplit=1
        )[0].strip()
        if VERBOSE and completion != "" and stripped_completion == "":
            tqdm.write("WARNING: Stripped completion is empty but completion wasn't.")
        trimmed_completions.append(prompt + " " + stripped_completion)
        model_completion_lengths.append(len(stripped_completion))

    assert len(trimmed_completions) == original_length, (
        "The number of Trimmed completions should be the same as the number of original"
        " completions."
    )
    return trimmed_completions


def create_file_dir_if_not_exists(file_path: str) -> None:
    """Create the directory for a file if it doesn't already exist."""
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)