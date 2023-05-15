"""
Using GPT-4, get qualitative evaluations of the completions.
"""

from enum import Enum
import json
import os
from typing import Any
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import jsonlines
import numpy as np
import openai
from tqdm import tqdm

from evaluation_utils import trim_generations
from superhf.utils import set_seed


class EvaluationMode(Enum):
    """The mode of evaluation."""

    PREFERENCES = 0
    RELEVANCE = 1
    DIVERSITY = 2
    AVOIDANCE = 3
    GAMING = 4
    BIAS = 5


# Config
EVALUATION_MODE = EvaluationMode.PREFERENCES
MOCK_API = False
COMPLETION_PATHS = [
    "./experiments/evaluations/test_completions/llama-7b.json",
    "./experiments/evaluations/test_completions/alpaca_7b.json",
    "./experiments/evaluations/test_completions/sft-on-preferences-v1.json",
    "./experiments/evaluations/test_completions/rlhf-v3-lr-5.0e-6-batch-16@gold-run.json",
    "./experiments/evaluations/test_completions/shf-7b-default.json",
    "./experiments/openai_generations/gpt-3.5-turbo_2023-05-13_completions_output.json",
    "./experiments/openai_generations/gpt-4_2023-05-13_completions_output.json",
]
OPENAI_MODEL = "gpt-4"
OUTPUT_DIR = "./eval_results/gp4_qualitative"
PREFERENCE_COMPARISONS_PER_DATASET = 256
REQUEST_SLEEP_INTERVAL = 1  # seconds


def create_file_dir_if_not_exists(file_path: str) -> None:
    """Create the directory for a file if it doesn't already exist."""
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


@retry(wait=wait_random_exponential(min=0.5, max=10), stop=stop_after_attempt(8))
def query_api(system_prompt: str, user_prompt: str) -> Any:
    """Query the API for a completion."""
    if MOCK_API:
        return "A" if len(user_prompt) > len(system_prompt) else "B"
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1,
    )
    time.sleep(REQUEST_SLEEP_INTERVAL)
    return response.choices[0].message.content


def extract_prompt_and_completion(example: str) -> tuple[str, str]:
    """Given a `\n\nHuman:...\n\nAssistant...` example, extract the prompt and completion."""
    assistant_delimiter = "\n\nAssistant:"
    prompt_parts = example.split(assistant_delimiter)[0].split("\n\nHuman: ")
    assert len(prompt_parts) == 2
    prompt = prompt_parts[1]
    assert len(prompt) > 0
    completion = assistant_delimiter + example.split(assistant_delimiter)[1]
    completion = trim_generations([completion])[0].split(assistant_delimiter)[1]
    assert assistant_delimiter not in prompt
    assert assistant_delimiter not in completion
    return prompt, completion


def strip_and_remove_newlines(text: str) -> str:
    """Strip and remove newlines from a string."""
    return text.strip().replace("\n\n", " ").replace("\n", " ")


def run_preferences(names_to_completions: dict[str, Any]) -> None:
    """Run the preferences evaluation."""

    # pylint: disable=too-many-locals

    system_prompt = (
        "We are evaluating the holistic quality of two AI chatbot assistants, A and B."
        " Please select the response you prefer. Reply with just the letter and nothing"
        " else."
    )
    output_path = os.path.join(OUTPUT_DIR, "preferences.jsonl")
    create_file_dir_if_not_exists(output_path)
    with jsonlines.open(output_path, "w") as writer:
        for test_set in tqdm(
            names_to_completions["llama-7b.json"].keys(), desc="Test set"
        ):
            for index in tqdm(
                range(PREFERENCE_COMPARISONS_PER_DATASET), desc="Comparison"
            ):
                # Randomly choose 2 of the models to compare
                model_names_np = np.random.choice(
                    list(names_to_completions.keys()), size=2, replace=False
                )
                np.random.shuffle(model_names_np)
                model_names = [str(name) for name in model_names_np]

                # Get the completions for each model at this index
                model_a_example = names_to_completions[model_names[0]][test_set][index]
                model_b_example = names_to_completions[model_names[1]][test_set][index]
                _, model_a_completion = extract_prompt_and_completion(model_a_example)
                _, model_b_completion = extract_prompt_and_completion(model_b_example)
                model_a_completion = strip_and_remove_newlines(model_a_completion)
                model_b_completion = strip_and_remove_newlines(model_b_completion)

                # Also get the llama completion because we know it's prompt is good
                llama_example = names_to_completions["llama-7b.json"][test_set][index]
                prompt, _ = extract_prompt_and_completion(llama_example)

                # Format the final user prompt
                user_prompt = (
                    f"Prompt: {prompt}\nA: {model_a_completion}\nB:"
                    f" {model_b_completion}"
                )

                # Query the API
                rating = query_api(system_prompt, user_prompt)
                assert rating is not None

                # Write everything to the file
                writer.write(
                    {
                        "test_set": test_set,
                        "index": index,
                        "model_a": model_names[0],
                        "model_b": model_names[1],
                        "rating": rating,
                        "model_a_completion": model_a_completion,
                        "model_b_completion": model_b_completion,
                        "prompt": prompt,
                    }
                )


def main() -> None:
    """Run the evaluations for the given models and evaluation names."""

    # Set seed
    set_seed(66)

    # Load completions
    names_to_completions = {}
    for path in COMPLETION_PATHS:
        with open(path, "r", encoding="utf-8") as file:
            names_to_completions[path.rsplit("/", maxsplit=1)[-1]] = json.load(file)

    # Switch on evaluation type
    print(f"Running evaluations for {EVALUATION_MODE.name}...")
    if EVALUATION_MODE == EvaluationMode.PREFERENCES:
        run_preferences(names_to_completions)
    else:
        raise ValueError(f"Invalid evaluation mode: {EVALUATION_MODE}")


if __name__ == "__main__":
    main()
