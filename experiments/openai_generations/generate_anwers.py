"""
A script to get completions using an openai model from a list of prompts, and write the
completions to a file in json format. The data adheres to the format:

{
    "completions": [

        "\n\nHuman: What is the capital of France?\n\nAssistant: Paris",
        ...
    ]
}

# TODO: label the data with the model used to generate it
# TODO: Use the test splits we decided on

This script analyzes 1,000 randomly selected prompts.
"""
import sys
import argparse
import json
from datetime import datetime

# import random
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor


import tqdm
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from superhf.data import get_superhf_prompts
from superhf.constants import SUPPORTED_DATASETS

# TODO: Add to requirements.txt openai and tenacity

NUMBER_OF_PROMPTS = 1000
RANDOM_SEED = 0

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="OpenAI GPT-3 generation script")
    parser.add_argument(
        "--key",
        type=str,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="completions_output.json",
        help=(
            "Output file to append at the end of filename. Filename has the format "
            "engine_date_output_file"
        ),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Max tokens",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo",
        help="Model to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode, only work with 5 prompts from the dataset",
    )
    parser.add_argument(
        "--max_prompt_char_length",
        type=int,
        default=1024,
        help="Max prompt char length, anything longer get's filtered out",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=(
            "Dataset to use, loaded from superhf.data.get_superhf_prompts(). 'All'"
            " generates for all supported datasets except mock"
        ),
    )
    args = parser.parse_args()
    return args


# Define function to generate answers for a list of questions
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_answers(question=None, engine="gpt-3.5-turbo", max_tokens=64) -> list:
    """
    Given a sigle question, generate and return an anwser for it.
    Retries with exponential backoff if the request fails.
    """
    assert question is not None, "Question cannot be None"
    completion = openai.ChatCompletion.create(
        model=engine,
        messages=[question],
        max_tokens=max_tokens,
        stop=None,
        n=1,
        top_p=0.95,
        temperature=0.7,
    )
    return completion["choices"][0]["message"]["content"]


def main() -> None:
    """
    Main function
    """
    args = parse_args()
    # Set OpenAI API key
    openai.api_key = args.key

    completions_dict = {}
    if args.dataset == "all":
        for dataset in SUPPORTED_DATASETS:
            if dataset == "mock":
                continue
            completions = generate_for_dataset(args, dataset)
            completions_dict[dataset] = completions
    else:
        completions_dict[args.dataset] = generate_for_dataset(args, args.dataset)

    # get the current day
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")

    with open(
        f"{args.engine}_{date}_{args.output_file}", "w", encoding="utf-8"
    ) as file:
        json.dump(completions_dict, file, indent=4)
        file.write("\n")


def generate_for_dataset(args, dataset_name: str) -> List[str]:
    """
    Generate answers for a single dataset

    Args:
        args: Command line arguments
        dataset_name: Name of the dataset to generate answers for

    Returns:
        List of answers
    """
    # pylint: disable=too-many-locals

    print("Generating completions for dataset: " + dataset_name)
    prompts = get_superhf_prompts(dataset_name, split="test")
    # Filter out prompts that are too long
    old_prompt_count = len(prompts)
    prompts = [
        prompt for prompt in prompts if len(prompt) < args.max_prompt_char_length
    ]
    if args.debug:
        prompts = prompts[:5]
        assert len(prompts) < 10
    prompts = [
        prompt.split("\n\nAssistant:")[0].strip("\n\nHuman: ") for prompt in prompts
    ]
    print(
        f"Filtered {old_prompt_count - len(prompts)} prompts over "
        f"{args.max_prompt_char_length} chars from dataset {dataset_name}."
    )
    print(f"Loaded {len(prompts)} prompts for dataset {dataset_name}")

    def generate_answer_wrapper(prompt):
        return generate_answers(prompt, args.engine, args.max_tokens)

    gpt_prompts = []
    for prompt in prompts:
        input_prompt = {"role": "user", "content": prompt}
        gpt_prompts.append(input_prompt)
    answers = ["" for _ in gpt_prompts]

    with ThreadPoolExecutor() as executor:
        answers = list(
            tqdm.tqdm(
                executor.map(generate_answer_wrapper, gpt_prompts),
                total=len(gpt_prompts),
            )
        )
    n_requests = len(answers)
    print(
        "Just processed "
        + str(n_requests)
        + " requests to OpenAI APIs with model "
        + args.engine
    )

    completions = []
    for i, question in enumerate(prompts):
        # Stitch together the prompt and answer
        completion = "\n\nHuman: " + question + "\n\nAssistant: " + answers[i]
        completions.append(completion)

    # Write answers to a file
    if dataset_name == "openai/webgpt_comparisons":
        dataset_name = "webgpt_comparisons"
    # output_file = dataset_name + "_" + args.output_file
    # with open(output_file, "w", encoding="utf-8") as outfile:
    #     # write the result to the output file in json format
    #     json.dump({"completions": completions}, outfile)
    return completions


if __name__ == "__main__":
    main()
