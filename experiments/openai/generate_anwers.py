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
import random
import logging
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
        help="Output file",
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
        "--dataset",
        type=str,
        default="webgpt_comparisons",
        help=(
            "Dataset to use, loaded from superhf.data.get_superhf_prompts(). 'All'"
            " generates for all supported datasets except mock"
        ),
    )
    args = parser.parse_args()
    return args


# Define function to generate answers for a list of questions
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_answers(
    question="NO QUESTION PROVIDED", engine="gpt-3.5-turbo", max_tokens=64
) -> list:
    """
    Given a sigle question, generate and return an anwser for it.
    Retries with exponential backoff if the request fails.
    """
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

    if args.dataset == "all":
        for dataset in SUPPORTED_DATASETS:
            if dataset == "mock":
                continue
            generate_for_dataset(args, dataset)
    else:
        generate_for_dataset(args, args.dataset)


def generate_for_dataset(args, dataset: str) -> None:
    """
    Generate answers for a dataset
    """

    print("Generating completions for dataset: " + dataset)
    questions = get_superhf_prompts(dataset)
    if args.debug:
        questions = questions[:5]
        assert len(questions) < 10

    # randomize the order of the prompts
    random.seed(RANDOM_SEED)
    random.shuffle(questions)
    questions = questions[:NUMBER_OF_PROMPTS]

    questions = [
        question.split("\n\nAssistant:")[0].strip("\n\nHuman: ")
        for question in questions
    ]

    def generate_answer_wrapper(prompt):
        return generate_answers(prompt, args.engine, args.max_tokens)

    prompts = []
    for question in questions:
        prompt = {"role": "user", "content": question}
        prompts.append(prompt)
    answers = ["" for _ in prompts]
    n_requests = 0

    with ThreadPoolExecutor() as executor:
        answers = list(
            tqdm.tqdm(
                executor.map(generate_answer_wrapper, prompts), total=len(prompts)
            )
        )

    print(
        "Just processed "
        + str(n_requests)
        + " requests to OpenAI APIs with model "
        + args.engine
    )

    completions = []
    for i, question in enumerate(questions):
        # Stitch together the prompt and answer
        completion = "\n\nHuman: " + question + "\n\nAssistant: " + answers[i]
        completions.append(completion)

    # Write answers to a file
    output_file = dataset + "_" + args.output_file
    with open(output_file, "w", encoding="utf-8") as outfile:
        # write the result to the output file in json format
        json.dump({"completions": completions}, outfile)


if __name__ == "__main__":
    main()
