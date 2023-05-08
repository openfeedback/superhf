"""
This is a file for running a reward model to score a text file of completions
"""

import argparse
import os
import json

from rlhf.rlhf_v1.run_rlhf import score_completions, load_reward_tokenizer
from rlhf.rlhf_v1.run_rlhf import load_reward_model
from openai_generations.read_answers import read_openai_answers
import torch
import accelerate

import wandb

from superhf.utils import print_gpu_utilization

# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     AutoModelForSeq2SeqLM,
#     HfArgumentParser,
#     LlamaTokenizer,
#     PreTrainedTokenizer,
#     AutoModelForCausalLM,
#     pipeline,
#     get_scheduler,
#     PreTrainedTokenizerBase,
#     PreTrainedModel,
# )
# from peft import LoraConfig, get_peft_model

# from superhf import constants
# from reward_modelling.reward_model import RewardModel

WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "rlhf-trl-v1"


def parse_args() -> argparse.Namespace:
    """
    This function parses the arguments for which completions to load, and
    which reward model(s) to use, if any.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--completions_file",
        type=str,
        default=None,
        help="Path to the file containing the completions to score",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="The wandb run id of the run to load completions from",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="The wandb project name of the run to load completions from",
    )
    parser.add_argument(
        "--wandb_entity_name",
        type=str,
        default=None,
        help="The wandb entity name of the run to load completions from",
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default=None,
        help="The name of the reward model to use",
    )

    return parser.parse_args()


def load_completions_wandb(
    entity_name=WANDB_ENTITY_NAME, project_name=WANDB_PROJECT_NAME, run_id="0tjh64pg"
) -> tuple[list[list[str]], list[list[float]]]:
    """
    Calls the wandb api to load completions and scores from a specified run.
    It is run sequentially, and it downloads all versions of the run until it
    encounters an error.

    TODO: make this run in parallel

    Returns:
        completions: a list of lists of completions. Shape is (num_epochs, batch_size)
        scores: a list of lists of corresponding scores. Shape is (num_epochs, batch_size)
    """

    assert run_id is not None, "Must specify a run id"

    # authenticate with wandb
    wandb.login()

    # initialize the API
    api = wandb.Api()

    completions = []
    scores = []
    version_num = 0

    while True:
        # retrieve the table data
        wandb_path_to_data = os.path.join(
            entity_name,
            project_name,
            "run-" + run_id + "-game_log:v" + str(version_num),
        )
        try:
            artifact = api.artifact(wandb_path_to_data, type="run_table")
            artifact_save_path = artifact.download()
        except wandb.errors.CommError:
            break

        # retrieve the json file
        with open(
            os.path.join(artifact_save_path, "game_log.table.json"),
            "r",
            encoding="utf-8",
        ) as table_file:
            game_log = json.load(table_file)

        completions.append(
            [game_log["data"][i][1] for i in range(len(game_log["data"]))]
        )
        scores.append([game_log["data"][i][2] for i in range(len(game_log["data"]))])
        version_num += 1

    return completions, scores


def main():
    """
    Main function for running the reward model
    """
    args = parse_args()
    # Option 1 is we're loading a wandb artifact
    # completions = load_completions_wandb()
    #  print(completions)
    # Option 2 is we're loading a file
    scores = None
    if args.completions_file is not None:
        completions = read_openai_answers(args.completions_file)
    else:
        completions, scores = load_completions_wandb(
            entity_name=args.wandb_entity_name,
            project_name=args.wandb_project_name,
            run_id=args.wandb_run_id,
        )
    print(completions[:5])
    if scores is None:
        accelerator = accelerate.Accelerator()
        reward_model_name = args.reward_model
        device = 0 if torch.cuda.is_available() else "cpu"

        reward_model, reward_model_pipe = load_reward_model(
            args.reward_model, device=device
        )
        if reward_model_pipe is None:
            tokenizer = load_reward_tokenizer(args.reward_model)
        if not isinstance(reward_model, str):
            reward_model = accelerator.prepare(reward_model)
        print(
            f"Instantiated reward model: {reward_model_name} on device"
            f" {reward_model.device}"
        )
        print(f"The device is {device}, with reward model placed on device.")
        print_gpu_utilization()
        if reward_model_pipe is None:
            scores = score_completions(
                reward_model=reward_model,
                reward_model_tokenizer=tokenizer,
                completions=completions,
            )
        else:
            scores = reward_model_pipe(completions)
    print(scores[:5])


if __name__ == "__main__":
    main()
