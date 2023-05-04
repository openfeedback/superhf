"""
This is a file for running a reward model to score a text file of completions
"""

import argparse
import os
import json
import wandb

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
) -> None:
    """
    This loads the wandb data from the run. It is run sequentially, and it downloads all versions
    of the run until it runs out of versions.

    TODO: make this run in parallel

    Returns:
        completions: a list of lists of completions. Shape is (num_epochs, batch_size)
        scores: a list of lists of corresponding scores. Shape is (num_epochs, batch_size)
    """

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
    # Option 1 is we're loading a wandb artifact
    completions = load_completions_wandb()
    print(completions)


if __name__ == "__main__":
    main()
