"""
This is a file for running a reward model to score a text file of completions
"""

import argparse
import os
import json
from typing import Optional, List, Dict, Any
import yaml

from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    GPT2Tokenizer,
)
import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator, find_executable_batch_size

from model_loading import load_eval_model_and_tokenizer
from tqdm import tqdm


import wandb

from superhf.data import get_superhf_prompts
from superhf.utils import print_memory_utilization
from superhf.training import SuperHFTrainingArguments, SuperHFTrainer
from superhf.filtering import CompletionFilterTopK
from superhf.mocking import MockRewardModel

# Default parameters
WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "rlhf-trl-v1"

# folder to be used for saving and loading test set eval data
TEST_COMPLETIONS_DIR = "test_completions"
TEST_SCORES_DIR = "test_scores"


def parse_args() -> argparse.Namespace:
    """
    This function parses the arguments for which completions to load, and
    which reward model(s) to use, if any.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config file containing the completions to score",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        yaml_args = yaml.safe_load(file)

    values_dict = {k: v["value"] for k, v in yaml_args.items()}
    return argparse.Namespace(**values_dict)


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


def read_openai_completions(input_file: str) -> list[str]:
    """
    Read openai completions from a file.

    Args:
        input_file: The path to the file containing the answers

    Returns:
        A list of answers
    """
    # check if it's a file
    if not os.path.isfile(input_file):
        raise ValueError(
            f"Input file {input_file} does not exist. cwd is {os.getcwd()}"
        )

    with open(input_file, "r", encoding="utf-8") as file:
        completions = json.load(file)["completions"]
    return completions


def score_completions(
    starting_batch_size_rm: int,
    reward_model: PreTrainedModel,
    reward_model_tokenizer: PreTrainedTokenizer,
    completions: List[str],
    reward_model_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Scores the completions from the reward model.

    Args:
        reward_model: The reward model to use.
        reward_model_tokenizer: The tokenizer for the reward model.
        completions: The completions to score, in text format.
        reward_model_kwargs: The arguments to pass to the reward model.

    Returns:
        A list of scores (logits as torch.Tensor) for each completion.
    """
    if reward_model_kwargs is None:
        reward_model_kwargs = {}

    reward_model_tokenizer.pad_token = reward_model_tokenizer.eos_token
    tqdm.write(f"We are scoring {len(completions)} completions.")
    completions_tokenized = reward_model_tokenizer(
        completions, padding=True, truncation=True, return_tensors="pt"
    )
    completions_tokenized = completions_tokenized.to(reward_model.device)
    tqdm.write(f"Moved completions to {reward_model.device}.")
    assert reward_model.device != "cpu", "Reward model must be on GPU."

    # Create a DataLoader
    tensor_dataset = TensorDataset(
        completions_tokenized["input_ids"], completions_tokenized["attention_mask"]
    )
    data_loader = DataLoader(tensor_dataset, batch_size=starting_batch_size_rm)

    scores = []
    with torch.no_grad():
        tqdm.write("Scoring completions.")
        for batch in tqdm(data_loader, desc="scoring completion batch"):
            input_ids, attention_mask = batch
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **reward_model_kwargs,
            }
            batch_scores = reward_model(**model_inputs)
            if not isinstance(batch_scores, torch.Tensor):
                batch_scores: torch.Tensor = batch_scores.logits
            scores.append(batch_scores)
    return torch.cat(scores), starting_batch_size_rm


def generate_completions_from_lm(
    starting_batch_size, language_model, existing_model, prompts_dict
):
    """
    Given a language model and test prompts and then generates completions

    Args:
        args.language_model:
        completions_dict: the dictionary containing dataset: list of completions
    """
    # pylint: disable=too-many-locals
    print_memory_utilization()

    language_model_path = language_model.split("@")[0]
    try:
        revision = language_model.split("@")[1]
    except IndexError:
        revision = None
    language_model, language_tokenizer = load_eval_model_and_tokenizer(
        language_model_path, existing_model, revision=revision, model_type="language"
    )
    print_memory_utilization()

    # Check for unix
    # if os.name == "posix":
    # TODO Update torch and compile it
    #     print("Compiling models...")
    #     print(type(language_model))
    #     language_model = torch.compile(language_model)
    #     print("Compiled models.")
    #     print_memory_utilization()

    training_args = SuperHFTrainingArguments(
        minibatch_size_generating=64,
        max_new_tokens=128,
        temperature=0.7,
        kl_coefficient=-1,
        superbatch_size=1,  # we don't want to duplicate prompts
    )
    # this completion filter is not used because we do not call trainer.train
    completion_filter = CompletionFilterTopK(1)

    reward_model_train = MockRewardModel()
    # load gpt2 tokenizer
    reward_tokenizer_train = GPT2Tokenizer.from_pretrained("gpt2")

    trainer = SuperHFTrainer(
        language_model=language_model,
        reward_model_train=reward_model_train,
        reward_model_val=None,
        language_tokenizer=language_tokenizer,
        reward_tokenizer_train=reward_tokenizer_train,
        reward_tokenizer_val=None,
        completion_filter=completion_filter,
        training_args=training_args,
    )

    completions_dict = {}
    for dataset_name in tqdm(prompts_dict.keys(), desc="Datasets"):
        prompts = prompts_dict[dataset_name]
        if starting_batch_size == 0:
            starting_batch_size = len(prompts)

        completions_raw = find_executable_batch_size(
            trainer.generate_completions,
            starting_batch_size,
        )(prompts)

        tqdm.write(
            f"The length of completions is {len(completions_raw)} for dataset"
            f" {dataset_name}"
        )
        completions_dict[dataset_name] = completions_raw

    return completions_dict, language_model, starting_batch_size


def load_prompts_dictionary(args):
    """
    Args:
        prompt_dataset_names: the list of dataets to load
    Returns:
        A dicitonary with the dataset name: list of prompts
    """
    # Get the prompt dataset
    completions_dict = {}
    for dataset_name in args.prompt_dataset_names:
        prompts = get_superhf_prompts(dataset_name, split="test")

        # Filter out prompts that are too long
        old_prompt_count = len(prompts)
        prompts = [
            prompt for prompt in prompts if len(prompt) < args.max_prompt_char_length
        ]
        print(
            f"Filtered {old_prompt_count - len(prompts)} prompts over "
            f"{args.max_prompt_char_length} chars from dataset {dataset_name}"
        )
        print(f"Loaded {len(prompts)} prompts for dataset {dataset_name}")
        completions_dict[dataset_name] = prompts
    return completions_dict


def load_completions_json(completions_file):
    """
    Loads completions from a json file

    Args:
        completions_file: the json file to load

    Returns:
        A dictionary with dataset name: list of completions
    """
    with open(completions_file, "r", encoding="utf-8") as file:
        completions_dict = json.load(file)
    return completions_dict


def save_completions(completions_dict, filename):
    """
    Saves completions as a json to TEST_COMPLETIONS_DIR

    Args:
        completions_dict:

    Returns:
        The filename of where the completions were saved as a json file
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(completions_dict, file)
    tqdm.write(f"Saved completions to {filename}")
    return filename

    # if completions_file is not None:
    #     if "openai" in args.completions_file:
    #         completions = read_openai_completions(args.completions_file)
    #     else:
    #         # raise an unimplemented error
    #         raise NotImplementedError(
    #             "Need to implement general loading from a file. "
    #             " Specify openai flag otherwsie"
    #         )
    # else:


def save_scores(scores_dict, filename):
    """
    Saves scores as a json to TEST_SCORES_DIR
    """
    tqdm.write(f"Saving scores to {filename}")
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(scores_dict, file)
    return filename


def main() -> None:
    """
    Main function for running the reward model
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    args = parse_args()

    try:
        language_model_names = args.language_model_names
    except AttributeError:
        language_model_names = None
    try:
        scoring_mode = args.scoring_mode
    except AttributeError:
        scoring_mode = False
    # print(f"Current directory is {os.getcwd()}")
    script_path_dir = os.path.dirname(os.path.abspath(__file__))
    # get all the filenames in TEST_COMPLETIONS_DIR
    test_completions_dir = os.path.join(script_path_dir, TEST_COMPLETIONS_DIR)
    if not os.path.exists(test_completions_dir):
        os.makedirs(test_completions_dir)
    already_generated_completions = os.listdir(test_completions_dir)
    already_generated_completions = [
        completion.split(".")[0] for completion in already_generated_completions
    ]

    # if args.wandb_run_id is not None:
    #     # grab completions from wandb
    #     completions_batched, scores_batched = load_completions_wandb(
    #         entity_name=args.wandb_entity_name,
    #         project_name=args.wandb_project_name,
    #         run_id=args.wandb_run_id,
    #     )
    #     for batch in completions_batched:
    #         completions.extend(batch)
    #     for batch_scores in scores_batched:
    #         scores.extend(batch_scores)
    if language_model_names is not None:
        starting_batch_size_lm = args.starting_batch_size_lm
        prompts_dict = load_prompts_dictionary(args)
        cached_model = None
        for language_model_name in tqdm(
            args.language_model_names, desc="Language models"
        ):
            #  it does, don't generate here
            language_model_base_name = language_model_name.split(os.path.sep)[-1]
            if language_model_base_name in already_generated_completions:
                tqdm.write(
                    "Already generated completions for language model"
                    f" {language_model_name}"
                )
                continue
            tqdm.write(
                f"Generating completions with language model {language_model_name}"
            )
            completions_dict, cached_model, starting_batch_size_lm = (
                find_executable_batch_size(
                    generate_completions_from_lm, starting_batch_size_lm
                )(language_model_name, cached_model, prompts_dict)
            )
            filename = os.path.join(
                test_completions_dir, f"{language_model_base_name}.json"
            )
            save_completions(completions_dict, filename)

        # del prompts_dict
        # del cached_model
    elif not scoring_mode:
        raise NotImplementedError(
            "Must specify at least a language model or wandb to load generations from,"
            " or a completions file with completions from openAI. Or be in scoring mode"
        )

    scores_dict = {}
    if scoring_mode:
        starting_batch_size_rm = args.starting_batch_size_rm
        # we are in scoring mode, so score completions
        accelerator = Accelerator()
        reward_model, reward_tokenizer = load_eval_model_and_tokenizer(
            args.reward_model, model_type="reward"
        )
        if not isinstance(reward_model, str):
            reward_model = accelerator.prepare(reward_model)
        print_memory_utilization()

        # TODO: Find executable batch size
        already_generated_completions = os.listdir(test_completions_dir)
        already_generated_completions = [
            completion.split(".")[0] for completion in already_generated_completions
        ]
        test_scores_dir = os.path.join(script_path_dir, TEST_SCORES_DIR)
        if not os.path.exists(test_scores_dir):
            os.makedirs(test_scores_dir)
        already_generated_scores = os.listdir(test_scores_dir)
        already_generated_scores = [
            score.split(".")[0] for score in already_generated_scores
        ]
        for language_model_name in tqdm(
            already_generated_completions, desc="Scoring per language models"
        ):
            if language_model_name in already_generated_scores:
                tqdm.write(
                    f"Already generated scores for language model {language_model_name}"
                )
                continue
            language_model_base_name = language_model_name.split(os.path.sep)[-1]
            completions_dict = load_completions_json(
                os.path.join(test_completions_dir, f"{language_model_base_name}.json")
            )
            for dataset_name in tqdm(args.prompt_dataset_names, desc="Datasets"):
                completions = completions_dict[dataset_name]
                scores, starting_batch_size_rm = find_executable_batch_size(
                    score_completions, starting_batch_size_rm
                )(
                    reward_model=reward_model,
                    reward_model_tokenizer=reward_tokenizer,
                    completions=completions,
                )
                scores_dict[dataset_name] = scores.tolist()
            filename = os.path.join(test_scores_dir, f"{language_model_base_name}.json")
            save_scores(scores_dict, filename)

    # print(f"there are {len(completions_dict)} completions")
    # print(f"there are {len(scores)} scores")

    # directory for test_completions
    # json with generations
    # in the json file, generations are per dataset
    # directory for rewards with reward model
    # json name with the scores
    # statistics lilke mean, std, quartiles per dagtaset
    # for language models grab just the last part, and include the @

    # look through the test_completions
    # reward model
    # -> model_name
    # -> dataset1
    # -> completions
    # -> scores
    # -> dataset2
    # -> completions
    # -> scores
    # -> dataset3
    # -> completions
    # -> scores
    # output = {}
    # try decreasing learning rate as we increase batch size
    # get rid of kl annealing


if __name__ == "__main__":
    main()


#     completions_raw = find_executable_batch_size(
#                 self.generate_completions,
#                 self.training_args.minibatch_size_generating,
#             )(superbatch_prompts)

# class LMResponseGenerator():
#     def __init__(self,
#         language_model: PreTrainedModel,
#         reward_model_train: PreTrainedModel,
#         reward_model_val: Optional[PreTrainedModel],
#         language_tokenizer: PreTrainedTokenizerBase,
#         reward_tokenizer_train: PreTrainedTokenizerBase,
#         reward_tokenizer_val: Optional[PreTrainedTokenizerBase],
#         completion_filter: CompletionFilterBase,
#         training_args: SuperHFTrainingArguments):


#     def collate_fn_lm_completions(self, batch: list[str]) -> BatchEncoding:
#         """
#         Collate function for the language model completions DataLoader.

#         Prepends the system prompt to each prompt. By default this is the empty string.
#         """
#         return self.language_tokenizer(
#             batch,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=self.training_args.max_length_rm,
#         )

#     def generate_completions(
#         self,
#         minibatch_size: int,
#         superbatch_prompts: list[str],
#     ) -> list[str]:
#         """
#         Generate completions for the prompts in the superbatch.

#         Args:
#             minibatch_size: The minibatch size to use for generation.
#             superbatch_prompts: The prompts in the superbatch.
#         """

#         tqdm.write(f"Trying generation with batch size {minibatch_size}")
#         print_memory_utilization()

#         # Duplicate each prompt superbatch_size numbers time with system prompt
#         system_prompt = self.training_args.conversation_prompt
#         prompt_batch_duplicated = [
#             system_prompt + prompt
#             for prompt in superbatch_prompts
#             for _ in range(self.training_args.superbatch_size)
#         ]

#         completion_dataloader = DataLoader(
#             ListDataset(prompt_batch_duplicated),
#             batch_size=minibatch_size,
#             collate_fn=self.collate_fn_lm_completions,
#             pin_memory=True,
#         )

#         completions_encoded: list[TensorType["batch", "seq_len"]] = []
#         with torch.no_grad():
#             for minibatch in tqdm(
#                 completion_dataloader,
#                 desc="Generation",
#                 total=len(completion_dataloader),
#                 file=sys.stdout,
#             ):
#                 encodings = minibatch
#                 # with torch.cuda.amp.autocast(dtype=self.training_args.dtype):  # type: ignore
#                 encodings.to(self.language_model.device)
#                 outputs = self.language_model.generate(  # type: ignore
#                     **encodings,
#                     max_new_tokens=self.training_args.max_new_tokens,
#                     temperature=self.training_args.temperature,
#                     top_p=self.training_args.top_p,
#                     do_sample=True,
#                     num_return_sequences=1,
#                     pad_token_id=self.language_tokenizer.pad_token_id,
#                     logits_processor=self.training_args.logits_processors,
#                 )
#                 completions_encoded.extend(outputs.to("cpu"))
#         # completions_gathered: list[str] = accelerator.gather(
#         #     completions
#         # )  # Not needed on single GPU
#         completions_text: list[str] = self.language_tokenizer.batch_decode(
#             completions_encoded, skip_special_tokens=True
#         )
#         return completions_text

#     def collate_fn_rm_train(
#         self, completions: list[str]
#     ) -> tuple[list[str], BatchEncoding, list[int]]:
#         """
#         Collate function for the reward model's dataloader.

#         Takes encoded completions from the language model, decodes them, encodes them for the
#         reward model, and returns both the decoded completion text and re-encoded completions.
#         """

#         # Remove completions after any extra "\n\nHuman:", "\n\nA:", "\n\nH:", or similar.
#         # This is to prevent the model from learning to generate additional turns of conversation.
#         prompts_and_completions = [
#             separate_prompt_from_completion(completion) for completion in completions
#         ]
#         completions_for_lm: list[str] = []
#         completions_for_rm: list[str] = []
#         completion_lengths: list[int] = []
#         for prompt, completion in prompts_and_completions:
#             stripped_completion = re.split(
#                 constants.PROMPT_DELIMITER_REGEX_MEDIUM, completion, maxsplit=1
#             )[0].strip()
#             completion_lengths.append(len(stripped_completion))
#             joined_completion_normal = prompt + " " + stripped_completion
#             completions_for_lm.append(joined_completion_normal)
#             if self.training_args.reward_model_is_steamshp:
#                 # Handle the weird SteamSHP format
#                 prompt_only = prompt.split(constants.HUMAN_DELIMITER)[1].split(
#                     constants.PROMPT_DELIMITER
#                 )[0]
#                 joined_completion_shp = (
#                     f"POST:{prompt_only}\n\n"
#                     f" RESPONSE A: {stripped_completion}\n\n RESPONSE B: .\n\n Which"
#                     " response is better? RESPONSE"
#                 )
#                 completions_for_rm.append(joined_completion_shp)
#             else:
#                 # Concat normally
#                 completions_for_rm.append(joined_completion_normal)

#         return (
#             completions_for_lm,
#             self.reward_tokenizer_train(
#                 completions_for_rm,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=self.training_args.max_length_rm,
#             ).to(self.reward_model_train.device),
#             completion_lengths,
#         )
