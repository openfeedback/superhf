"""
This is a file for running a reward model to score a text file of completions
"""

import argparse
import os
import json
from typing import List
import yaml

# pylint: disable=import-error
from rlhf.rlhf_v1.run_rlhf import score_completions, load_reward_tokenizer
from rlhf.rlhf_v1.run_rlhf import load_reward_model
from openai_generations.read_answers import read_openai_answers

import torch

from accelerate import Accelerator, find_executable_batch_size

import wandb

from superhf.data import get_superhf_prompts
from superhf.utils import print_gpu_utilization
from superhf.training import SuperHFTrainingArguments, SuperHFTrainer
from superhf.filtering import CompletionFilterTopK
from superhf.mocking import MockRewardModel
from experiments.superhf.shf_iterative_v1.run_shf_iterative import (
    load_language_model,
    load_language_tokenizer,
)


WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "rlhf-trl-v1"


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

    return yaml_args


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


def generate_completions_from_lm(args):
    """
    Given a language model, loads in test prompts and then generates completions
    """
    device = torch.device(
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    )
    # Get the prompt dataset
    prompts: list[str] = []
    for dataset_name in wandb.config.prompt_dataset_names:
        prompts.extend(get_superhf_prompts(dataset_name, split="test"))

    # Filter out prompts that are too long
    old_prompt_count = len(prompts)
    prompts = [
        prompt
        for prompt in prompts
        if len(prompt) < wandb.config.max_prompt_char_length
    ]
    print(
        f"Filtered {old_prompt_count - len(prompts)} prompts over "
        f"{wandb.config.max_prompt_char_length} chars."
    )
    print(f"Loaded {len(prompts)} prompts.")
    print_gpu_utilization()

    language_model = load_language_model(device, args.language_model)
    print_gpu_utilization()

    language_tokenizer = load_language_tokenizer(args.language_model)

    # Check for unix
    if os.name == "posix":
        print("Compiling models...")
        print(type(language_model))
        language_model = torch.compile(language_model)
        print("Compiled models.")
        print_gpu_utilization()

    training_args = SuperHFTrainingArguments(
        minibatch_size_generating=16, max_new_tokens=128, temperature=0.7
    )
    # this completion filter is not used because we do not call trainer.train
    completion_filter = CompletionFilterTopK(1)

    reward_model_train = MockRewardModel()
    reward_tokenizer_train = load_reward_tokenizer("mock")

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

    if prompt_batch_size == 0:
        prompt_batch_size = len(prompts)

    completions_raw = find_executable_batch_size(
        trainer.generate_completions,
        trainer.training_args.minibatch_size_generating,
    )(prompts)

    print(f"The length of completions is {len(completions_raw)}")

    return completions_raw


def main() -> None:
    """
    Main function for running the reward model
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    args = parse_args()

    completions: List[str] = []
    scores: List[float] = []
    if args.completions_file is not None:
        if args.openai:
            completions = read_openai_answers(args.completions_file)
        else:
            # raise an unimplemented error
            raise NotImplementedError(
                "Need to implement general loading from a file. "
                " Specify openai flag otherwsie"
            )

    else:
        if args.wandb_run_id is not None:
            completions_batched, scores_batched = load_completions_wandb(
                entity_name=args.wandb_entity_name,
                project_name=args.wandb_project_name,
                run_id=args.wandb_run_id,
            )
            for batch in completions_batched:
                completions.extend(batch)
            for batch_scores in scores_batched:
                scores.extend(batch_scores)
        elif args.language_model is not None:
            print(f"Generating completions with language model {args.language_model}")
            completions = generate_completions_from_lm(args)
        else:
            raise NotImplementedError(
                "Must specify at least a language model or wandb to load generations"
                " from, or a comletions file with completions"
            )
    print("first completion is:")
    print(completions[0])
    if not scores:
        accelerator = Accelerator()
        reward_model_name = args.reward_model
        device = 0 if torch.cuda.is_available() else "cpu"

        reward_model, reward_model_pipe = load_reward_model(
            args.reward_model, device=device
        )
        if reward_model_pipe is None:
            tokenizer = load_reward_tokenizer(args.reward_model)
        if not isinstance(reward_model, str):
            reward_model = accelerator.prepare(reward_model)
        print(f"Instantiated reward model: {reward_model_name} on device {device}")
        print(f"The device is {device}, with reward model placed on device.")
        print_gpu_utilization()
        if reward_model_pipe is None:
            scores = score_completions(
                reward_model=reward_model,
                reward_model_tokenizer=tokenizer,
                completions=completions,
            )
        else:
            if not isinstance(reward_model, str):
                print("WARNING: reward model is str, and pipeline is not None.")
            scores = reward_model_pipe(completions)
    print("First score is:")
    print(scores[0])


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
#         print_gpu_utilization()

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
