"""
Script for running RLHF to compare with SuperHF.

Utilizes hugging face pipelines for the reward model, and PPO trainer from TRL
to train the language model.

Implements LoRA based on this guide -
https://github.com/lvwerra/trl/blob/52fecee8839ad826ad1e6c83a95c99a4116e98d2/
examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py

Example usage:
    python run_rlhf.py \
        --config configs/rlhf_v1.yaml \
        --notes "Testing RLHF with TRL"
        --sweep_id xxxxx
"""

import os
import random
import re
from typing import Optional, TypeVar, List, Union, Tuple
from dataclasses import dataclass, field
import time

from tqdm import tqdm

import torch
from torch.optim import Adam

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    LlamaTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    pipeline,
    get_scheduler,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model

from datasets import Dataset

from utils import separate_prompt_from_completion

from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)

# from trl.core import LengthSampler
import wandb

from superhf import constants
from superhf.data import get_superhf_prompts
from superhf.utils import set_seed, print_gpu_utilization
from superhf.mocking import MockRewardModel
from reward_modelling.reward_model import RewardModel

T = TypeVar("T")

WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "rlhf-trl-v1"


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit
    # mode models like gpt-neo* models are more suitable.
    config: Optional[str] = field(
        # Get file pa`th relative to this file
        default=os.path.join(os.path.dirname(__file__), "configs", "rlhf_config.yaml"),
        metadata={"help": "The name of the Weights and Biases config to use."},
    )
    notes: Optional[str] = field(
        default="", metadata={"help": "notes to add to the wandb run"}
    )
    sweep_id: Optional[str] = field(
        default="", metadata={"help": "sweep id to use to for a sweep"}
    )


def parse_args():
    """
    This function parses the arguments passed to the script. It utilizes the
    HfArgumentParser class from the transformers library.
    """
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    return script_args


def build_dataset(
    dataset_names,
    # tokenizer,
    max_prompt_char_length=1024,
    debug_max_prompts=0,
    conversation_prompt="",
    seed=66,
):
    """
    Currentlty we don't use the tokenizer becauses the internal trainer
    for some reason throws away the tokenized exmples.

    Returns:
        a pytorch dataset that implements the __getitem__ and __len__ methods.
        PPO trainer converts this to a pytorch dataloader.
        torch.utils.data.Dataset
    # TODO move into src/data.py
    """
    set_seed(seed)
    prompts: list[str] = []
    for dataset in dataset_names:
        prompts.extend(get_superhf_prompts(dataset))

    random.shuffle(prompts)

    # Filter out prompts that are too long
    old_prompt_count = len(prompts)
    prompts = [
        conversation_prompt + prompt
        for prompt in prompts
        if len(conversation_prompt + prompt) < max_prompt_char_length
    ]
    print(
        f"Filtered {old_prompt_count - len(prompts)} prompts over "
        f"{max_prompt_char_length} chars."
    )

    # Only load the first section of prompts
    if debug_max_prompts != 0:
        prompts = prompts[:debug_max_prompts]

    print(f"Loaded {len(prompts)} prompts.")

    def tokenize(sample):
        dictionized_example = {}
        # dictionized_example["input_ids"] = tokenizer.encode(sample)
        dictionized_example["query"] = (
            sample  # tokenizer.decode(dictionized_example["input_ids"])
        )
        return dictionized_example

    prompts_2 = [tokenize(prompt) for prompt in prompts]
    prompts_3 = {"query": [d["query"] for d in prompts_2]}
    dataset = Dataset.from_dict(prompts_3)
    dataset.set_format(type="torch")
    return dataset


def load_reward_tokenizer(reward_tokenizer_name: str) -> PreTrainedTokenizerBase:
    """Load the reward model's tokenizer."""
    if reward_tokenizer_name == "mock":
        reward_tokenizer_name = "gpt2"
    elif (
        "rm_combined" in reward_tokenizer_name or "oliversssf2" in reward_tokenizer_name
    ):
        reward_tokenizer_name = "EleutherAI/gpt-neo-1.3B"

    if "llama" in reward_tokenizer_name or "alpaca" in reward_tokenizer_name:
        # Fix for misnamed class in the NLP Cluster's Alpaca tokenizer config
        reward_tokenizer_train = LlamaTokenizer.from_pretrained(reward_tokenizer_name)
    else:
        reward_tokenizer_train = AutoTokenizer.from_pretrained(reward_tokenizer_name)

    print(f"Instantiated reward tokenizer {reward_tokenizer_name}")
    return reward_tokenizer_train


def load_reward_model(
    reward_model_name: str, device: torch.device
) -> Tuple[PreTrainedModel, Union[pipeline, None]]:
    """Load the reward model.

    Returns either None for pipeline if using oliver's rm, otherwise returns a pipeline object.
    """
    reward_model_pipe = None
    if reward_model_name == "mock":
        reward_model_train = MockRewardModel()
    elif "rm_combined" in reward_model_name or "oliversssf2" in reward_model_name:
        reward_model_train = RewardModel.from_pretrained(
            reward_model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,  # Force half for these large RMs
        )
    elif "SteamSHP-flan-t5" in reward_model_name:
        reward_model_train = AutoModelForSeq2SeqLM.from_pretrained(reward_model_name)
    else:
        # reward_model_train = AutoModelForSequenceClassification.from_pretrained(
        #     reward_model_name,
        #     low_cpu_mem_usage=True,
        #     torch_dtype=torch.bfloat16,
        # )
        reward_model_train = reward_model_name
        reward_model_pipe = pipeline(
            model=reward_model_name,
            device=device,  # TODO move device out of here for style reasons.
        )

    if not isinstance(reward_model_name, str):
        reward_model_train.to(device)
        reward_model_train.device = device
    print(f"Instantiated reward model: {reward_model_name} on device {device}")
    return reward_model_train, reward_model_pipe


def get_configs():
    """
    Organizes all the configs into one place, and returns all of them.
    """
    # TODO implement this to organize codes
    ppo_config = PPOConfig(
        model_name=wandb.config.model_name,
        steps=20000,
        learning_rate=wandb.config.learning_rate,
        adap_kl_ctrl=True,
        init_kl_coef=wandb.config.init_kl_coef,
        clip_kl=wandb.config.clip_kl,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        batch_size=wandb.config.batch_size,
        forward_batch_size=None,
        mini_batch_size=wandb.config.mini_batch_size,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        ppo_epochs=4,
        remove_unused_columns=True,
        log_with=wandb.config.log_with,
        tracker_kwargs={},
        accelerator_kwargs={},
        tracker_project_name="trl",
        max_grad_norm=None,
        seed=66,
        optimize_cuda_cache=False,
    )

    assert ppo_config.mini_batch_size <= ppo_config.batch_size

    lora_config = None
    if wandb.config.lora_r != 0 and wandb.config.lora_alpha != 0:
        # Set up low-rank adapters (LoRA)
        target_modules = (
            wandb.config.lora_target_modules
            if len(wandb.config.lora_target_modules) > 0
            else None
        )
        lora_config = LoraConfig(
            r=wandb.config.lora_r,
            lora_alpha=wandb.config.lora_alpha,
            target_modules=target_modules,  # handled automatically by peft
            lora_dropout=wandb.config.lora_dropout,
            task_type="CAUSAL_LM",
            fan_in_fan_out=False,
        )

    reward_model_kwargs = {
        "function_to_apply": "none",
        "batch_size": wandb.config.batch_size,
    }  # arguments for the reward pipeline.

    language_tokenizer = None
    if "llama" in ppo_config.model_name or "alpaca" in ppo_config.model_name:
        # Fix for misnamed class in the NLP Cluster's Alpaca tokenizer config
        language_tokenizer = LlamaTokenizer.from_pretrained(
            ppo_config.model_name, padding_side="left"
        )
    else:
        language_tokenizer = AutoTokenizer.from_pretrained(
            ppo_config.model_name, padding_side="left"
        )

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": wandb.config.top_p,
        "do_sample": True,
        "pad_token_id": language_tokenizer.eos_token_id,
        "max_new_tokens": wandb.config.max_new_tokens,
    }

    return (
        ppo_config,
        lora_config,
        reward_model_kwargs,
        language_tokenizer,
        generation_kwargs,
    )


def score_completions(
    reward_model: PreTrainedModel,
    reward_model_tokenizer: PreTrainedTokenizer,
    completions: List[str],
) -> torch.Tensor:
    """
    Scores the completions from the reward model.

    Args:
        reward_model: The reward model to use.
        reward_model_tokenizer: The tokenizer for the reward model.
        completions: The completions to score, in text format.
        reward_model_kwargs: The arguments to pass to the reward model.

    Returns:
        A list of scores (logits) for each completion.
    """
    reward_model_tokenizer.pad_token = reward_model_tokenizer.eos_token
    print(f"We are scoring {len(completions)} completions.")
    completions_tokenized = reward_model_tokenizer(
        completions, padding=True, truncation=True, return_tensors="pt"
    )
    completions_tokenized = completions_tokenized.to(reward_model.device)
    print(
        f"Moved completions to {reward_model.device}. THey are on device"
        f" {completions_tokenized.device}."
    )
    assert reward_model.device != "cpu", "Reward model must be on GPU."
    with torch.no_grad():
        print("Scoring completions.")
        scores = reward_model(**completions_tokenized)
    if not isinstance(scores, torch.Tensor):
        scores: torch.Tensor = scores.logits
    return scores


def main(script_args: ScriptArguments):
    """
    Main function
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        notes=script_args.notes,
        save_code=True,
        config=script_args.config,
    )

    # Enable tf32 training if supported
    if (
        wandb.config.try_tf32_training
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8
    ):
        print("Enabling tf32 training.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # the ppo trainer deletes the wandb.config object for some reason (only verified problem on cpu)
    normalize_reward = wandb.config.normalize_reward
    run_name = wandb.run.name
    hub_repo_id = wandb.config.hub_repo_id
    save_every = wandb.config.save_every
    reward_mean = wandb.config.reward_mean
    reward_model_name = wandb.config.reward_model_name

    (
        ppo_config,
        lora_config,
        reward_model_kwargs,
        language_tokenizer,
        generation_kwargs,
    ) = get_configs()

    language_model = AutoModelForCausalLM.from_pretrained(ppo_config.model_name)
    model_ref = None
    if wandb.config.lora_r != 0 and wandb.config.lora_alpha != 0:
        # Set up low-rank adapters (LoRA)
        language_model = get_peft_model(language_model, lora_config)
        language_model.print_trainable_parameters()
    else:
        # Copy the entire model in order to calculate the KL divergence
        model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
            ppo_config.model_name
        )
    language_model = AutoModelForCausalLMWithValueHead.from_pretrained(language_model)

    # set seed before initializing value head for deterministic eval
    dataset = build_dataset(
        wandb.config.dataset_names,
        # language_tokenizer,
        max_prompt_char_length=wandb.config.max_prompt_char_length,
        debug_max_prompts=wandb.config.debug_max_prompts,
        conversation_prompt=wandb.config.conversation_prompt,
        seed=ppo_config.seed,
    )

    def collator(data):
        """
        Collator for the dataset
        """
        return dict((key, [d[key] for d in data]) for key in data[0])

    scheduler = None
    optimizer = Adam(
        filter(lambda p: p.requires_grad, language_model.parameters()),
        lr=wandb.config.learning_rate,
    )
    if wandb.config.scheduler_name != "":
        num_training_steps = len(dataset) // (
            wandb.config.batch_size * wandb.config.gradient_accumulation_steps
        )
        scheduler = get_scheduler(
            wandb.config.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=wandb.config.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # create a ppo trainer config, model, ref_model, tokenizer,
    # dataset=dataset, data_collator=collator)
    # the dataset and collator get bundled in a data loader together.
    print(
        "The language model has property peft?"
        f" {getattr(language_model, 'is_peft_model', False)}"
    )
    ppo_trainer = PPOTrainer(
        ppo_config,
        language_model,
        model_ref,
        language_tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    # This pipelinle is for the reward model
    # TODO: look at superhf code to load the reward model and then
    reward_model_tokenizer = load_reward_tokenizer(reward_model_name)
    reward_model, reward_model_pipe = load_reward_model(
        reward_model_name, device=device
    )
    if "llama" in reward_model_name or "alpaca" in reward_model_name:
        assert device == 0, "LLAMA and Alpaca models only work on GPU 0"
        if not isinstance(reward_model, str):
            assert (
                reward_model.device == 0
            ), "LLAMA and Alpaca models only work on GPU 0"
    # reward_model.to(device) # TODO Might be redundant
    # reward_model_pipe.tokenizer.pad_token_id = reward_model.config.eos_token_id
    print(f"The device is {device}, with reward model placed on device.")
    print_gpu_utilization()

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)

    # output_min_length = 4
    # output_max_length = 16
    # output_length_sampler = LengthSampler(output_min_length, output_max_length)
    language_tokenizer.pad_token = language_tokenizer.eos_token
    for epoch, batch in tqdm(
        enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)
    ):
        tqdm.write(f"Epoch {epoch}")
        print_gpu_utilization()

        query_tensors = [
            language_tokenizer(q, return_tensors="pt")["input_ids"].squeeze().to(device)
            for q in batch["query"]
        ]

        # Get response from the model
        response_tensors = []
        start_time = time.time()
        for query in query_tensors:
            # gen_len = output_length_sampler()
            # generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze())
        batch["response"] = trim_generations(
            [language_tokenizer.decode(r.squeeze()) for r in response_tensors]
        )
        tqdm.write(f"Finished generating responses. took {time.time() - start_time}")
        print_gpu_utilization()

        # Compute sentiment score
        start_time = time.time()
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        if reward_model_pipe is not None:
            pipe_outputs = reward_model_pipe(texts, **reward_model_kwargs)
            original_rewards = [
                torch.tensor(output[0]["score"]) for output in pipe_outputs
            ]
            rewards = torch.stack(original_rewards)
        else:
            rewards = score_completions(
                reward_model=reward_model,
                reward_model_tokenizer=reward_model_tokenizer,
                completions=texts,
            )
            original_rewards = [datum.item() for datum in rewards]
        print(f"Time to score completions: {time.time() - start_time}")

        # add the negative of the mean to every reward so that the mean is zero
        # and then add reward_mean to every reward so that the mean is reward_mean
        if normalize_reward:
            curr_mean_reward = torch.mean(rewards)
            rewards = [r - curr_mean_reward + reward_mean for r in rewards]

        # Run PPO step
        start_time = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, original_rewards)

        tqdm.write(f"Finished PPO step. Took {time.time() - start_time} seconds.")
        print_gpu_utilization()

        if len(hub_repo_id) > 0 and (
            epoch == len(ppo_trainer.dataloader) - 1
            or (epoch > 0 and epoch % save_every == 0)
        ):
            tqdm.write(
                f"Pushing model and tokenizer to the Hub! Location: {hub_repo_id}"
            )
            ppo_trainer.model.push_to_hub(
                repo_id=hub_repo_id,
                commit_message=f"Upload model from batch {epoch}, run {run_name}",
            )
            ppo_trainer.tokenizer.push_to_hub(
                repo_id=hub_repo_id,
                commit_message=f"Upload tokenizer from batch {epoch}, run {run_name}",
            )
    # Explicit finish to avoid wandb hanging
    wandb.alert(
        title="FINISHED SuperHF run!", text="FINISHED SuperHF run! <@W011580CVSN>"
    )  # Peter's member ID
    wandb.finish()


def trim_generations(raw_completions: list[str]) -> list[str]:
    """
    Trim the generated completions to remove the prompt and the model's
    repetition of the prompt. Copied from SuperHF code
    """
    original_length = len(raw_completions)
    prompts_and_completions = [
        separate_prompt_from_completion(completion) for completion in raw_completions
    ]
    trimmed_completions: list[str] = []
    model_completion_lengths: list[int] = []
    for prompt, completion in prompts_and_completions:
        if completion == "":
            print("WARNING: Completion is empty.")
        stripped_completion = re.split(
            constants.PROMPT_DELIMITER_REGEX_MEDIUM, completion, maxsplit=1
        )[0].strip()
        if stripped_completion == "":
            print("WARNING: Stripped completion is empty.")
        trimmed_completions.append(prompt + " " + stripped_completion)
        model_completion_lengths.append(len(stripped_completion))

    assert len(trimmed_completions) == original_length, (
        "The number of Trimmed completions should be the same as the number of original"
        " completions."
    )
    return trimmed_completions


if __name__ == "__main__":
    args = parse_args()
    if args.sweep_id != "":
        # Run sweeps
        # with open(args.sweep, encoding="utf-8") as f:
        #     sweep_params = yaml.load(f, Loader=yaml.FullLoader)
        wandb.agent(
            args.sweep_id,
            function=lambda: main(args),
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            count=1,
        )
    else:
        main(args)
