"""
Script for running RLHF to compare with SuperHF.

Utilizes the PPO trainer from TRL to train the language model.
If using our custom reward model, calls the model's generate function dirctly.

Implements LoRA based on this guide -
https://github.com/lvwerra/trl/blob/52fecee8839ad826ad1e6c83a95c99a4116e98d2 \
/examples/stack_llama/scripts/rl_training.py

Example usage:
    python run_rlhf.py \
        --config configs/rlhf_v1.yaml \
        --notes "Testing RLHF with TRL" \
        --sweep_id xxxxx \
        --sweep_param_name kl \
"""

import os
import random
import re
from typing import Optional, TypeVar, List, Union, Tuple, Dict, Any
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
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from datasets import Dataset

from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)

# from trl.core import LengthSampler
import wandb

from superhf import constants
from superhf.data import get_superhf_prompts
from superhf.utils import set_seed, print_memory_utilization
from superhf.mocking import MockRewardModel
from superhf.constants import PROMPT_DELIMITER
from reward_modelling.reward_model import RewardModel

T = TypeVar("T")

WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "rlhf-trl-v1"
MAX_OOM_ALLOWED = 16

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"  # all these set acording to alpaca_farm
DEFAULT_UNK_TOKEN = "<unk>"


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
        # Get file path relative to this file
        default=os.path.join(os.path.dirname(__file__), "configs", "rlhf_config.yaml"),
        metadata={"help": "The name of the Weights and Biases config to use."},
    )
    notes: Optional[str] = field(
        default="", metadata={"help": "notes to add to the wandb run"}
    )
    sweep_id: Optional[str] = field(
        default="", metadata={"help": "sweep id to use to for a sweep"}
    )
    sweep_param_name: Optional[str] = field(
        default="", metadata={"help": "sweep parameter name"}
    )
    # Extra arguments
    extra_args: dict = field(default_factory=dict)


def parse_args():
    """
    This function parses the arguments passed to the script. It utilizes the
    HfArgumentParser class from the transformers library.
    """
    parser = HfArgumentParser(ScriptArguments)
    # pylint: disable=unbalanced-tuple-unpacking
    script_args, unknown_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    script_args.extra_args = unknown_args
    return script_args


def build_dataset(
    prompt_dataset_names,
    num_prompts,
    max_prompt_char_length,
    seed,
):
    """
    Currentlty we don't use the tokenizer becauses the internal trainer
    for some reason throws away the tokenized exmples.

    Args:
        prompt_dataset_names: list of dataset names to use for prompts
        num_prompts: number of prompts to use
        max_prompt_char_length: max length of prompt in characters
        seed: random seed

    Returns:
        a pytorch dataset that implements the __getitem__ and __len__ methods.
        PPO trainer converts this to a pytorch dataloader.
        torch.utils.data.Dataset
    """
    # Configure seed
    set_seed(seed)

    # Get the prompt dataset
    prompts: list[str] = []
    num_datasets = len(prompt_dataset_names)
    num_prompts_per_dataset = num_prompts // num_datasets if num_prompts > 0 else 0
    for dataset_name in prompt_dataset_names:
        new_prompts = get_superhf_prompts(
            dataset_name, max_length_chars=max_prompt_char_length
        )
        total_loaded = len(new_prompts)
        if num_prompts_per_dataset != 0:
            new_prompts = new_prompts[:num_prompts_per_dataset]
        total_filtered = len(new_prompts)
        print(f"Loaded {total_filtered}/{total_loaded} prompts from {dataset_name}.")
        prompts.extend(new_prompts)
    random.shuffle(prompts)

    dataset = Dataset.from_dict({"query": prompts})
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

    Args:
        reward_model_name: The name of the reward model to load.
        device: The device to load the reward model onto.

    Returns:
        A tuple of the reward model and the reward model pipeline.
        The pipeline is either None if using oliver's rm, otherwise returns a pipeline object.
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
            device=device,
        )
    return reward_model_train, reward_model_pipe


def load_language_model(language_model_name: str) -> PreTrainedModel:
    """Load the language model."""
    try:
        peft_config = PeftConfig.from_pretrained(language_model_name)
        base_model_path = peft_config.base_model_name_or_path
    except ValueError:
        # Probably isn't a PEFT model, so just load it from scratch
        peft_config = None
        base_model_path = language_model_name
    language_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
    )
    if peft_config is not None:
        # Already has pretrained adapter weights, load them
        # pylint: disable=protected-access
        assert (
            peft_config.base_model_name_or_path == language_model.config._name_or_path
        )
        language_model = PeftModel.from_pretrained(
            language_model,
            language_model_name,
        )

    elif wandb.config.lora_r != 0 and wandb.config.lora_alpha != 0:
        # Set up low-rank adapters (LoRA)
        lora_config = LoraConfig(
            r=wandb.config.lora_r,
            lora_alpha=wandb.config.lora_alpha,
            lora_dropout=wandb.config.lora_dropout,
            target_modules=wandb.config.lora_target_modules,
            task_type="CAUSAL_LM",
            fan_in_fan_out=False,
        )
        language_model = get_peft_model(language_model, lora_config)
        language_model.print_trainable_parameters()

    print(f"Instantiated language model: {language_model_name}")
    return language_model


def get_configs():
    """
    Organizes all the configs into one place, and returns all of them.
    """
    accelerator_kwargs = {}
    if torch.backends.mps.is_available():
        print(
            "Detected mps. Forcing CPU instead because older macbook pros don't have"
            " enough memmory for trainiing LLMs."
        )
        os.environ["ACCELERATE_USE_CPU"] = "1"
        print("Set ACCELERATE_USE_CPU=1")
    ppo_config = PPOConfig(
        model_name=wandb.config.model_name,
        steps=20000,
        learning_rate=wandb.config.learning_rate,
        adap_kl_ctrl=False,
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
        accelerator_kwargs=accelerator_kwargs,
        tracker_project_name="trl",
        max_grad_norm=None,
        seed=wandb.config.seed,
        optimize_cuda_cache=False,
        whiten_rewards=wandb.config.whiten_rewards,
    )

    assert ppo_config.mini_batch_size <= ppo_config.batch_size

    reward_model_kwargs = {
        "function_to_apply": "none",
        "batch_size": wandb.config.batch_size,
        "penalize_bad_endings": wandb.config.penalize_bad_endings,
    }  # arguments for the reward pipeline.

    language_tokenizer = None
    if "llama" in ppo_config.model_name or "alpaca" in ppo_config.model_name:
        # Fix for misnamed class in the NLP Cluster's Alpaca tokenizer config
        language_tokenizer = LlamaTokenizer.from_pretrained(
            ppo_config.model_name, padding_side="left"
        )
        language_tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        language_tokenizer = AutoTokenizer.from_pretrained(
            ppo_config.model_name, padding_side="left"
        )
        language_tokenizer.pad_token = language_tokenizer.eos_token
    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    extra_generation_args = wandb.config.generation_kwargs
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": wandb.config.top_p,
        "do_sample": True,
        "pad_token_id": language_tokenizer.pad_token_id,
        "max_new_tokens": wandb.config.max_new_tokens,
    }
    # loop over extra_generation_args two at a time
    for i in range(len(extra_generation_args) // 2):
        if extra_generation_args[i] == "pad_token_id":
            if extra_generation_args[i + 1] == "eos_token_id":
                generation_kwargs[extra_generation_args[i]] = (
                    language_tokenizer.eos_token_id
                )
            elif extra_generation_args[i + 1] == "bos_token_id":
                generation_kwargs[extra_generation_args[i]] = (
                    language_tokenizer.bos_token_id
                )
            elif extra_generation_args[i + 1] == "unk_token_id":
                generation_kwargs[extra_generation_args[i]] = (
                    language_tokenizer.unk_token_id
                )
            elif extra_generation_args[i + 1] == "pad_token_id":
                generation_kwargs[extra_generation_args[i]] = (
                    language_tokenizer.pad_token_id
                )
            continue
        generation_kwargs[extra_generation_args[i]] = extra_generation_args[i + 1]

    return (
        ppo_config,
        reward_model_kwargs,
        language_tokenizer,
        generation_kwargs,
    )


def score_completions(
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
    with torch.no_grad():
        tqdm.write("Scoring completions.")
        scores = reward_model(**completions_tokenized, **reward_model_kwargs)
    if not isinstance(scores, torch.Tensor):
        scores: torch.Tensor = scores.logits
    return scores


def consider_pushing_to_hub(
    script_args,
    ppo_config,
    hub_repo_id,
    ppo_trainer,
    hf_api,
    epoch,
    run_name,
    save_every,
    extra_push_to_hub,
):
    """
    Considers pushing the model and tokenizer to the Hub.
    Args:
        script_args: The arguments to the script.
        ppo_config: The PPO config.
        hub_repo_id: The repo ID to push to.
        ppo_trainer: The PPO trainer.
        hf_api: The HuggingFace API.
        epoch: The current epoch.
        run_name: The name of the run.
        save_every: How often to save.
        extra_push_to_hub: Extra epochs to push to the Hub.
    """
    # pylint: disable=too-many-locals
    should_push_to_hub = False
    if len(hub_repo_id) == 0:
        return
    if (
        epoch == len(ppo_trainer.dataloader) - 1
        or (epoch % save_every == 0 and epoch != 0)
        or epoch in extra_push_to_hub
    ):
        should_push_to_hub = True
    if not should_push_to_hub:
        return
    tqdm.write(f"Pushing model and tokenizer to the Hub! Location: {hub_repo_id}")
    repo_name = hub_repo_id
    if script_args.sweep_param_name is not None and script_args.sweep_param_name != "":
        assert (
            script_args.sweep_param_name != "pythia"
            or "pythia" in ppo_config.model_name
        ), "Must use a pythia model to add a pythia model size to the repo name."
        param_name_to_value = {
            # get the actual params
            "kl": ppo_config.init_kl_coef,
            "lr": ppo_config.learning_rate,
            "batch": ppo_config.batch_size * ppo_config.gradient_accumulation_steps,
            "seed": ppo_config.seed,
        }
        if "pythia" == script_args.sweep_param_name:
            param_name_to_value["pythia"] = (ppo_config.model_name.split("-")[1],)
        # get the param value specified by the sweep
        param_value = param_name_to_value[script_args.sweep_param_name]
        repo_name += f"-{script_args.sweep_param_name}-{param_value}"

    tqdm.write(
        str(
            ppo_trainer.model.push_to_hub(
                repo_id=repo_name,
                commit_message=f"Upload model from batch {epoch}, run {run_name}",
            )
        )
    )
    tqdm.write(
        str(
            ppo_trainer.tokenizer.push_to_hub(
                repo_id=repo_name,
                commit_message=f"Upload tokenizer from batch {epoch}, run {run_name}",
            )
        )
    )

    # Create a new branch with the superbatch index as the name
    hf_username = hf_api.whoami()["name"]
    repo_id = hf_username + "/" + repo_name
    branch = f"step-{epoch:05}"
    try:
        result = hf_api.create_branch(repo_id=repo_id, branch=branch)
    except HfHubHTTPError:
        # Delete the branch first
        hf_api.delete_branch(repo_id=repo_id, branch=branch)
        result = hf_api.create_branch(repo_id=repo_id, branch=branch)
    tqdm.write(str(result))


def parse_type(value):
    """
    Parses a value to the appropriate type.
    """
    if value == "True":
        value = True
    elif value == "False":
        value = False
    elif "." in value and value.replace(".", "").isdigit():
        value = float(value)
    elif value.isdigit():
        value = int(value)
    return value


def main(script_args: ScriptArguments):
    """
    Main function. Downloads the prompt dataset, reads the config file, and then
    executes the main PPO training loop. For the gradient updates, it uses the
    trl library's PPOTrainer class.
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

    # Process any extra arguments, converting values to appropriate types
    extra_args_dict = {}
    for arg in script_args.extra_args:
        value: Any
        key, value = arg.split("=")

        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].split(",")
            value = [parse_type(v) for v in value]
        else:
            value = parse_type(value)
        extra_args_dict[key] = value
        if isinstance(value, List):
            print(f"Value is a list for key {key}")
    wandb.config.update(extra_args_dict, allow_val_change=True)

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
    extra_push_to_hub = wandb.config.extra_push_to_hub
    reward_mean = wandb.config.reward_mean
    offset_reward = wandb.config.offset_reward
    reward_model_name = wandb.config.reward_model_name
    test_reward_model_name = wandb.config.test_reward_model_name
    conversation_prompt = wandb.config.conversation_prompt
    try:
        trim_generations_or_not = (
            wandb.config.trim_generations_or_not
        )  # decides whether to filter generations or not
    except AttributeError:
        print("Warning: trim_generations_or_not not set. Defaulting to True.")
        trim_generations_or_not = True
    (
        ppo_config,
        reward_model_kwargs,
        language_tokenizer,
        generation_kwargs,
    ) = get_configs()

    hf_api = None
    if wandb.config.hub_repo_id != "":
        hf_api = HfApi()
        assert hf_api.whoami(), (
            "Must be logged in to the Hugging Face Hub to push models to the hub. "
            "Run `huggingface-cli login` to log in."
        )

    language_model = load_language_model(ppo_config.model_name)
    # language_model = AutoModelForCausalLM.from_pretrained(ppo_config.model_name)
    model_ref = None
    if not hasattr(language_model, "disable_adapter"):
        # Copy the entire model in order to calculate the KL divergence
        # model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        #     ppo_config.model_name
        # )
        model_ref = load_language_model(ppo_config.model_name)
        model_ref = AutoModelForCausalLMWithValueHead(model_ref)
    language_model = AutoModelForCausalLMWithValueHead.from_pretrained(language_model)

    # set seed before initializing value head for deterministic eval
    dataset = build_dataset(
        prompt_dataset_names=wandb.config.prompt_dataset_names,
        num_prompts=wandb.config.num_prompts,
        max_prompt_char_length=wandb.config.max_prompt_char_length,
        seed=wandb.config.seed,
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
    print(f"Using device {device}.")
    # This pipeline is for the reward model
    reward_model_tokenizer = load_reward_tokenizer(reward_model_name)
    reward_model, reward_model_pipe = load_reward_model(
        reward_model_name, device=device
    )
    if test_reward_model_name != "":
        test_reward_model, _ = load_reward_model(test_reward_model_name, device=device)

    if not isinstance(reward_model, str):
        reward_model = ppo_trainer.accelerator.prepare(reward_model)
    print(
        f"Instantiated reward model: {reward_model_name} on device"
        f" {reward_model.device}"
    )
    if test_reward_model_name != "":
        test_reward_model = ppo_trainer.accelerator.prepare(test_reward_model)
        print(
            f"Instantiated test reward model: {test_reward_model_name} on device"
            f" {test_reward_model.device}"
        )
    if "llama" in reward_model_name or "alpaca" in reward_model_name:
        assert device == 0, "LLAMA and Alpaca models only work on GPU 0"
        if not isinstance(reward_model, str):
            assert (
                reward_model.device == 0
            ), "LLAMA and Alpaca models only work on GPU 0"
    print(f"The device is {device}, with reward model(s) placed on device.")
    print_memory_utilization()

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)

    # output_min_length = 4
    # output_max_length = 16
    # output_length_sampler = LengthSampler(output_min_length, output_max_length)
    oom_count = 0
    language_tokenizer.pad_token = language_tokenizer.eos_token
    for epoch, batch in tqdm(
        enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)
    ):
        # main RLHF Loop
        try:
            tqdm.write(f"Epoch {epoch}")
            print_memory_utilization()

            # prepends the conversatioin prompt before tokenizing
            query_tensors = [
                language_tokenizer(conversation_prompt + q, return_tensors="pt")[
                    "input_ids"
                ]
                .squeeze()
                .to(device)
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
            batch["response"] = [
                language_tokenizer.decode(r.squeeze()) for r in response_tensors
            ]
            if trim_generations_or_not:
                batch["response"] = trim_generations(batch["response"])
            tqdm.write(
                f"Finished generating responses. took {time.time() - start_time}"
            )
            print_memory_utilization()

            # Compute sentiment score
            start_time = time.time()
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            test_rewards = []
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
                if test_reward_model_name != "":
                    test_rewards = score_completions(
                        reward_model=test_reward_model,
                        reward_model_tokenizer=reward_model_tokenizer,
                        completions=texts,
                    )
                original_rewards = [datum.item() for datum in rewards]
            tqdm.write(f"Time to score completions: {time.time() - start_time}")

            # add the negative of the mean to every reward so that the mean is zero
            # and then add reward_mean to every reward so that the mean is reward_mean
            if normalize_reward:
                curr_mean_reward = torch.mean(rewards)
                rewards = [r - curr_mean_reward + reward_mean for r in rewards]
            else:
                rewards = [r + offset_reward for r in rewards]

            # Run PPO step
            start_time = time.time()
            tqdm.write("Running PPO step")
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, original_rewards, test_rewards)

            tqdm.write(f"Finished PPO step. Took {time.time() - start_time} seconds.")
            print_memory_utilization()
        # catch cuda OOM exception
        except RuntimeError as exc:
            oom_count += 1
            print(exc)
            print(
                "⚠️ WARNING: Error during this training iteration. Total OOM count:"
                f" {oom_count}/{MAX_OOM_ALLOWED}"
            )
            if oom_count > MAX_OOM_ALLOWED:
                raise exc
        consider_pushing_to_hub(
            script_args=script_args,
            ppo_config=ppo_config,
            hub_repo_id=hub_repo_id,
            ppo_trainer=ppo_trainer,
            hf_api=hf_api,
            epoch=epoch,
            run_name=run_name,
            save_every=save_every,
            extra_push_to_hub=extra_push_to_hub,
        )

    # Explicit finish to avoid wandb hanging
    wandb.alert(
        title="FINISHED SuperHF run!", text="FINISHED SuperHF run! <@W011580CVSN>"
    )  # Peter's member ID
    wandb.finish()


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
            conversation turns. Copied from Superhf
    """
    original_length = len(raw_completions)
    prompts_and_completions = [
        separate_prompt_from_completion(completion) for completion in raw_completions
    ]
    trimmed_completions: list[str] = []
    model_completion_lengths: list[int] = []
    for prompt, completion in prompts_and_completions:
        if completion == "":
            tqdm.write("WARNING: Completion is empty.")
        stripped_completion = re.split(
            constants.PROMPT_DELIMITER_REGEX_MEDIUM, completion, maxsplit=1
        )[0].strip()
        if stripped_completion == "":
            tqdm.write("WARNING: Stripped completion is empty.")
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
        COUNT = 1
        if hasattr(args, "count"):
            COUNT = args.count
        # Run sweeps
        # try:
        #     print(f"{args.sweep_id}")
        # except AttributeError:
        #     print("No sweep id found. If doing a sweep, must provide a --sweep-id")
        #     raise AttributeError
        # with open(args.sweep, mode="r", encoding="utf-8") as f:
        #     sweep_params = yaml.load(f, Loader=yaml.FullLoader)
        wandb.agent(
            args.sweep_id,
            function=lambda: main(args),
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            count=COUNT,
        )
    else:
        main(args)
