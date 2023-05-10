"""Functions for efficiently loading the models, taking advantage of LoRA."""

import gc
import os
import time
from typing import Any, Optional

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
import numpy as np
from peft import PeftConfig, PeftModel
from peft.mapping import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PEFT_TYPE_TO_CONFIG_MAPPING,
)
from peft.tuners.lora import LoraLayer
from peft.utils import (
    WEIGHTS_NAME,
    PeftType,
    set_peft_model_state_dict,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    LlamaTokenizer,
)
from tqdm import tqdm

from superhf.utils import print_memory_utilization
from superhf.mocking import MockLanguageModel, MockRewardModel
from reward_modelling.reward_model import RewardModel


def load_eval_model_and_tokenizer(
    model_path: str,
    prev_model: Optional[torch.nn.Module] = None,
    prev_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    verbose: bool = False,
    revision: Optional[str] = None,
    model_type: Optional[str] = "language",
    tokenizer_padding_side: Optional[str] = None,
    **model_kwargs: Any,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """
    Efficiently load a new model and tokenizer, possibly reusing weights from the base model.

    If prev_model is none, simply load the model from the path.

    Otherwise, this assumes we're loading LoRA weights for a model: Check if the LoRA weights for
    model_path will match the prev_model. If so, it will replace any LoRA adapters on prev_model
    with the new adapters. If not, it will reload a new base model and then add the LoRA adapters.

    A similar process happens with loading the appropriate tokenizer.

    Args:
        model_path: The path to the model to load.
        prev_model: The previous model to reuse weights from.
        prev_tokenizer: The previous tokenizer to reuse weights from.
        verbose: Whether to print out progress.
        revision: The hugging face branch of the model to load.
        model_type: The type of model to load, either "language" or "reward".
        **model_kwargs: Any additional kwargs to pass to the model such as bfloat16.
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-locals

    assert (prev_model is None and prev_tokenizer is None) or (
        prev_model is not None and prev_tokenizer is not None
    ), "Either both prev_model and prev_tokenizer should be None, or neither should."

    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path
        tokenizer_path = base_model_path
    except ValueError:
        # Probably isn't a PEFT model, so just load it from scratch
        peft_config = None
        base_model_path = model_path
        tokenizer_path = model_path

    if tokenizer_path == "mock":
        tokenizer_path = "gpt2"

    if (
        prev_model is not None
        and peft_config is not None
        and peft_config.base_model_name_or_path == prev_model.config._name_or_path  # type: ignore
    ):
        # We know we have the right base model, reuse it.
        model = prev_model
        if isinstance(model, PeftModel):
            model = model.get_base_model()
        tokenizer = prev_tokenizer
    else:
        # If we don't have a previous model, or it's different from the one we want to load, reload
        if prev_model is not None:
            # Free memory first
            tqdm.write(f"Freeing memory for {model_path}.")
            print_memory_utilization()
            prev_model.cpu()
            del prev_model
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
            print_memory_utilization()

        if verbose:
            tqdm.write(f"Loading model and tokenizer from scratch for {model_path}.")
        if "reward" in model_type:
            # load a reward model
            tqdm.write("Loading a reward model")
            if base_model_path == "mock":
                model = MockRewardModel()
            elif "rm_combined" in base_model_path or "oliversssf2" in base_model_path:
                model = RewardModel.from_pretrained(
                    base_model_path,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                )
                tokenizer_path = "EleutherAI/gpt-neo-1.3B"
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_path,
                    low_cpu_mem_usage=True,
                )
        else:
            # load a language model that is not a peft model
            model = (
                MockLanguageModel()
                if base_model_path == "mock"
                else AutoModelForCausalLM.from_pretrained(
                    base_model_path, **model_kwargs
                )
            )

        # Fix for misnamed class in the NLP Cluster's Alpaca tokenizer config
        tokenizer_class = (
            LlamaTokenizer
            if "llama" in tokenizer_path or "alpaca" in tokenizer_path
            else AutoTokenizer
        )
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            if verbose:
                print(f"Added pad token to tokenizer for {model_path}.")

    if peft_config is not None:
        assert peft_config.base_model_name_or_path == model.config._name_or_path  # type: ignore

        if verbose:
            tqdm.write(f"Loading PEFT adapters for {model_path}.")
        model = peftmodel_from_pretrained_revision(
            model, model_path, revision=revision, **model_kwargs
        )

    # Set eval mode
    # HACK for peft==0.2.0: manually disable merge_weights. Otherwise, .eval() will error.
    for layer in model.modules():
        if isinstance(layer, LoraLayer):
            layer.merge_weights = False
    model.eval()

    # Set device
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    prev_padding_side = tokenizer.padding_side
    if (
        tokenizer_padding_side is not None
        and tokenizer_padding_side != prev_padding_side
    ):
        tqdm.write(
            f"Changing padding side from {prev_padding_side} to"
            f" {tokenizer_padding_side}."
        )
        tokenizer.padding_side = tokenizer_padding_side
    return model, tokenizer


def peftmodel_from_pretrained_revision(
    model: Any, model_id: str, **kwargs: Any
) -> PeftModel:
    r"""
    Copied from PeftModel.from_pretrained() but with support for revisions.
    """

    # load the config
    config = PEFT_TYPE_TO_CONFIG_MAPPING[
        PeftConfig.from_pretrained(model_id).peft_type
    ].from_pretrained(model_id)

    if getattr(model, "hf_device_map", None) is not None:
        PeftModel.remove_hook_from_submodules(model)

    # pylint: disable=consider-iterating-dictionary
    if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
        model = PeftModel(model, config)
    else:
        model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config)

    # load weights if any
    if os.path.exists(os.path.join(model_id, WEIGHTS_NAME)):
        filename = os.path.join(model_id, WEIGHTS_NAME)
    else:
        try:
            revision = kwargs.get("revision", None)
            filename = hf_hub_download(model_id, WEIGHTS_NAME, revision=revision)

        except ValueError as exc:
            raise ValueError(
                f"Can't find weights for {model_id} in {model_id} or in the Hugging"
                f" Face Hub. Please check that the file {WEIGHTS_NAME} is present at"
                f" {model_id}."
            ) from exc

    adapters_weights = torch.load(
        filename,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # load the weights into the model
    model = set_peft_model_state_dict(model, adapters_weights)
    if getattr(model, "hf_device_map", None) is not None:
        device_map = kwargs.get("device_map", "auto")
        max_memory = kwargs.get("max_memory", None)
        no_split_module_classes = (
            model._no_split_modules  # pylint: disable=protected-access
        )
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == "balanced_low_0"),
            )
        if isinstance(device_map, str):
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
            )
        model = dispatch_model(model, device_map=device_map)
        hook = AlignDevicesHook(io_same_device=True)
        if model.peft_config.peft_type == PeftType.LORA:
            add_hook_to_module(model.base_model.model, hook)
        else:
            remove_hook_from_submodules(model.prompt_encoder)
            add_hook_to_module(model.base_model, hook)
    return model


def run_tests() -> None:
    """Test the above function."""
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    verbose = True
    test_prompt = "Sphinx of black quartz, judge my vow!"

    model: Optional[torch.nn.Module] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None

    tqdm.write("\n### Testing loading pretrained models alone.")
    model_paths = [
        "EleutherAI/pythia-70m-deduped",
        "facebook/opt-125m",
    ]
    for model_path in tqdm(model_paths):
        model, tokenizer = load_eval_model_and_tokenizer(model_path, verbose=verbose)
        assert model is not None
        assert tokenizer is not None

    tqdm.write("\n### Testing loading adapter models alone.")
    model_paths = [
        "gmukobi/model-test-pythia-70m-A",
        "gmukobi/model-test-pythia-70m-B",
        "gmukobi/model-test-opt-125m-C",
    ]
    for model_path in tqdm(model_paths):
        model, tokenizer = load_eval_model_and_tokenizer(model_path, verbose=verbose)
        assert model is not None
        assert tokenizer is not None
        assert isinstance(model, PeftModel)

    tqdm.write("\n### Testing reusing the base model for the best case order.")
    model_paths = [
        "EleutherAI/pythia-70m-deduped",
        "gmukobi/model-test-pythia-70m-A",
        "gmukobi/model-test-pythia-70m-B",
        "facebook/opt-125m",
        "gmukobi/model-test-opt-125m-C",
    ]
    model = None
    tokenizer = None
    outputs = []
    for model_path in tqdm(model_paths):
        model, tokenizer = load_eval_model_and_tokenizer(
            model_path, model, tokenizer, verbose=verbose
        )
        assert model is not None
        assert tokenizer is not None
        outputs.append(
            model(**tokenizer(test_prompt, return_tensors="pt").to(model.device))
            .logits.detach()
            .cpu()
            .numpy()
        )
    # Check that all the outputs are distinct
    for i, output in enumerate(outputs):
        for j, other_output in enumerate(outputs):
            if i != j:
                assert output.shape != other_output.shape or not np.allclose(
                    output, other_output
                ), f"Outputs {i} and {j} should be distinct, but are not."

    tqdm.write("\n### Testing reusing the base model for a bad order.")
    model_paths = [
        "gmukobi/model-test-opt-125m-C",
        "gmukobi/model-test-pythia-70m-B",
        "facebook/opt-125m",
        "gmukobi/model-test-pythia-70m-A",
        "EleutherAI/pythia-70m-deduped",
    ]
    model = None
    tokenizer = None
    outputs = []
    for model_path in tqdm(model_paths):
        model, tokenizer = load_eval_model_and_tokenizer(
            model_path, model, tokenizer, verbose=verbose
        )
        assert model is not None
        assert tokenizer is not None
        outputs.append(
            model(**tokenizer(test_prompt, return_tensors="pt").to(model.device))
            .logits.detach()
            .cpu()
            .numpy()
        )
    # Check that all the outputs are distinct
    for i, output in enumerate(outputs):
        for j, other_output in enumerate(outputs):
            if i != j:
                assert output.shape != other_output.shape or not np.allclose(
                    output, other_output
                ), f"Outputs {i} and {j} should be distinct, but are not."

    tqdm.write("\n### Testing getting different revisions of the same model.")
    model_paths = [
        "gmukobi/model-test-pythia-70m-A",
        "gmukobi/model-test-pythia-70m-B",
        "gmukobi/model-test-opt-125m-C",
    ]
    revisions = [
        "main",
        "step-0001",
        "step-0002",
        "step-0004",
        "step-0007",
        "step-0008",
        "step-0011",
    ]
    for model_path in model_paths:
        for revision in tqdm(revisions):
            previous_model_weights = None
            model, tokenizer = load_eval_model_and_tokenizer(
                model_path, model, tokenizer, verbose=verbose, revision=revision
            )
            assert model is not None
            assert tokenizer is not None
            if previous_model_weights is not None:
                assert not np.allclose(
                    previous_model_weights,
                    next(model.parameters()).detach().cpu().numpy(),
                ), "Model weights should be different, but are not."

    if torch.cuda.is_available():
        tqdm.write("\n### Testing that the model is on GPU.")
        model = None
        tokenizer = None
        model_paths = [
            "EleutherAI/pythia-70m-deduped",
            "gmukobi/model-test-pythia-70m-A",
        ]
        for model_path in tqdm(model_paths):
            model, tokenizer = load_eval_model_and_tokenizer(
                model_path, model, tokenizer, verbose=verbose
            )
            assert model is not None
            assert tokenizer is not None
            assert next(model.parameters()).is_cuda
            assert "cuda" in str(model.device.type)  # TODO

    tqdm.write("\n### All tests passed! 😊 🚀 ✅")


if __name__ == "__main__":
    run_tests()
