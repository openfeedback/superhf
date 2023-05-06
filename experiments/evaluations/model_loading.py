"""Functions for efficiently loading the models, taking advantage of LoRA."""

from typing import Any, Optional

import numpy as np
from peft import PeftConfig, PeftModel
from peft.tuners.lora import LoraLayer
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    LlamaTokenizer,
)
from tqdm import tqdm


def load_eval_model_and_tokenizer(
    model_path: str,
    prev_model: Optional[torch.nn.Module] = None,
    prev_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    verbose: bool = False,
    revision: str = "main",
    **model_kwargs: Any,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """
    Efficiently load a new model and tokenizer, possibly reusing weights from the base model.

    If prev_model is none, simply load the model from the path.

    Otherwise, this assumes we're loading LoRA weights for a model: Check if the LoRA weights for
    model_path will match the prev_model. If so, it will replace any LoRA adapters on prev_model
    with the new adapters. If not, it will reload a new base model and then add the LoRA adapters.

    A similar process happens with loading the appropriate tokenizer.
    """
    # pylint: disable=protected-access

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
        if verbose:
            tqdm.write(f"Loading model and tokenizer from scratch for {model_path}.")
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
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
        model = PeftModel.from_pretrained(
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

    return model, tokenizer


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
