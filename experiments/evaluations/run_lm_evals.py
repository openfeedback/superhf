"""Evaluate general capabilities and some safety metrics using lm-evals."""

import argparse

from lm_eval.base import CacheHook
from lm_eval.models.gpt2 import HFLM

# import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from model_loading import load_eval_model_and_tokenizer


class CustomEvalModel(HFLM):
    """Wrapper for a given CausalLM model and tokenizer."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 1,
    ):
        """Overrides HFLM.__init__."""
        # pylint: disable=super-init-not-called
        # From the LM.__init__ since we don't want to call super().__init__:
        self.cache_hook = CacheHook(None)

        assert isinstance(model, PreTrainedModel)
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        assert model.training is False

        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size


def run_evaluations(model_paths: list[str], eval_names: list[str]) -> None:
    """Run the evaluations for the given models and evaluation names."""
    model = None
    tokenizer = None
    for model_path in tqdm(model_paths, desc="Models"):
        print(
            "\n" + "-" * 66 + f"\n\n###### Running evaluations for {model_path} ######"
        )
        model, tokenizer = load_eval_model_and_tokenizer(
            model_path, model, tokenizer, verbose=True
        )
        # eval_model = CustomEvalModel(model, tokenizer)
        for eval_name in tqdm(eval_names, desc="Evaluations"):
            print(f"\n# Running evaluation {eval_name} #")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--evals", type=str, nargs="+", required=True)
    args = parser.parse_args()
    run_evaluations(args.models, args.evals)
