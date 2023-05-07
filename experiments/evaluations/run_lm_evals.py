"""
Evaluate general capabilities and some safety metrics using lm-evals.

Mostly based off main.py from the lm evals GitHub repo.
https://github.com/EleutherAI/lm-evaluation-harness
"""

import argparse
import fnmatch
import json
import os
from typing import Any

from lm_eval import tasks, evaluator
from lm_eval.base import CacheHook
from lm_eval.models.gpt2 import HFLM

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from model_loading import load_eval_model_and_tokenizer


class CustomEvalModel(HFLM):
    """Wrapper for a given CausalLM model and tokenizer."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 1,
    ):
        """Overrides HFLM.__init__."""
        # pylint: disable=super-init-not-called
        # From the LM.__init__ since we don't want to call super().__init__:
        self.cache_hook = CacheHook(None)

        assert isinstance(model, torch.nn.Module)
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        assert model.training is False

        self.gpt2 = model
        self.tokenizer = tokenizer
        self._device = self.gpt2.device

        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size

    def _model_call(self, inps: torch.Tensor) -> Any:
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            # Had to override to remove hardcoded GPT2 tokenizer size.
            return self.gpt2(inps)[0][:, :, : self.vocab_size]


class MultiChoice:
    """Class for multiple choice arguments."""

    def __init__(self, choices: list[Any]):
        self.choices = choices

    def __contains__(self, values: str) -> bool:
        """Simple wildcard support (linux filename patterns)."""
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self) -> Any:
        for choice in self.choices:
            yield choice


def pattern_match(patterns: list[str], source_list: list[str]) -> list[str]:
    """Returns a list containing all values of the source_list that match at least one pattern."""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


def run_evaluations(args: argparse.Namespace) -> None:
    """Run the evaluations for the given models and evaluation names."""

    assert args.output_folder.strip() != "", "Output folder cannot be empty"

    task_names = pattern_match(args.tasks, tasks.ALL_TASKS)
    print(f"Selected Tasks: {task_names}")
    raw_model = None
    tokenizer = None

    for model_path in tqdm(args.models, desc="Models"):
        tqdm.write(
            "\n" + "-" * 66 + f"\n\n###### Running evaluations for {model_path} ######"
        )
        raw_model, tokenizer = load_eval_model_and_tokenizer(
            model_path, raw_model, tokenizer, verbose=True
        )
        eval_model = CustomEvalModel(raw_model, tokenizer, batch_size=args.batch_size)

        results = evaluator.simple_evaluate(
            model=eval_model,
            # model_args=args.model_args,
            tasks=task_names,
            # num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            # device=args.device,
            no_cache=True,  # Cache doesn't work if model not a str
            # limit=args.limit,
            # description_dict=description_dict,
            # decontamination_ngrams_path=args.decontamination_ngrams_path,
            # check_integrity=args.check_integrity,
        )

        results["config"]["model"] = model_path  # Allow serialization
        dumped = json.dumps(results, indent=2)
        print(dumped)

        output_dir = os.path.join("./eval_results", "lm_evals", args.output_folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(
            output_dir,
            model_path.split("/")[-1] + ".json",
        )
        with open(output_path, "w", encoding="utf8") as file:
            file.write(dumped)

        print(evaluator.make_table(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help=(
            "See all tasks at"
            " https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.3.0/docs/task_table.md"
        ),
        choices=MultiChoice(tasks.ALL_TASKS),
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    run_evaluations(parser.parse_args())
