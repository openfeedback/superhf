"""
Mock classes for testing.
"""

import time
from typing import Any

import numpy as np
import torch
from transformers import (
    GenerationMixin,
    PretrainedConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from superhf.utils import BestOfNWrapper


def make_mock_llama_model() -> Any:
    """
    Creates a mock LlamaForCausalLM model.

    The llama model has one hidden layer, hidden size 2, and 2 attention heads.
    We increase vocab size to accomodate a special pad token.
    """
    config = LlamaConfig(
        vocab_size=32001,
        hidden_size=1,
        intermediate_size=1,
        num_hidden_layers=1,
        num_attention_heads=1,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=32000,  # '[PAD]'
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )

    model = LlamaForCausalLM(config)
    return model


def make_mock_llama_tokenizer() -> Any:
    """
    Creates a mock LlamaTokenizer.
    """
    return LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")


class MockLanguageModel(torch.nn.Module, GenerationMixin):
    """Mocks a HuggingFace AutoModelForCausalLM class."""

    def __init__(self) -> None:
        """Mocks initialization."""
        super().__init__()
        self.device = torch.device("cpu")
        self.start_time = time.time()
        self.config = PretrainedConfig()

    def generate(
        self,
        input_ids: torch.Tensor,
        **_: Any,
    ) -> Any:
        """Mocks the generate method for sequence generation."""
        return input_ids

    def __call__(
        self,
        **_: Any,
    ) -> CausalLMOutputWithCrossAttentions:
        """Mocks the __call__ method for fine-tuning."""
        elapsed_time = time.time() - self.start_time
        output = CausalLMOutputWithCrossAttentions()
        # Make a fake loss curve
        output.loss = torch.tensor(
            np.sin(elapsed_time * 2 * 3.14159 / 10) * 0.5 + np.random.randn() * 0.1
        ).requires_grad_()
        return output

    def train(self, _: bool = False) -> Any:
        """Mocks enabling training mode."""
        return self

    def to(
        self,
        *_: Any,
    ) -> Any:
        """Mocks moving the model to a device."""
        return self

    def parameters(self, _: bool = False) -> Any:
        """Mocks returning the model parameters."""
        return [torch.randn(1).requires_grad_()]

    def forward(
        self,
        **kwargs: Any,
    ) -> Any:
        """Mocks the forward method."""
        return self(**kwargs)


class MockRewardModel(torch.nn.Module):
    """Mocks a HuggingFace AutoModelForSequenceClassification class."""

    def __init__(self) -> None:
        """Mocks initialization."""
        super().__init__()
        self.device = torch.device("cpu")
        self.backbone_model = MockLanguageModel()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        **_: Any,
    ) -> Any:
        """Mocks the __call__ method for sequence classification."""
        output = type("", (), {})()  # TODO use an actual mocking library
        # Return a random float for each input in the batch
        output.logits = torch.randn(input_ids.shape[0])
        return output

    def forward(
        self,
        **kwargs: Any,
    ) -> Any:
        """Mocks the forward method for sequence classification."""
        return self(**kwargs)


def test_llama_mocking():
    """
    Tests and pushes to hugging face hub a mock llama model and tokenizer.
    """
    print("Testing mocking.py")
    print("Testing making a llama model")
    llama_model = make_mock_llama_model()
    print(llama_model)
    # print the number of parameters this model has
    print("Number of parameters:", llama_model.num_parameters())
    # save mock model to the hub
    print("Saving to hub")
    # llama_model.push_to_hub("mock_llama")
    tokenizer = make_mock_llama_tokenizer()
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(tokenizer)
    # pylint: disable=not-callable
    # tokenizer.push_to_hub("mock_llama")


def test_best_of_n():
    """
    Tests best of n wrapper on mock models.
    """
    # load a mock rm
    reward_model = MockRewardModel()
    # load a mock llama model
    model = make_mock_llama_model()
    # load a mock tokenizer
    tokenizer_lm = make_mock_llama_tokenizer()
    # get gpt2 tokenizer
    tokenizer_rw = AutoTokenizer.from_pretrained("gpt2")

    # wrap the model in the best of n wrapper
    model = BestOfNWrapper(model, reward_model, tokenizer_lm, tokenizer_rw)

    # generate some text
    input_ids = tokenizer_lm("Hello, my name is", return_tensors="pt").input_ids
    output = model.generate(input_ids=input_ids, best_of_n=5)
    print(output)
    print(tokenizer_lm.decode(output[0].tolist()))


if __name__ == "__main__":
    test_best_of_n()
