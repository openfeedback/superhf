"""
Mock classes for testing.
"""

import time
from typing import Optional, Any

import numpy as np
import torch
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class MockLanguageModel(GenerationMixin):
    """Mocks a HuggingFace AutoModelForCausalLM class."""

    def __init__(self) -> None:
        """Mocks initialization."""
        super().__init__()
        self.device = torch.device("cpu")
        self.start_time = time.time()

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

    def train(self) -> None:
        """Mocks enabling training mode."""

    def to(self) -> None:  # pylint:disable=invalid-name
        """Mocks moving the model to a device."""

    def parameters(self) -> Any:
        """Mocks returning the model parameters."""
        return [torch.randn(1).requires_grad_()]


class MockRewardModel(torch.nn.Module):
    """Mocks a HuggingFace AutoModelForSequenceClassification class."""

    def __init__(self) -> None:
        """Mocks initialization."""
        super().__init__()
        self.device = torch.device("cpu")

    def __call__(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
