"""
Mock classes for testing.
"""

from typing import Optional, Any

import torch
from transformers import GenerationMixin, BatchEncoding


class MockLanguageModel(GenerationMixin):
    """Mocks a HuggingFace AutoModelForCausalLM class."""

    def generate(
        self,
        inputs: BatchEncoding,
        **_: Any,
    ) -> Any:
        """Mocks the generate method for sequence generation."""
        return inputs.input_ids

    def __call__(
        self,
        inputs: BatchEncoding = None,
        **_: Any,
    ) -> Any:
        """Mocks the __call__ method for fine-tuning."""
        return None

    def train(self) -> None:
        """Mocks enabling training mode."""


class MockRewardModel(torch.nn.Module):
    """Mocks a HuggingFace AutoModelForSequenceClassification class."""

    def __call__(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        **_: Any,
    ) -> Any:
        """Mocks the __call__ method for sequence classification."""
        # Return a random float for each input in the batch
        if input_ids is None:
            return torch.randn(1)
        return torch.randn(input_ids.shape[0])

    def forward(
        self,
        **kwargs: Any,
    ) -> Any:
        """Mocks the forward method for sequence classification."""
        return self(**kwargs)
