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


class MockRewardModel(torch.nn.Module):
    """Mocks a HuggingFace AutoModelForSequenceClassification class."""

    def __call__(
        self,
        inputs: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Any:
        """Mocks the __call__ method for sequence classification."""
        # Return a random float for each input in the batch
        if inputs is None:
            return torch.randn(1)
        return torch.randn(inputs.shape[0])

    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Any:
        """Mocks the forward method for sequence classification."""
        return self(inputs)
