"""
Client code showing how to call the training loop for the iterative version of the SuperHF model.
"""


from transformers import (
    AutoTokenizer,
    # AutoModelForCausalLM,
    # AutoModelForSequenceClassification,
    # GPTNeoForCausalLM,
)

from superhf.data import get_superhf_prompts
from superhf.training import SuperHFTrainingArguments, SuperHFTrainer
from superhf.utils import set_seed
from superhf.mocking import MockLanguageModel, MockRewardModel
from superhf.filtering import CompletionFilterTopK


# TODO use argparse and wandb config for these instead
LANGUAGE_MODEL_NAME = "eleutherai/gpt-neo-1.3B"
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-base"
DEBUG_MAX_PROMPTS = 0
MAX_PROMPT_CHAR_LENGTH = 1024


def main() -> None:
    """
    Instantiate and train the SuperHF model.
    """

    # device = torch.device(
    #     torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    # )

    set_seed(66)

    # Get the prompt dataset
    prompts = get_superhf_prompts("anthropic-red-team")

    # Filter out prompts that are too long
    old_prompt_count = len(prompts)
    prompts = [prompt for prompt in prompts if len(prompt) < MAX_PROMPT_CHAR_LENGTH]
    print(
        f"Filtered {old_prompt_count - len(prompts)} prompts over {MAX_PROMPT_CHAR_LENGTH} chars."
    )

    # Testing: only load the first section of prompts
    if DEBUG_MAX_PROMPTS != 0:
        prompts = prompts[:DEBUG_MAX_PROMPTS]

    print(f"Loaded {len(prompts)} prompts.")

    # Instantiate our language and reward models and tokenizers
    language_model = (
        MockLanguageModel()
    )  # AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_NAME)
    reward_model = (
        MockRewardModel()
    )  # AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)
    language_tokenizer = AutoTokenizer.from_pretrained(
        LANGUAGE_MODEL_NAME, padding_side="left"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)

    # Set our training arguments
    training_args = SuperHFTrainingArguments()
    completion_filter_top_k = 8
    completion_filter = CompletionFilterTopK(completion_filter_top_k)

    # Instantiate our trainer
    trainer = SuperHFTrainer(
        language_model=language_model,
        reward_model=reward_model,
        language_tokenizer=language_tokenizer,
        reward_tokenizer=reward_tokenizer,
        completion_filter=completion_filter,
        training_args=training_args,
    )

    # Begin our experiment
    # wandb.init(project="superhf", entity="superhf", config=training_args)

    # Run training
    trainer.train(prompts)

    # Finish the experiment
    # wandb.finish()


if __name__ == "__main__":
    main()
