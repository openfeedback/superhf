"""
Client code showing how to call the training loop for the iterative version of the SuperHF model.
"""

from superhf.data import get_superhf_prompts
from superhf.training import SuperHFTrainingArguments, SuperHFTrainer

# import wandb


def main() -> None:
    """
    Instantiate and train the SuperHF model.
    """

    # Get the prompt dataset
    prompts = get_superhf_prompts("anthropic-red-team")

    # Instantiate our language and reward models and tokenizers
    language_model = None
    reward_model = None
    language_tokenizer = None
    reward_tokenizer = None

    # Set our training arguments
    training_args = SuperHFTrainingArguments(
        language_model=language_model,
        reward_model=reward_model,
        language_tokenizer=language_tokenizer,
        reward_tokenizer=reward_tokenizer,
        prompts=prompts,
        temperature=1.0,
        completions_per_prompt=2,
    )

    # Instantiate our trainer
    trainer = SuperHFTrainer(training_args)

    # Begin our experiment
    # wandb.init(project="superhf", entity="superhf", config=training_args)

    # Run training
    trainer.train()

    # Finish the experiment
    # wandb.finish()


if __name__ == "__main__":
    main()
