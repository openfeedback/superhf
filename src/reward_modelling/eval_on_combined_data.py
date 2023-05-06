from transformers import HfArgumentParser, AutoTokenizer
from reward_modelling.reward_model import RewardModelTrainer, RewardModel, ModelArguments, PreferenceDataCollator, compute_metrics
from combined_datasets import CombinedDataset

if __name__ == '__main__':
    device = "cuda:0"
    model = RewardModel.from_pretrained("/nlp/scr/fongsu/rm_combined/gptneo_1.3B_first_half").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
   
    combined_dataset = CombinedDataset.load("/juice2/scr2/fongsu/rm_combined/gptneo_1.3B_first_half/combined_dataset.pt") 

    training_set = [combined_dataset.__getitem__(i, "train") for i in range(combined_dataset.__len__("train"))]
    trainer = RewardModelTrainer(
            model=model,
            data_collator=PreferenceDataCollator(tokenizer),
            compute_metrics=compute_metrics)
    result = trainer.evaluate(training_set)
    print(result)
   

