# 1. Load all data
# 2. Load model
# 3. Run model on all data samples,  collect scores and correctness
# 4. Compute Calibration Curve
#    a. divide into bins
#    b. calculate accuacy for each bin
# 5. Plot Calibration Curve

import numpy as np
from optimum.bettertransformer import BetterTransformer
import torch
from transformers import HfArgumentParser, AutoTokenizer
from tqdm import tqdm
from reward_modelling.reward_model import RewardModelTrainer, RewardModel, ModelArguments, PreferenceDataCollator
from combined_datasets import CombinedDataset

def compute_calibration_curve(diff, correct):
    diff_bins = np.arange(0, 3.0, 0.1)
    pm_accuracy = np.array([]) 
    for i in range(len(diff_bins) - 1):
        lower = diff_bins[i]
        upper = diff_bins[i + 1]
        selected_indices = np.where((diff >= lower) & (diff < upper))[0]
        selected_human_prefs = correct[selected_indices]

        if len(selected_indices) > 0:
            correct_preds = np.sum(selected_human_prefs)
            accuracy = correct_preds / len(selected_indices)
        else:
            accuracy = None

        pm_accuracy = np.append(pm_accuracy, accuracy)
    return diff_bins[:-1] + 0.05, pm_accuracy


if __name__ == '__main__':
    device = "cuda:0"
    model = RewardModel.from_pretrained("/nlp/scr/fongsu/rm_combined/gptneo_1.3B_first_half").to(device)
    # model = BetterTransformer.transform(model_hf, keep_original_model=True)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", max_length=300)
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
        
    collator = PreferenceDataCollator(tokenizer)

    combined_dataset = CombinedDataset.load("/juice2/scr2/fongsu/rm_combined/gptneo_1.3B_first_half/combined_dataset.pt")

    training_set = [combined_dataset.__getitem__(i, "train") for i in range(combined_dataset.__len__("train"))]
    validation_set = [combined_dataset.__getitem__(i, "val") for i in range(combined_dataset.__len__("val"))]
    full_set = training_set + validation_set

    import random
    diff = np.array([])
    correct = np.array([])
    for i, pair in enumerate(tqdm(full_set)):
        inputs = collator([pair]).to(device)
        outputs = model(**inputs)
        diff = np.append(diff, abs((outputs[1]-outputs[0]).detach().cpu().item()))
        correct = np.append(correct, (outputs[1] < outputs[0]).detach().cpu().item())
    
        # if i > 1000:
        #    break 

    print(diff)
    print(correct)

    print(sum(correct))
    
    bins, rm_accuracies = compute_calibration_curve(diff, correct)
    print(bins)
    print(rm_accuracies)


    
        
