import re
from torch.utils.data import Dataset
from datasets import load_dataset


class WebGPTComparisons(Dataset):
    def __init__(self, split="train"):
        data = load_dataset("openai/webgpt_comparisons", split=split)
        self.filtered_responses = [
            (self.remove_reference_numbers(example[f'answer_{winner_idx}']),
             self.remove_reference_numbers(example[f'answer_{loser_idx}']))
            for example in data
            if example['score_0'] != example['score_1']
            for winner_idx, loser_idx in [(0, 1) if example['score_0'] > example['score_1'] else (1, 0)]
        ]

    @staticmethod
    def remove_reference_numbers(text):
        return re.sub(r'\[\d+\]', '', text)

    def __getitem__(self, idx):
        return self.filtered_responses[idx]

    def __len__(self):
        return len(self.filtered_responses)



