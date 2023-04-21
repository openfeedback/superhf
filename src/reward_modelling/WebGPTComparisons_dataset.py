import re
from torch.utils.data import Dataset
from datasets import load_dataset


class WebGPTComparisons(Dataset):
    def __init__(self, split="train"):
        data = load_dataset("openai/webgpt_comparisons", split=split)
        self.winner_responses = []
        self.loser_responses = []

        for example in data:
            if example['score_0'] != example['score_1']:
                winner_idx, loser_idx = (0, 1) if example['score_0'] > example['score_1'] else (1, 0)
                question = f"Question: {example['question']['full_text']}"
                winner_response = f"{question} Answer: {self.remove_reference_numbers(example[f'answer_{winner_idx}'])}"
                loser_response = f"{question} Answer: {self.remove_reference_numbers(example[f'answer_{loser_idx}'])}"
                self.winner_responses.append(winner_response)
                self.loser_responses.append(loser_response)

    @staticmethod
    def remove_reference_numbers(text):
        return re.sub(r'\[\d+\]', '', text)

    def __getitem__(self, idx):
        return self.winner_responses[idx], self.loser_responses[idx]

    def __len__(self):
        return len(self.winner_responses)

