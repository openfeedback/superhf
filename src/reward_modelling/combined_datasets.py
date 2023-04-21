import random

class CombinedDataset:
    def __init__(self, datasets, split_ratio=0.5):
        self.train_winner_pairs = []
        self.train_loser_pairs = []
        self.val_winner_pairs = []
        self.val_loser_pairs = []

        for dataset in datasets:
            dataset_length = len(dataset)
            split_idx = int(dataset_length * split_ratio)

            for idx in range(dataset_length):
                winner, loser = dataset.winner_responses[idx], dataset.loser_responses[idx]
                if idx < split_idx:
                    self.train_winner_pairs.append(winner)
                    self.train_loser_pairs.append(loser)
                else:
                    self.val_winner_pairs.append(winner)
                    self.val_loser_pairs.append(loser)

        self.train_indices = list(range(len(self.train_winner_pairs)))
        self.val_indices = list(range(len(self.val_winner_pairs)))

        random.shuffle(self.train_indices)
        random.shuffle(self.val_indices)

    def __getitem__(self, idx, split="train"):
        if split == "train":
            idx = self.train_indices[idx]
        elif split == "val":
            idx = self.val_indices[idx]
        else:
            raise ValueError("Invalid split, use 'train' or 'val'")

        if split == "train":
            return self.train_winner_pairs[idx], self.train_loser_pairs[idx]
        else:
            return self.val_winner_pairs[idx], self.val_loser_pairs[idx]

    def __len__(self, split="train"):
        if split == "train":
            return len(self.train_indices)
        elif split == "val":
            return len(self.val_indices)
        else:
            raise ValueError("Invalid split, use 'train' or 'val'")

    def shuffle(self):
        random.shuffle(self.train_indices)
        random.shuffle(self.val_indices)

# Instantiate your dataset classes here
# web_gpt_comparisons = WebGPTComparisons()
# anthropic_helpful_harmless = AnthropicHelpfulHarmless()
# summarize_feedback = SummarizeFromFeedbackComparisons()
# instrucptgpt_pairwise = CompatibleSyntheticInstructGPTJPairwise()

# combined_dataset = CombinedDataset([
#     web_gpt_comparisons,
#     anthropic_helpful_harmless,
#     instrucptgpt_pairwise,
#     summarize_feedback
# ])

# combined_dataset.shuffle()

