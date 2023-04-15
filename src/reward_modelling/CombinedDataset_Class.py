import random

class CombinedDataset:
    def __init__(self, datasets, split_ratio=0.8):
        self.response_pairs = []

        for dataset in datasets:
            for idx in range(len(dataset)):
                winner, loser = dataset.winner_responses[idx], dataset.loser_responses[idx]
                self.response_pairs.append((winner, loser))

        self.total_length = len(self.response_pairs)
        self.indices = list(range(self.total_length))

        # Shuffle the indices before splitting
        random.shuffle(self.indices)

        split_idx = int(self.total_length * split_ratio)
        self.train_indices = self.indices[:split_idx]
        self.val_indices = self.indices[split_idx:]

    def __getitem__(self, idx, split="train"):
        if split == "train":
            idx = self.train_indices[idx[0]]
        elif split == "val":
            idx = self.val_indices[idx[0]]
        else:
            raise ValueError("Invalid split, use 'train' or 'val'")

        return self.response_pairs[idx]

    def __len__(self, split="train"):
        if split == "train":
            return len(self.train_indices)
        elif split == "val":
            return len(self.val_indices)
        else:
            raise ValueError("Invalid split, use 'train' or 'val'")

    def shuffle(self):
        random.shuffle(self.indices)


web_gpt_comparisons = WebGPTComparisons()
anthropic_helpful_harmless = AnthropicHelpfulHarmless()


combined_dataset = CombinedDataset([
    web_gpt_comparisons,
    anthropic_helpful_harmless,
])


combined_dataset.shuffle()

