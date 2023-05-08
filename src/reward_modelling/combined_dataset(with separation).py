import random

class CombinedDataset:
    def __init__(self, datasets, split_ratio=0.5, lm_split_ratio=0.05):
        self.train_winner_pairs = []
        self.train_loser_pairs = []
        self.val_winner_pairs = []
        self.val_loser_pairs = []
        self.LM_train_winner_pairs = []
        self.LM_train_loser_pairs = []
        self.LM_val_winner_pairs = []
        self.LM_val_loser_pairs = []

        for dataset in datasets:
            dataset_length = len(dataset)
            indices = list(range(dataset_length))
            random.shuffle(indices)

            train_split_idx = int(dataset_length * split_ratio)
            lm_train_split_idx = int(train_split_idx * lm_split_ratio)
            lm_val_split_idx = int((dataset_length - train_split_idx) * lm_split_ratio)

            for idx in range(dataset_length):
                winner, loser = dataset.winner_responses[indices[idx]], dataset.loser_responses[indices[idx]]
                if idx < lm_train_split_idx:
                    self.LM_train_winner_pairs.append(winner)
                    self.LM_train_loser_pairs.append(loser)
                elif idx < train_split_idx:
                    self.train_winner_pairs.append(winner)
                    self.train_loser_pairs.append(loser)
                elif idx < train_split_idx + lm_val_split_idx:
                    self.LM_val_winner_pairs.append(winner)
                    self.LM_val_loser_pairs.append(loser)
                else:
                    self.val_winner_pairs.append(winner)
                    self.val_loser_pairs.append(loser)

        self.train_indices = list(range(len(self.train_winner_pairs)))
        self.val_indices = list(range(len(self.val_winner_pairs)))
        self.LM_train_indices = list(range(len(self.LM_train_winner_pairs)))
        self.LM_val_indices = list(range(len(self.LM_val_winner_pairs)))

        random.shuffle(self.train_indices)
        random.shuffle(self.val_indices)
        random.shuffle(self.LM_train_indices)
        random.shuffle(self.LM_val_indices)

    def __getitem__(self, idx, split="train"):
        if split == "train":
            idx = self.train_indices[idx]
        elif split == "val":
            idx = self.val_indices[idx]
        elif split == "LM_train":
            idx = self.LM_train_indices[idx]
        elif split == "LM_val":
            idx = self.LM_val_indices[idx]
        else:
            raise ValueError("Invalid split, use 'train', 'val', 'LM_train', or 'LM_val'")

        if split == "train":
            return self.train_winner_pairs[idx], self.train_loser_pairs[idx]
        elif split == "val":
            return self.val_winner_pairs[idx], self.val_loser_pairs[idx]
        elif split == "LM_train":
            return self.LM_train_winner_pairs[idx], self.LM_train_loser_pairs[idx]
        else:
            return self.LM_val_winner_pairs[idx], self.LM_val_loser_pairs[idx]

    def __len__(self, split="train"):
        if split == "train":
            return len(self.train_indices)
        elif split == "val":
            return len(self.val_indices)
        elif split == "LM_train":
            return len(self.LM_train_indices)
        elif split == "LM_val":
            return len(self.LM_val_indices)
        else:
            raise ValueError("Invalid split, use 'train', 'val', 'LM_train', or 'LM_val'")

    def shuffle(self):
        random.shuffle(self.train_indices)
        random.shuffle(self.val_indices)
        random.shuffle(self.LM_train_indices)
        random.shuffle(self.LM_val_indices
