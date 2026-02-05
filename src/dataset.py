import torch
from torch.utils.data import Dataset
from src.utils import encode, pad_sequence, PAD_TOKEN

class NERDataset(Dataset):
    def __init__(self, data, token2id, label2id, max_len):
        self.data = data
        self.token2id = token2id
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        token_ids = encode(row["tokens"], self.token2id)
        label_ids = [self.label2id[l] for l in row["labels"]]

        token_ids = pad_sequence(
            token_ids,
            self.max_len,
            self.token2id[PAD_TOKEN]
        )

        label_ids = pad_sequence(
            label_ids,
            self.max_len,
            self.label2id["O"]
        )

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long)
        )
