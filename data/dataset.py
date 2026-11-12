import numpy as np
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """Returns (x, y) windows where y is x shifted one character to the right."""

    def __init__(self, bin_path, context_length):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.context_length + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
