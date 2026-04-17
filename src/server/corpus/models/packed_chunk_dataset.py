import datasets
import torch
from torch.utils.data import Dataset

from ai_clients.tokenizer import load_tokenizer

class PackedChunkDataset(Dataset):
    """
    Packs dataset rows ("tokens" column) into sequences up to block_size tokens.
    Samples are generated dynamically instead of precomputed.
    """

    dataset: datasets.Dataset
    block_size: int


    def __init__(self, dataset: datasets.Dataset, block_size=512):
        self.dataset = dataset
        self.block_size = block_size
        self.tokenizer = load_tokenizer()


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx: int):
        buffer = []
        i = idx

        while len(buffer) < self.block_size and i < len(self.dataset):
            row = self.dataset[i]
            tokens = row.get("tokens")

            # Skip rows without tokens
            if not tokens:
                i += 1
                continue

            # Remove invalid values
            tokens = [t for t in tokens if t is not None]

            if not tokens:
                i += 1
                continue

            if len(buffer) + len(tokens) > self.block_size:
                break

            buffer.extend(tokens)
            i += 1

        # Fallback if we didn't gather enough tokens
        if len(buffer) < 2:
            tokens = self.dataset[idx].get("tokens") or []
            tokens = [t for t in tokens if t is not None]
            buffer = tokens[:self.block_size]

        # Pad if needed
        if len(buffer) < self.block_size:
            pad_id = getattr(self, "pad_token_id", 0)
            buffer.extend([pad_id] * (self.block_size - len(buffer)))

        tokens = torch.tensor(buffer[:self.block_size], dtype=torch.long)

        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }