import datasets
import torch
from transformers import (
    TokenizersBackend, 
    SentencePieceBackend,
)

from dataset import Dataset


class PackedChunkDataset(datasets.Dataset):
    """
    Packs normalized dataset rows ("chunk" column) into sequences
    up to block_size tokens.
    """


    def __init__(self, dataset: Dataset, tokenizer: TokenizersBackend | SentencePieceBackend, block_size: int = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.samples = self._pack_sequences()


    def _pack_sequences(self) -> list[torch.Tensor]:
        samples = []
        buffer = []

        for row in self.dataset:
            text = row.get("chunk")
            if not text:
                continue

            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False
            )

            # if adding this chunk would overflow the block
            if len(buffer) + len(tokens) > self.block_size:
                if len(buffer) > 1:
                    samples.append(torch.tensor(buffer[:self.block_size]))
                buffer = []

            buffer.extend(tokens)

        # final partial chunk
        if len(buffer) > 1:
            samples.append(torch.tensor(buffer[:self.block_size]))

        return samples


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> dict:
        tokens = self.samples[idx]

        input_ids = tokens[:-1]
        labels = tokens[1:]

        return {
            "input_ids": input_ids.long(),
            "labels": labels.long(),
        }