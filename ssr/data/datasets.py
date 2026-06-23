"""Dataset loaders: GRID corpus, LRS2/LRS3, and custom self-recorded data."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class GridCorpusDataset(Dataset):
    def __init__(self, root: Path):
        self.root = Path(root)
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


class CustomPhraseDataset(Dataset):
    """Self-recorded phrase dataset (Phase 1, 50-200 phrases)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError
