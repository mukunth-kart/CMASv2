"""
SELFIES Datasets
================
Two dataset classes for SELFIES-based VAE training:

* `BinarySELFIESDataset` — memory-maps a pre-tokenized `.npy` produced
  by `data/smiles_to_selfies.py` (used for large pretraining corpora).
* `SELFIESTextDataset`   — reads a plain-text SMILES or SELFIES file and
  tokenizes on the fly (convenient for small fine-tuning sets like AKT1).

Both yield `{"input_ids": LongTensor, "labels": LongTensor}` where
`labels` equals `input_ids` with `<pad>` positions masked to `-100`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import selfies as sf
except ImportError:
    sf = None


__all__ = ["BinarySELFIESDataset", "SELFIESTextDataset"]


class BinarySELFIESDataset(Dataset):
    """Memory-mapped tokenized SELFIES dataset."""

    def __init__(self, npy_path: str, pad_token_id: int = 0):
        self.data = np.load(npy_path, mmap_mode="r")
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.from_numpy(self.data[idx].astype(np.int64))
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}


class SELFIESTextDataset(Dataset):
    """
    Text-mode SELFIES dataset.  Accepts SMILES or SELFIES (auto-detected
    by the presence of `[` in the first non-empty lines).
    """

    def __init__(
        self,
        file_path: Path,
        tokenizer,
        max_length: int = 128,
        input_format: str = "auto",
    ):
        if sf is None:
            raise ImportError("Install `selfies` to use SELFIESTextDataset.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lines = self._load_lines(Path(file_path))
        self.input_format = self._detect_format(input_format)

    def _load_lines(self, path: Path) -> List[str]:
        with path.open("r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def _detect_format(self, hint: str) -> str:
        if hint != "auto":
            return hint
        for ln in self.lines[:5]:
            if ln.startswith("["):
                return "selfies"
        return "smiles"

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw = self.lines[idx]
        if self.input_format == "smiles":
            try:
                selfies_str = sf.encoder(raw)
            except Exception:
                selfies_str = ""
        else:
            selfies_str = raw

        tokens = list(sf.split_selfies(selfies_str)) if selfies_str else []
        spaced = " ".join(tokens)
        final  = f"{self.tokenizer.bos_token} {spaced} {self.tokenizer.eos_token}"

        enc = self.tokenizer(
            final,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)

        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        vocab_limit = len(self.tokenizer)
        labels[input_ids >= vocab_limit] = -100
        input_ids[input_ids >= vocab_limit] = self.tokenizer.unk_token_id or 0

        return {"input_ids": input_ids, "labels": labels}
