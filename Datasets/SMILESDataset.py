import re
import torch
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    """
    A PyTorch Dataset for loading and tokenizing SMILES strings from a text file.

    This dataset reads SMILES strings from a file, tokenizes them using a regex pattern to split chemical tokens,
    and prepares them for training with a tokenizer. It handles BOS/EOS tokens, padding, and label masking.

    Attributes:
        tokenizer: The tokenizer used for encoding SMILES strings.
        max_length (int): Maximum sequence length for tokenization.
        lines (List[str]): List of SMILES strings loaded from the file.
        regex (re.Pattern): Compiled regex pattern for splitting SMILES into tokens.

    Args:
        file_path (Path): Path to the text file containing SMILES strings, one per line.
        tokenizer: The tokenizer object for encoding sequences.
        max_length (int, optional): Maximum length of tokenized sequences. Default is 128.
    """
    def __init__(self, file_path: Path, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lines = self._load_lines(file_path)

        # Regex pattern to split SMILES into individual chemical tokens, handling atoms like Cl, Br, and symbols
        self.regex = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")

    def _load_lines(self, path: Path) -> List[str]:
        """
        Loads and strips lines from the text file, filtering out empty lines.

        Args:
            path (Path): Path to the text file.

        Returns:
            List[str]: List of non-empty, stripped lines from the file.
        """
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        """
        Returns the number of SMILES strings in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.lines)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Retrieves and tokenizes a SMILES string at the given index.

        Processes the SMILES string by splitting into tokens, adding BOS/EOS, tokenizing,
        and preparing input_ids, attention_mask, and labels for training.

        Args:
            idx (int): Index of the SMILES string to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'input_ids', 'attention_mask', and 'labels'.
        """
        line = self.lines[idx]

        # Split SMILES into tokens using regex
        tokens = self.regex.findall(line)

        # Join tokens with spaces for tokenizer compatibility
        spaced_text = " ".join(tokens)

        # Add BOS and EOS tokens to the sequence
        final_text = f"{self.tokenizer.bos_token} {spaced_text} {self.tokenizer.eos_token}"

        encodings = self.tokenizer(
            final_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        vocab_limit = len(self.tokenizer)
        labels[input_ids >= vocab_limit] = -100
        input_ids[input_ids >= vocab_limit] = self.tokenizer.unk_token_id or 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
