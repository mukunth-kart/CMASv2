"""
SMILES -> SELFIES Preprocessing
===============================
Reads a newline-delimited SMILES file (e.g. `ChemBL_Smiles.txt` or an
AKT1 SMILES dump), converts every molecule to its SELFIES representation
(Krenn et al. 2019, https://arxiv.org/abs/1905.13741), builds a
WordLevel HuggingFace tokenizer over the SELFIES alphabet, and writes a
padded token-id matrix to disk for fast binary training.

Usage
-----
    # 1) Pretraining corpus — builds the vocab.
    python data/smiles_to_selfies.py \
        --input ./data/ChemBL_Smiles.txt \
        --out_npy ./data/ChemBL_Selfies.npy \
        --vocab_out ./selfies_vocab.json --build_vocab

    # 2) AKT1 corpus — reuses the SAME vocab.
    python data/smiles_to_selfies.py \
        --input ./data/AKT1_smiles.txt \
        --out_npy ./data/AKT1_Selfies.npy \
        --vocab_out ./selfies_vocab.json

Outputs
-------
* A HF `tokenizer.json` file with the SELFIES WordLevel vocab.
* A memory-mappable `int32` `.npy` of shape (N, MAX_LEN) containing
  `<s> ... </s>` token ids, padded with `<pad>`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

try:
    import selfies as sf
except ImportError as exc:
    raise ImportError(
        "The `selfies` package is required. Install with: pip install selfies"
    ) from exc

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing


SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
DEFAULT_MAX_LEN = 128


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def smiles_to_selfies_safe(smi: str) -> Optional[str]:
    """SMILES -> SELFIES. Returns None on failure."""
    try:
        return sf.encoder(smi.strip())
    except Exception:
        return None


def selfies_to_tokens(selfies_str: str) -> List[str]:
    """Split a SELFIES string into atomic `[...]` tokens."""
    return list(sf.split_selfies(selfies_str))


# ---------------------------------------------------------------------------
# Tokenizer construction
# ---------------------------------------------------------------------------

def build_tokenizer(selfies_corpus: List[str]) -> Tokenizer:
    """Build a WordLevel tokenizer over the SELFIES alphabet of the corpus."""
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for s in tqdm(selfies_corpus, desc="Scanning SELFIES alphabet"):
        for t in selfies_to_tokens(s):
            if t not in vocab:
                vocab[t] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>",  vocab["<s>"]),
            ("</s>", vocab["</s>"]),
        ],
    )
    tokenizer.enable_padding(pad_id=vocab["<pad>"], pad_token="<pad>")
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    input_path: str,
    out_npy: str,
    vocab_path: str,
    build_vocab: bool,
    max_len: int = DEFAULT_MAX_LEN,
) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    lines = [ln.strip() for ln in input_path.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    print(f"Loaded {len(lines)} SMILES from {input_path}")

    # ----- SMILES -> SELFIES --------------------------------------------------
    selfies_strs: List[str] = []
    failed = 0
    for smi in tqdm(lines, desc="SMILES -> SELFIES"):
        s = smiles_to_selfies_safe(smi)
        if s is None or len(s) == 0:
            failed += 1
            continue
        selfies_strs.append(s)
    print(f"Converted {len(selfies_strs)} / {len(lines)} molecules "
          f"({failed} failed)")

    # ----- Tokenizer ----------------------------------------------------------
    if build_vocab:
        tokenizer = build_tokenizer(selfies_strs)
        Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(vocab_path)
        print(f"Saved vocab ({tokenizer.get_vocab_size()} tokens) -> {vocab_path}")
    else:
        tokenizer = load_tokenizer(vocab_path)
        print(f"Loaded vocab from {vocab_path} "
              f"({tokenizer.get_vocab_size()} tokens)")

    pad_id = tokenizer.token_to_id("<pad>")

    # ----- Tokenize + pad -----------------------------------------------------
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>", length=max_len)
    tokenizer.enable_truncation(max_length=max_len)

    arr = np.full((len(selfies_strs), max_len), pad_id, dtype=np.int32)
    for i, s in enumerate(tqdm(selfies_strs, desc="Tokenising")):
        spaced = " ".join(selfies_to_tokens(s))
        enc = tokenizer.encode(spaced)
        ids = enc.ids[:max_len]
        arr[i, : len(ids)] = ids

    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, arr)
    print(f"Saved tokenized array {arr.shape} -> {out_npy}")

    meta = {
        "num_molecules": int(arr.shape[0]),
        "max_len": int(max_len),
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_id": pad_id,
    }
    print(json.dumps(meta, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="Input SMILES .txt file")
    p.add_argument("--out_npy",    required=True, help="Output tokenised .npy")
    p.add_argument("--vocab_out",  required=True, help="Path to tokenizer .json")
    p.add_argument("--build_vocab", action="store_true",
                   help="If set, build a new vocab from this corpus. "
                        "Otherwise the vocab at --vocab_out is loaded.")
    p.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        input_path=args.input,
        out_npy=args.out_npy,
        vocab_path=args.vocab_out,
        build_vocab=args.build_vocab,
        max_len=args.max_len,
    )
