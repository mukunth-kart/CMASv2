"""
Build the AKT1 activity latent dataset for the SELFIES VAE.

Reads an AKT1 pChEMBL Excel export, extracts (SMILES, pChEMBL) pairs,
binarises the activity label at pChEMBL >= 6.0, encodes every SMILES
through a `SelfiesVAE` to a deterministic `mu` latent, and saves a
`.pt` with schema `{"z": FloatTensor[N,128], "y": FloatTensor[N,1]}`
for consumption by `Models/ActivityClassifier/train_mlp.py`
(or its SELFIES shim).

Default xlsx: `Models/ActivityClassifier/AKT1 CHEMBL.xlsx`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import selfies as sf
except ImportError as exc:
    raise ImportError("Install `selfies`: pip install selfies") from exc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Generators.SelfiesVAE import SelfiesVAE


DEFAULT_XLSX = "./Models/ActivityClassifier/AKT1 CHEMBL.xlsx"
DEFAULT_OUT  = "./Models/ActivityClassifier/latent_dataset_selfies.pt"
PCHEMBL_CUTOFF = 6.0
BATCH_SIZE = 256
MAX_LEN = 512


def _pick_pchembl_column(df: pd.DataFrame) -> str:
    """Find the pChEMBL column by name, falling back to the rightmost numeric column."""
    for name in df.columns:
        if str(name).strip().lower().startswith("pchembl"):
            return name
    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No pChEMBL-like or numeric column found in xlsx.")
    return numeric_cols[-1]


def _pick_smiles_column(df: pd.DataFrame) -> str:
    for name in df.columns:
        if str(name).strip().lower() in {"smiles", "canonical_smiles", "canonicalsmiles"}:
            return name
    raise ValueError("No SMILES column found in xlsx.")


def build(xlsx: str, vae_weights: str, vocab: str, out_path: str,
          cutoff: float = PCHEMBL_CUTOFF) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = SelfiesVAE(model_path=vae_weights)
    vae.load_model(vocab_base=vocab)
    vae.model.to(device).eval()

    df = pd.read_excel(xlsx)
    smi_col = _pick_smiles_column(df)
    pch_col = _pick_pchembl_column(df)
    print(f"SMILES column: '{smi_col}'  |  pChEMBL column: '{pch_col}'")

    df = df[[smi_col, pch_col]].dropna()
    df = df[df[smi_col].astype(str).str.len() > 0]
    print(f"Loaded {len(df)} rows from {xlsx}")

    z_rows: list[np.ndarray] = []
    y_rows: list[float] = []

    inner = vae.model

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Encoding AKT1"):
        batch = df.iloc[i : i + BATCH_SIZE]
        token_tensors, labels = [], []
        for smi, pch in zip(batch[smi_col].astype(str), batch[pch_col].astype(float)):
            try:
                s = sf.encoder(smi)
                if not s:
                    continue
                toks = list(sf.split_selfies(s))
                spaced = " ".join(toks)
                wrapped = f"{vae.tokenizer.bos_token} {spaced} {vae.tokenizer.eos_token}"
                enc = vae.tokenizer(wrapped, truncation=True, max_length=MAX_LEN,
                                    padding="max_length", return_tensors="pt")
                token_tensors.append(enc["input_ids"].squeeze(0))
                labels.append(1.0 if pch >= cutoff else 0.0)
            except Exception:
                continue

        if not token_tensors:
            continue

        input_ids = torch.stack(token_tensors).to(device)
        with torch.no_grad():
            embedded = inner.embedding(input_ids)
            _, h_n = inner.encoder_gru(embedded)
            h_cat = torch.cat((h_n[0], h_n[1]), dim=1)
            mu = inner.fc_mu(h_cat).detach().cpu().numpy().astype(np.float32)

        z_rows.extend(mu)
        y_rows.extend(labels)

    z = torch.tensor(np.stack(z_rows), dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.float32).unsqueeze(1)

    pos = int(y.sum().item())
    neg = int(y.numel() - pos)
    print(f"Encoded {len(z)} molecules  |  active (pChEMBL>={cutoff}): {pos}  inactive: {neg}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z, "y": y}, out_path)
    print(f"Saved activity latent dataset -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx",  default=DEFAULT_XLSX)
    p.add_argument("--vae",   required=True, help="SELFIES VAE .pt weights")
    p.add_argument("--vocab", required=True, help="SELFIES vocab JSON")
    p.add_argument("--out",   default=DEFAULT_OUT)
    p.add_argument("--cutoff", type=float, default=PCHEMBL_CUTOFF,
                   help="pChEMBL binarisation threshold (default 6.0)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    build(xlsx=a.xlsx, vae_weights=a.vae, vocab=a.vocab,
          out_path=a.out, cutoff=a.cutoff)
