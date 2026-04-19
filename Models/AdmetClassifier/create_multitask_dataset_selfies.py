"""
Build the multitask ADMET latent dataset for the SELFIES VAE.

For every task folder under `Models/AdmetClassifier/Auto_ML_dataset/`,
read the train and test CSVs, convert each SMILES to SELFIES, encode it
through the SELFIES VAE (`SelfiesVAE`) to a deterministic `mu` latent,
and persist the aggregated `(z, y, task_idx)` list as a `.pt` file.

Two encoder variants are supported:
  * `--variant pretrain` : ChEMBL-pretrained SELFIES VAE
                           -> admet_latent_selfies_pretrain_{train,test}.pt
  * `--variant akt1`     : AKT1-fine-tuned SELFIES VAE
                           -> admet_latent_selfies_akt1_{train,test}.pt

Output schema (identical to the existing bidirec `.pt` files):

    {
      "tasks": [<11 sorted task names>],
      "data":  [{"z": np.float32[128], "y": float, "task_idx": int}, ...],
      "latent_dim": 128,
    }
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

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


DEFAULT_DATA_ROOT = "./Models/AdmetClassifier/Auto_ML_dataset"
BATCH_SIZE = 256
MAX_LEN = 512  # SELFIES token sequences; 512 comfortably covers drug-like molecules


def encode_split(
    vae: SelfiesVAE,
    data_root: Path,
    split_suffix: str,
    device: torch.device,
) -> tuple[list, list[str]]:
    """Iterate all task folders for a given split and encode every SMILES."""
    tasks = sorted(d.name for d in data_root.iterdir()
                   if d.is_dir() and not d.name.startswith("."))
    print(f"Found {len(tasks)} tasks: {tasks}")

    pad_id = vae.tokenizer.pad_token_id or 0
    all_rows: list = []

    for task_idx, task in enumerate(tasks):
        csv_files = list((data_root / task).glob(f"*{split_suffix}.csv"))
        if not csv_files:
            print(f"  [skip] {task}: no *{split_suffix}.csv")
            continue
        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)
        label_col = "bioclass" if "bioclass" in df.columns else df.columns[-1]
        smiles_list = df["SMILES"].astype(str).tolist()
        labels_list = df[label_col].tolist()

        print(f"  Processing {task}  ({len(df)} rows)")
        for i in tqdm(range(0, len(df), BATCH_SIZE), leave=False, desc=task):
            batch_smi = smiles_list[i : i + BATCH_SIZE]
            batch_lbl = labels_list[i : i + BATCH_SIZE]

            token_batches, valid_labels = [], []
            for smi, lbl in zip(batch_smi, batch_lbl):
                try:
                    s = sf.encoder(str(smi))
                    if not s:
                        continue
                    toks = list(sf.split_selfies(s))
                    spaced = " ".join(toks)
                    wrapped = f"{vae.tokenizer.bos_token} {spaced} {vae.tokenizer.eos_token}"
                    enc = vae.tokenizer(
                        wrapped,
                        truncation=True,
                        max_length=MAX_LEN,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    token_batches.append(enc["input_ids"].squeeze(0))
                    valid_labels.append(float(lbl))
                except Exception:
                    continue

            if not token_batches:
                continue

            input_ids = torch.stack(token_batches).to(device)
            inner = vae.model.module if hasattr(vae.model, "module") else vae.model
            inner.eval()
            with torch.no_grad():
                embedded = inner.embedding(input_ids)
                _, h_n = inner.encoder_gru(embedded)
                h_cat = torch.cat((h_n[0], h_n[1]), dim=1)
                mu = inner.fc_mu(h_cat).detach().cpu().numpy().astype(np.float32)

            for z_vec, y in zip(mu, valid_labels):
                all_rows.append({"z": z_vec, "y": y, "task_idx": task_idx})

    return all_rows, tasks


def build(
    vae_weights: str,
    vocab: str,
    variant: str,
    data_root: str = DEFAULT_DATA_ROOT,
    out_dir: str = "./Models/AdmetClassifier",
) -> None:
    if not torch.cuda.is_available():
        print("[warn] CUDA not detected — encoding will run on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = SelfiesVAE(model_path=vae_weights)
    vae.load_model(vocab_base=vocab)
    vae.model.to(device)

    data_root = Path(data_root)
    latent_dim = vae.model.latent_dim
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_suffix, tag in [("_train_set", "train"), ("_test_set", "test")]:
        rows, tasks = encode_split(vae, data_root, split_suffix, device)
        out_file = out_dir / f"admet_latent_selfies_{variant}_{tag}.pt"
        torch.save(
            {"tasks": tasks, "data": rows, "latent_dim": latent_dim},
            out_file,
        )
        print(f"Saved {len(rows)} samples -> {out_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["pretrain", "akt1"], required=True)
    p.add_argument("--vae", required=True, help="SELFIES VAE .pt weights")
    p.add_argument("--vocab", required=True, help="SELFIES vocab JSON")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--out_dir", default="./Models/AdmetClassifier")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        vae_weights=args.vae,
        vocab=args.vocab,
        variant=args.variant,
        data_root=args.data_root,
        out_dir=args.out_dir,
    )
