"""
Train one AttentionOracle per ADMET task (one model per task, attention-based).

Loads the existing latent dataset produced by create_multitask_dataset_selfies.py,
groups samples by task, auto-configures the transformer depth and width per task
(more data → deeper / wider model), then trains and saves one checkpoint per task.

Auto-sizing heuristic:
    n_layers  : 1 (<500 samples) / 2 (500-2000) / 3 (>2000)
    model_dim : scales with data density, rounded to mult of 16
    n_heads   : model_dim // 16, at least 1
    n_patches : largest divisor of latent_dim close to 8

Usage:
    python Models/AdmetClassifier/train_attention_admet.py --variant pretrain
    python Models/AdmetClassifier/train_attention_admet.py --variant akt1

    # Pin specific architecture (disables auto-sizing for that param):
    python Models/AdmetClassifier/train_attention_admet.py --variant akt1 \\
        --n_patches 8 --model_dim 64 --n_heads 4 --n_layers 2 --epochs 200

Outputs: ckpts/attention_admet/{variant}/{task_name}.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from attention_oracle import AttentionOracle


# ---------------------------------------------------------------------------
# Auto-config
# ---------------------------------------------------------------------------

def auto_config(n_samples: int, latent_dim: int) -> dict:
    """Scale transformer depth and width with dataset size."""
    if n_samples < 500:
        n_layers, model_dim, epochs, patience = 1, 32, 100, 15
    elif n_samples < 2000:
        n_layers, model_dim, epochs, patience = 2, 64, 200, 20
    else:
        n_layers, epochs, patience = 3, 300, 25
        model_dim = min(128, (latent_dim // 2 // 16) * 16)
        model_dim = max(model_dim, 32)
    return {"n_layers": n_layers, "model_dim": model_dim,
            "epochs": epochs, "patience": patience}


# ---------------------------------------------------------------------------
# Dataset helpers (identical pattern to train_multitask_grownet.py)
# ---------------------------------------------------------------------------

def load_task_tensors(
    dataset_path: Path, task_name: str, task_idx: int
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    pack = torch.load(dataset_path, weights_only=False)
    zs, ys = [], []
    for item in pack["data"]:
        if item["task_idx"] == task_idx:
            zs.append(torch.tensor(item["z"], dtype=torch.float32))
            ys.append(torch.tensor(float(item["y"]), dtype=torch.float32))
    if not zs:
        return None, None, pack["latent_dim"]
    return torch.stack(zs), torch.stack(ys).unsqueeze(1), pack["latent_dim"]


# ---------------------------------------------------------------------------
# Per-task training
# ---------------------------------------------------------------------------

def train_task(
    task_name: str,
    task_idx: int,
    dataset_path: Path,
    save_dir: Path,
    n_patches_override: int | None = None,
    model_dim_override: int | None = None,
    n_heads_override: int | None = None,
    n_layers_override: int | None = None,
    epochs_override: int | None = None,
    batch_size: int = 256,
    lr: float = 1e-3,
    dropout: float = 0.1,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zs, ys, latent_dim = load_task_tensors(dataset_path, task_name, task_idx)
    if zs is None:
        print(f"  [skip] {task_name}: no samples in dataset")
        return

    cfg = auto_config(len(zs), latent_dim)
    if model_dim_override  is not None: cfg["model_dim"] = model_dim_override
    if n_layers_override   is not None: cfg["n_layers"]  = n_layers_override
    if epochs_override     is not None: cfg["epochs"]    = epochs_override

    model = AttentionOracle(
        latent_dim=latent_dim,
        task_name=task_name,
        n_patches=n_patches_override or 0,   # 0 = auto
        model_dim=cfg["model_dim"],
        n_heads=n_heads_override or 0,       # 0 = auto
        n_layers=cfg["n_layers"],
        dropout=dropout,
    ).to(device)

    print(
        f"\n  {task_name}: {len(zs)} samples | "
        f"patches={model.n_patches}  model_dim={model.model_dim}  "
        f"n_heads={model.n_heads}  n_layers={model.n_layers}  "
        f"epochs={cfg['epochs']}"
    )

    eff_batch = min(batch_size, max(2, len(zs)))
    loader    = DataLoader(TensorDataset(zs, ys), batch_size=eff_batch, shuffle=True)
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    crit      = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=cfg["patience"]
    )

    best_loss, no_improve = float("inf"), 0
    patience_2x = cfg["patience"] * 2

    for epoch in range(cfg["epochs"]):
        model.train()
        total, seen = 0.0, 0
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            if zb.size(0) < 2:
                continue
            loss = crit(model(zb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * zb.size(0)
            seen  += zb.size(0)

        avg = total / max(1, seen)
        sched.step(avg)

        if avg < best_loss - 1e-4:
            best_loss, no_improve = avg, 0
        else:
            no_improve += 1
            if no_improve >= patience_2x:
                print(f"    early stop @ epoch {epoch + 1}  loss={avg:.4f}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch + 1:4d}  loss={avg:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    model.is_fitted = True
    out_path = save_dir / f"{task_name}.pt"
    model.save(str(out_path))
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def train(
    variant: str,
    models_dir: str = "./Models/AdmetClassifier",
    ckpt_dir: str   = "./ckpts/attention_admet",
    n_patches: int | None = None,
    model_dim: int | None = None,
    n_heads: int | None   = None,
    n_layers: int | None  = None,
    epochs: int | None    = None,
    batch_size: int = 256,
    lr: float = 1e-3,
    dropout: float = 0.1,
) -> None:
    models_dir = Path(models_dir)
    train_pt   = models_dir / f"admet_latent_selfies_{variant}_train.pt"
    if not train_pt.exists():
        raise FileNotFoundError(
            f"Dataset not found: {train_pt}\n"
            "Run create_multitask_dataset_selfies.py first."
        )

    pack  = torch.load(train_pt, weights_only=False)
    tasks: list[str] = pack["tasks"]
    save_dir = Path(ckpt_dir) / variant
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Attention ADMET training  variant={variant}  tasks={tasks}")
    for task_idx, task_name in enumerate(tasks):
        train_task(
            task_name=task_name,
            task_idx=task_idx,
            dataset_path=train_pt,
            save_dir=save_dir,
            n_patches_override=n_patches,
            model_dim_override=model_dim,
            n_heads_override=n_heads,
            n_layers_override=n_layers,
            epochs_override=epochs,
            batch_size=batch_size,
            lr=lr,
            dropout=dropout,
        )

    print(f"\nAll tasks done. Checkpoints -> {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train one AttentionOracle per ADMET task.")
    p.add_argument("--variant",   choices=["pretrain", "akt1"], required=True)
    p.add_argument("--models_dir", default="./Models/AdmetClassifier")
    p.add_argument("--ckpt_dir",   default="./ckpts/attention_admet")
    p.add_argument("--n_patches",  type=int, default=None,
                   help="Number of patches z is split into (0 or omit = auto).")
    p.add_argument("--model_dim",  type=int, default=None,
                   help="Transformer hidden width (omit = auto).")
    p.add_argument("--n_heads",    type=int, default=None,
                   help="Attention heads (omit = auto).")
    p.add_argument("--n_layers",   type=int, default=None,
                   help="Transformer encoder depth (omit = auto).")
    p.add_argument("--epochs",     type=int, default=None,
                   help="Max epochs (early stopping still applies).")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout",    type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    train(
        variant=a.variant,
        models_dir=a.models_dir,
        ckpt_dir=a.ckpt_dir,
        n_patches=a.n_patches,
        model_dim=a.model_dim,
        n_heads=a.n_heads,
        n_layers=a.n_layers,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        dropout=a.dropout,
    )
