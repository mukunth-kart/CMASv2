"""
Train one GrowNetOracle per ADMET task (one model per task, no multi-task sharing).

Loads the existing latent dataset produced by create_multitask_dataset_selfies.py,
groups samples by task, auto-configures model complexity per task (like tree depth
in a Random Forest scaling with data), then trains and saves one checkpoint per task.

Auto-sizing heuristic (controllable via --max_learners):
    n_learners = ceil(log2(n_samples))  clamped to [4, max_learners]
    hidden_dim and epochs scale in 3 bands: <500, 500-2000, >2000 samples.

Usage:
    python Models/AdmetClassifier/train_multitask_grownet.py --variant pretrain
    python Models/AdmetClassifier/train_multitask_grownet.py --variant akt1
    python Models/AdmetClassifier/train_multitask_grownet.py --variant akt1 --max_learners 8

Outputs: ckpts/grownet_admet/{variant}/{task_name}.pt
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

from grownet_oracle import GrowNetOracle


# ---------------------------------------------------------------------------
# Auto-config: scales like Random Forest depth with data size
# ---------------------------------------------------------------------------

def auto_config(n_samples: int, latent_dim: int, max_learners: int) -> dict:
    """Determine model size from dataset size — more data, deeper ensemble."""
    n_learners = max(4, min(max_learners, int(math.ceil(math.log2(n_samples + 2)))))
    if n_samples < 500:
        return {"n_learners": n_learners, "hidden_dim": 128,  "epochs": 100, "patience": 15}
    if n_samples < 2000:
        return {"n_learners": n_learners, "hidden_dim": 256,  "epochs": 200, "patience": 20}
    hidden_dim = min(512, (latent_dim * 2 // 64) * 64)
    return     {"n_learners": n_learners, "hidden_dim": hidden_dim, "epochs": 300, "patience": 25}


# ---------------------------------------------------------------------------
# Dataset helpers
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
    max_learners: int = 16,
    n_learners_override: int | None = None,
    hidden_dim_override: int | None = None,
    epochs_override: int | None = None,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zs, ys, latent_dim = load_task_tensors(dataset_path, task_name, task_idx)
    if zs is None:
        print(f"  [skip] {task_name}: no samples in dataset")
        return

    cfg = auto_config(len(zs), latent_dim, max_learners)
    if n_learners_override is not None:
        cfg["n_learners"] = n_learners_override
    if hidden_dim_override is not None:
        cfg["hidden_dim"] = hidden_dim_override
    if epochs_override is not None:
        cfg["epochs"] = epochs_override

    print(
        f"\n  {task_name}: {len(zs)} samples | "
        f"learners={cfg['n_learners']}  hidden={cfg['hidden_dim']}  "
        f"epochs={cfg['epochs']}"
    )

    model = GrowNetOracle(
        latent_dim=latent_dim,
        num_tasks=1,
        task_names=[task_name],
        n_learners=cfg["n_learners"],
        hidden_dim=cfg["hidden_dim"],
    ).to(device)

    eff_batch = min(batch_size, max(2, len(zs)))
    loader = DataLoader(TensorDataset(zs, ys), batch_size=eff_batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
            seen += zb.size(0)

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
    ckpt_dir: str = "./ckpts/grownet_admet",
    max_learners: int = 16,
    n_learners: int | None = None,
    hidden_dim: int | None = None,
    epochs: int | None = None,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> None:
    models_dir = Path(models_dir)
    train_pt = models_dir / f"admet_latent_selfies_{variant}_train.pt"
    if not train_pt.exists():
        raise FileNotFoundError(
            f"Dataset not found: {train_pt}\n"
            "Run create_multitask_dataset_selfies.py first."
        )

    pack = torch.load(train_pt, weights_only=False)
    tasks: list[str] = pack["tasks"]
    save_dir = Path(ckpt_dir) / variant
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"GrowNet ADMET training  variant={variant}  tasks={tasks}")
    for task_idx, task_name in enumerate(tasks):
        train_task(
            task_name=task_name,
            task_idx=task_idx,
            dataset_path=train_pt,
            save_dir=save_dir,
            max_learners=max_learners,
            n_learners_override=n_learners,
            hidden_dim_override=hidden_dim,
            epochs_override=epochs,
            batch_size=batch_size,
            lr=lr,
        )

    print(f"\nAll tasks done. Checkpoints -> {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train one GrowNet per ADMET task.")
    p.add_argument("--variant", choices=["pretrain", "akt1"], required=True)
    p.add_argument("--models_dir", default="./Models/AdmetClassifier")
    p.add_argument("--ckpt_dir",   default="./ckpts/grownet_admet")
    p.add_argument(
        "--max_learners", type=int, default=16,
        help="Upper bound on auto-configured n_learners (analogous to max tree depth).",
    )
    p.add_argument(
        "--n_learners", type=int, default=None,
        help="Pin n_learners for every task (disables auto-sizing for this param).",
    )
    p.add_argument(
        "--hidden_dim", type=int, default=None,
        help="Pin hidden_dim for every task (disables auto-sizing for this param).",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Pin max epochs for every task (early stopping still applies).",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    train(
        variant=a.variant,
        models_dir=a.models_dir,
        ckpt_dir=a.ckpt_dir,
        max_learners=a.max_learners,
        n_learners=a.n_learners,
        hidden_dim=a.hidden_dim,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
    )
