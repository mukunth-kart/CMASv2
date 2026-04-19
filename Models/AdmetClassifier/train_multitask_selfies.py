"""
Train the multitask ADMET classifier on a SELFIES-encoded latent dataset.

Reuses `MultiHeadADMET` and `LatentDataset` from the existing
`train_multitask.py` (they operate on latent vectors and are therefore
alphabet-agnostic), but keeps the training loop self-contained here so
paths and a `--variant` flag can be injected cleanly without editing
the original file.

    python Models/AdmetClassifier/train_multitask_selfies.py --variant pretrain
    python Models/AdmetClassifier/train_multitask_selfies.py --variant akt1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

class MultiHeadADMET(nn.Module):
    def __init__(self, latent_dim=128, num_tasks=11):
        super().__init__()
        # Wider shared body for better chemical feature extraction
        self.shared_encoder = nn.Sequential(
            nn.Linear(latent_dim, 1024), 
            nn.BatchNorm1d(1024),
            nn.SiLU(), 
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU()
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 1)
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, z):
        features = self.shared_encoder(z)
        outputs = [head(features) for head in self.heads]
        return torch.cat(outputs, dim=1)

class LatentDataset(Dataset):
    def __init__(self, processed_path):
        data_pack = torch.load(processed_path, weights_only=False)
        self.samples = data_pack['data']
        self.tasks = data_pack['tasks']
        self.latent_dim = data_pack['latent_dim']
        print(f"✅ Loaded {len(self.samples)} samples.")
        print(f"📋 Task Order: {self.tasks}") # IMPORTANT: Remember this order!

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.tensor(item['z'], dtype=torch.float32), 
            torch.tensor(item['y'], dtype=torch.float32),
            torch.tensor(item['task_idx'], dtype=torch.long)
        )
 

def train(variant: str,
          epochs: int = 1000,
          batch_size: int = 1024,
          lr: float = 1e-3,
          models_dir: str = "./Models/AdmetClassifier") -> None:

    models_dir = Path(models_dir)
    train_pt  = models_dir / f"admet_latent_selfies_{variant}_train.pt"
    save_path = models_dir / f"admet_predictor_selfies_{variant}.pt"

    if not train_pt.exists():
        raise FileNotFoundError(
            f"Expected latent dataset at {train_pt}. "
            "Run create_multitask_dataset_selfies.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LatentDataset(str(train_pt))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultiHeadADMET(
        latent_dim=dataset.latent_dim,
        num_tasks=len(dataset.tasks),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))

    print(f"Training ADMET multitask ({variant}) — {len(dataset)} samples, "
          f"{len(dataset.tasks)} tasks")

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for z, y, task_ids in loader:
            z, y, task_ids = z.to(device), y.to(device), task_ids.to(device)
            optimizer.zero_grad()
            preds = model(z)
            tgt = preds[torch.arange(z.size(0)), task_ids]
            smooth_y = y * 0.9 + 0.05  # label smoothing
            loss = criterion(tgt, smooth_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / max(1, len(loader))
        scheduler.step(avg)

        if (epoch + 1) % 5 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:04d} | Loss: {avg:.4f} | LR: {lr_now:.6f}")
            if lr_now < 1e-6:
                print("LR hit floor — stopping early.")
                break

    torch.save(
        {"model_state": model.state_dict(), "task_names": dataset.tasks},
        save_path,
    )
    print(f"Saved ADMET predictor -> {save_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["pretrain", "akt1"], required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    train(variant=a.variant, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr)
