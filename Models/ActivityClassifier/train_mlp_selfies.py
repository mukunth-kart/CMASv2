"""
Train the activity (pIC50) MLP on the SELFIES latent dataset.

Reuses the `LatentPredictor` architecture from the existing
`train_mlp.py` — which is a latent-only MLP and therefore compatible
with SELFIES encodings unchanged — while using SELFIES-specific paths
for the input latent pack and the output checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))



DATA_PATH       = "Models/ActivityClassifier/latent_dataset_selfies.pt"
SAVE_MODEL_PATH = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp_selfies.pt"
LATENT_DIM  = 128
BATCH_SIZE  = 32
EPOCHS      = 500
LEARNING_RATE = 1e-3

class LatentPredictor(nn.Module):
    def __init__(self, input_dim=128):
        super(LatentPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
            # Sigmoid removed here, moved to loss function/inference
        )

    def forward(self, z):
        return self.net(z)

def train() -> None:
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. "
            "Run create_activity_latent_selfies.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    data = torch.load(DATA_PATH)
    X, y = data["z"].to(device), data["y"].to(device)
    print(f"Loaded {len(X)} samples")

    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    model = LatentPredictor(input_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # dynamic pos_weight from actual class counts
    pos = y.sum().item()
    neg = y.numel() - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for bz, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bz), by)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total/len(loader):.4f}")

    Path(SAVE_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Saved activity MLP -> {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    train()
