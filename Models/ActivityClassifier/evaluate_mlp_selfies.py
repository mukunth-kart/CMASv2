"""
Evaluate the AKT1 activity MLP trained on SELFIES latents.

Produces standard binary-classification diagnostics at the requested
sigmoid threshold (default 0.5): accuracy, precision/recall/F1,
ROC-AUC, PR-AUC, and the confusion matrix.

By default, holds out the last `--test_frac` of the
`latent_dataset_selfies.pt` as the test slice (shuffled with a fixed
seed for reproducibility).  Alternatively, `--test_pt` points at a
separate test pack with the same schema `{"z": [N,128], "y": [N,1]}`.

    python Models/ActivityClassifier/evaluate_mlp_selfies.py
    python Models/ActivityClassifier/evaluate_mlp_selfies.py --threshold 0.4
    python Models/ActivityClassifier/evaluate_mlp_selfies.py --test_pt path/to/akt1_test.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
except ImportError as exc:
    raise ImportError("Install scikit-learn: pip install scikit-learn") from exc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from train_mlp_selfies import LatentPredictor  # noqa: E402


DEFAULT_DATA_PT  = "Models/ActivityClassifier/latent_dataset_selfies.pt"
DEFAULT_MODEL_PT = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp_selfies.pt"


def _load_split(data_pt: str, test_pt: str | None, test_frac: float, seed: int):
    """Return (z_test, y_test) tensors on CPU."""
    if test_pt:
        pack = torch.load(test_pt, map_location="cpu")
        return pack["z"], pack["y"]

    pack = torch.load(data_pt, map_location="cpu")
    z, y = pack["z"], pack["y"]
    n = len(z)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * (1.0 - test_frac))
    idx = perm[cut:]
    return z[idx], y[idx]


def evaluate(
    data_pt: str = DEFAULT_DATA_PT,
    model_pt: str = DEFAULT_MODEL_PT,
    test_pt: str | None = None,
    test_frac: float = 0.2,
    seed: int = 42,
    threshold: float = 0.5,
    batch_size: int = 512,
) -> dict:
    if not Path(model_pt).exists():
        raise FileNotFoundError(f"Missing trained model: {model_pt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z, y = _load_split(data_pt, test_pt, test_frac, seed)
    print(f"Test set: {len(z)} samples  |  positives: {int(y.sum())}  "
          f"negatives: {int(y.numel() - y.sum())}")

    model = LatentPredictor(input_dim=z.shape[-1]).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=device))
    model.eval()

    loader = DataLoader(TensorDataset(z, y), batch_size=batch_size, shuffle=False)

    probs_all, y_all = [], []
    with torch.no_grad():
        for bz, by in loader:
            logits = model(bz.to(device)).cpu()
            probs_all.append(torch.sigmoid(logits))
            y_all.append(by)

    probs = torch.cat(probs_all).squeeze(-1).numpy()
    y_true = torch.cat(y_all).squeeze(-1).numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = {
        "n":         int(len(y_true)),
        "threshold": threshold,
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc":    float(average_precision_score(y_true, probs)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    print("\n=== Activity MLP (SELFIES) evaluation ===")
    for k, v in report.items():
        if k == "confusion":
            c = v
            print(f"  confusion  TN={c['tn']:5d}  FP={c['fp']:5d}  "
                  f"FN={c['fn']:5d}  TP={c['tp']:5d}")
        elif isinstance(v, float):
            print(f"  {k:<10} {v:.4f}")
        else:
            print(f"  {k:<10} {v}")

    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_pt",   default=DEFAULT_DATA_PT,
                   help="Full activity latent pack; used when --test_pt absent.")
    p.add_argument("--model_pt",  default=DEFAULT_MODEL_PT)
    p.add_argument("--test_pt",   default=None,
                   help="Optional dedicated test pack {z, y}.")
    p.add_argument("--test_frac", type=float, default=0.2,
                   help="Holdout fraction when splitting --data_pt.")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int,  default=512)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(
        data_pt=a.data_pt, model_pt=a.model_pt, test_pt=a.test_pt,
        test_frac=a.test_frac, seed=a.seed,
        threshold=a.threshold, batch_size=a.batch_size,
    )
