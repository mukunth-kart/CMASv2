"""
Evaluate the multitask ADMET classifier trained on SELFIES latents.

Loads the test latent pack produced by `create_multitask_dataset_selfies.py`
(default: `admet_latent_selfies_{variant}_test.pt`) and the trained
predictor (`admet_predictor_selfies_{variant}.pt`), then reports per-task
and macro-averaged metrics (accuracy, F1, ROC-AUC, PR-AUC) plus each
task's confusion matrix at the chosen threshold.

    python Models/AdmetClassifier/evaluate_multitask_selfies.py --variant pretrain
    python Models/AdmetClassifier/evaluate_multitask_selfies.py --variant akt1 --threshold 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        average_precision_score, confusion_matrix,
    )
except ImportError as exc:
    raise ImportError("Install scikit-learn: pip install scikit-learn") from exc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from train_multitask_selfies import MultiHeadADMET, LatentDataset  # noqa: E402


def _per_task_metrics(
    probs: np.ndarray,     # (N,) probabilities for this task
    labels: np.ndarray,    # (N,) {0,1} labels
    threshold: float,
) -> Dict[str, float]:
    pred = (probs >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "n":         int(len(labels)),
        "accuracy":  float(accuracy_score(labels, pred)),
        "f1":        float(f1_score(labels, pred, zero_division=0)),
    }
    if len(set(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        metrics["pr_auc"]  = float(average_precision_score(labels, probs))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"]  = float("nan")

    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    metrics["tn"], metrics["fp"] = int(tn), int(fp)
    metrics["fn"], metrics["tp"] = int(fn), int(tp)
    return metrics


def evaluate(
    variant: str,
    models_dir: str = "./Models/AdmetClassifier",
    threshold: float = 0.5,
    batch_size: int = 1024,
    temperature: float = 2.0,
) -> Dict[str, Dict[str, float]]:
    models_dir = Path(models_dir)
    test_pt  = models_dir / f"admet_latent_selfies_{variant}_test.pt"
    model_pt = models_dir / f"admet_predictor_selfies_{variant}.pt"

    if not test_pt.exists():
        raise FileNotFoundError(f"Missing test pack: {test_pt}")
    if not model_pt.exists():
        raise FileNotFoundError(f"Missing trained predictor: {model_pt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LatentDataset(str(test_pt))
    tasks: List[str] = dataset.tasks
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = MultiHeadADMET(
        latent_dim=dataset.latent_dim, num_tasks=len(tasks)
    ).to(device)
    ckpt = torch.load(model_pt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Collect per-task probabilities and labels
    probs_by_task: Dict[int, List[float]] = {i: [] for i in range(len(tasks))}
    labels_by_task: Dict[int, List[int]]  = {i: [] for i in range(len(tasks))}

    with torch.no_grad():
        for z, y, task_ids in loader:
            z = z.to(device)
            logits = model(z)                                # (B, T)
            logits = torch.clamp(logits, -10.0, 10.0)
            probs = torch.sigmoid(logits / temperature)      # match runtime scaling
            idx = torch.arange(z.size(0), device=device)
            target_probs = probs[idx, task_ids.to(device)]   # (B,)
            for p, lbl, t in zip(target_probs.cpu().tolist(),
                                 y.tolist(), task_ids.tolist()):
                probs_by_task[t].append(p)
                labels_by_task[t].append(int(lbl))

    # Per-task and macro metrics
    results: Dict[str, Dict[str, float]] = {}
    macro_keys = ["accuracy", "f1", "roc_auc", "pr_auc"]
    macro_accum = {k: [] for k in macro_keys}

    print(f"\n=== ADMET multitask ({variant}) evaluation "
          f"@ threshold={threshold}, temperature={temperature} ===")
    print(f"{'task':<22} {'n':>5} {'acc':>7} {'f1':>7} {'auroc':>7} "
          f"{'auprc':>7}  TN  FP  FN  TP")

    for i, task in enumerate(tasks):
        if not probs_by_task[i]:
            continue
        p = np.array(probs_by_task[i])
        l = np.array(labels_by_task[i])
        m = _per_task_metrics(p, l, threshold)
        results[task] = m

        for k in macro_keys:
            v = m[k]
            if not (isinstance(v, float) and np.isnan(v)):
                macro_accum[k].append(v)

        print(f"{task:<22} {m['n']:>5d} {m['accuracy']:>7.3f} {m['f1']:>7.3f} "
              f"{m['roc_auc']:>7.3f} {m['pr_auc']:>7.3f}  "
              f"{m['tn']:>3d} {m['fp']:>3d} {m['fn']:>3d} {m['tp']:>3d}")

    macro = {k: float(np.mean(v)) if v else float("nan")
             for k, v in macro_accum.items()}
    results["_macro"] = macro

    print(f"\n-- macro averages across {len(tasks)} tasks --")
    for k, v in macro.items():
        print(f"  {k:<10} {v:.4f}")

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",   choices=["pretrain", "akt1"], required=True)
    p.add_argument("--models_dir", default="./Models/AdmetClassifier")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Sigmoid temperature — must match the value used in "
                        "utils.ADMETClassifier at inference time (default 2.0).")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(
        variant=a.variant, models_dir=a.models_dir,
        threshold=a.threshold, batch_size=a.batch_size,
        temperature=a.temperature,
    )
