"""
Evaluate per-task AttentionOracle ADMET classifiers on the SELFIES latent test set.

Loads the test latent pack produced by create_multitask_dataset_selfies.py
and each task's AttentionOracle checkpoint from ckpts/attention_admet/{variant}/,
then reports per-task and macro-averaged metrics — same table format as
evaluate_multitask_selfies.py and evaluate_grownet_selfies.py.

    python Models/AdmetClassifier/evaluate_attention_selfies.py --variant pretrain
    python Models/AdmetClassifier/evaluate_attention_selfies.py --variant akt1 --threshold 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        average_precision_score, confusion_matrix,
    )
except ImportError as exc:
    raise ImportError("Install scikit-learn: pip install scikit-learn") from exc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from attention_oracle import AttentionOracle


# ---------------------------------------------------------------------------
# Metrics (identical helper to other evaluate_*.py scripts)
# ---------------------------------------------------------------------------

def _per_task_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    pred = (probs >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "n":        int(len(labels)),
        "accuracy": float(accuracy_score(labels, pred)),
        "f1":       float(f1_score(labels, pred, zero_division=0)),
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    variant: str,
    models_dir: str  = "./Models/AdmetClassifier",
    ckpt_dir: str    = "./ckpts/attention_admet",
    threshold: float  = 0.5,
    temperature: float = 2.0,
    batch_size: int   = 1024,
) -> Dict[str, Dict[str, float]]:
    test_pt = Path(models_dir) / f"admet_latent_selfies_{variant}_test.pt"
    attn_dir = Path(ckpt_dir) / variant

    if not test_pt.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_pt}\n"
            "Run create_multitask_dataset_selfies.py first."
        )
    if not attn_dir.exists():
        raise FileNotFoundError(
            f"Attention checkpoint dir not found: {attn_dir}\n"
            "Run train_attention_admet.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = torch.load(test_pt, weights_only=False)
    tasks: List[str] = pack["tasks"]

    # Group test samples by task
    zs_by_task: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(tasks))}
    ys_by_task: Dict[int, List[float]]       = {i: [] for i in range(len(tasks))}
    for item in pack["data"]:
        t = item["task_idx"]
        zs_by_task[t].append(item["z"])
        ys_by_task[t].append(float(item["y"]))

    # Load AttentionOracle checkpoints
    attn_models: Dict[str, AttentionOracle] = {}
    for pt_path in sorted(attn_dir.glob("*.pt")):
        task_name = pt_path.stem
        if task_name in tasks:
            attn_models[task_name] = AttentionOracle.load(str(pt_path), device=device)
            attn_models[task_name].eval()

    print(
        f"\n=== AttentionOracle ADMET evaluation ({variant}) "
        f"@ threshold={threshold}, temperature={temperature} ==="
    )
    print(
        f"{'task':<22} {'n':>5} {'acc':>7} {'f1':>7} {'auroc':>7} "
        f"{'auprc':>7}  TN  FP  FN  TP"
    )

    macro_keys = ["accuracy", "f1", "roc_auc", "pr_auc"]
    macro_accum: Dict[str, List[float]] = {k: [] for k in macro_keys}
    results: Dict[str, Dict[str, float]] = {}

    for task_idx, task_name in enumerate(tasks):
        if task_name not in attn_models:
            print(f"  [skip] {task_name}: no checkpoint found in {attn_dir}")
            continue
        if not zs_by_task[task_idx]:
            print(f"  [skip] {task_name}: no test samples")
            continue

        model  = attn_models[task_name]
        zs_np  = np.stack(zs_by_task[task_idx]).astype(np.float32)
        ys_np  = np.array(ys_by_task[task_idx])

        all_probs: List[float] = []
        with torch.no_grad():
            for start in range(0, len(zs_np), batch_size):
                zb     = torch.tensor(zs_np[start : start + batch_size]).to(device)
                logits = model(zb)                                # (B, 1)
                probs  = torch.sigmoid(
                    torch.clamp(logits, -10.0, 10.0) / temperature
                ).squeeze(1)                                      # (B,)
                all_probs.extend(probs.cpu().tolist())

        probs_np = np.array(all_probs)
        m = _per_task_metrics(probs_np, ys_np.astype(int), threshold)
        results[task_name] = m

        for k in macro_keys:
            v = m[k]
            if not (isinstance(v, float) and np.isnan(v)):
                macro_accum[k].append(v)

        print(
            f"{task_name:<22} {m['n']:>5d} {m['accuracy']:>7.3f} {m['f1']:>7.3f} "
            f"{m['roc_auc']:>7.3f} {m['pr_auc']:>7.3f}  "
            f"{m['tn']:>3d} {m['fp']:>3d} {m['fn']:>3d} {m['tp']:>3d}"
        )

    macro = {k: float(np.mean(v)) if v else float("nan") for k, v in macro_accum.items()}
    results["_macro"] = macro
    print(f"\n-- macro averages across {len(results) - 1} tasks --")
    for k, v in macro.items():
        print(f"  {k:<10} {v:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate per-task AttentionOracle classifiers.")
    p.add_argument("--variant",     choices=["pretrain", "akt1"], required=True)
    p.add_argument("--models_dir",  default="./Models/AdmetClassifier")
    p.add_argument("--ckpt_dir",    default="./ckpts/attention_admet")
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Sigmoid temperature — match the value used at walk time.")
    p.add_argument("--batch_size",  type=int,   default=1024)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(
        variant=a.variant,
        models_dir=a.models_dir,
        ckpt_dir=a.ckpt_dir,
        threshold=a.threshold,
        temperature=a.temperature,
        batch_size=a.batch_size,
    )
