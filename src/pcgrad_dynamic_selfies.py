"""
pcgrad_dynamic_selfies.py
=========================
PCGrad latent-space walk with dynamic task-priority weights (7.2).

Extends pcgrad_only_selfies.py with a deadlock-breaking mechanism:

  * At each step, compute per-task "gap" = distance from passing threshold.
  * Outside deadlock: weight each task's gradient proportionally to its gap
    (farther from threshold → larger gradient contribution).
  * Near-deadlock (>= deadlock_ratio fraction of tasks already passing):
    lagging tasks get `boost_factor` weight; passed tasks get 0.1 to
    maintain their scores without dominating.
  * Scaled gradients enter PCGrad conflict resolution unchanged — so the
    projection step still removes destructive interference.

CLI additions over pcgrad_only_selfies.py:
    --boost_factor    (default 8.0)   scale for lagging tasks in deadlock
    --deadlock_ratio  (default 0.75)  pass-rate threshold to trigger boost
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from rdkit import Chem

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Generators.SelfiesVAE import SelfiesVAE
from utils.ScoringEngine import ScoringEngine
from utils.utils import load_property_config


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (identical to pcgrad_only_selfies.py)
# ---------------------------------------------------------------------------
GOLDEN_SCAFFOLD = "O=C(Cc1nc(N2CCOCC2)cc(=O)[nH]1)N1CCc2ccccc21"

TASKS: Dict[str, str] = {
    "potency":            "high",
    "hERG_inhibition":    "low",
    "CYP3A4_inhibition":  "low",
    "BBBP":               "low",
    "Caco2_permeability": "high",
    "HLM_stability":      "high",
    "RLM_stability":      "high",
    "P-gp_substrate":     "low",
    "CYP1A2_inhibition":  "low",
    "CYP2C9_inhibition":  "low",
    "CYP2C19_inhibition": "low",
    "CYP2D6_inhibition":  "low",
}

THRESHOLDS: Dict[str, float] = {
    "potency":            0.5,
    "hERG_inhibition":    0.3,
    "CYP3A4_inhibition":  0.4,
    "BBBP":               0.3,
    "Caco2_permeability": 0.6,
    "HLM_stability":      0.6,
    "RLM_stability":      0.6,
    "P-gp_substrate":     0.4,
    "CYP1A2_inhibition":  0.4,
    "CYP2C9_inhibition":  0.4,
    "CYP2C19_inhibition": 0.4,
    "CYP2D6_inhibition":  0.4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_float(v) -> float:
    return v.item() if isinstance(v, torch.Tensor) else float(v)


def _valid_smiles(smi: Optional[str]) -> bool:
    return bool(smi) and Chem.MolFromSmiles(smi) is not None


def _passes(scores: Dict[str, float]) -> bool:
    for prop, thr in THRESHOLDS.items():
        if prop not in scores:
            continue  # task not returned by this engine — skip rather than fail
        v = _to_float(scores[prop])
        if TASKS[prop] == "high" and v < thr:
            return False
        if TASKS[prop] == "low"  and v > thr:
            return False
    return True


def _score_to_loss(prop: str, score: torch.Tensor) -> torch.Tensor:
    return -score if TASKS[prop] == "high" else score


def _log_step(step: int, scores: Dict[str, float], n_found: int,
              weights: Dict[str, float]) -> None:
    lines = [f"  step={step:5d}  found={n_found}"]
    for prop in TASKS:
        if prop not in scores:
            continue
        v   = _to_float(scores[prop])
        thr = THRESHOLDS[prop]
        ok  = (v >= thr) if TASKS[prop] == "high" else (v <= thr)
        w   = weights.get(prop, 1.0)
        lines.append(
            f"    {prop:<22}  score={v:.3f}  thr={'+' if ok else '-'}{thr:.2f}"
            f"  w={w:.2f}"
        )
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Dynamic priority weights
# ---------------------------------------------------------------------------
def _priority_weights(
    scores: Dict[str, float],
    boost_factor: float,
    deadlock_ratio: float,
) -> Dict[str, float]:
    """
    Scale each task's gradient by how far it is from passing.

    In near-deadlock (pass_ratio >= deadlock_ratio):
        lagging  → boost_factor
        passed   → 0.1  (maintain without dominating)
    Otherwise:
        weight = 1 + 5 * gap  (smooth proportional scaling)
    """
    gaps: Dict[str, float] = {}
    n_passed = 0
    for prop in TASKS:
        if prop not in scores:
            gaps[prop] = 0.0
            continue
        v = _to_float(scores[prop])
        thr = THRESHOLDS[prop]
        gap = max(0.0, thr - v) if TASKS[prop] == "high" else max(0.0, v - thr)
        gaps[prop] = gap
        if gap == 0.0:
            n_passed += 1

    pass_ratio = n_passed / max(1, len(TASKS))
    deadlock = pass_ratio >= deadlock_ratio

    weights: Dict[str, float] = {}
    for prop in TASKS:
        g = gaps.get(prop, 0.0)
        if deadlock:
            weights[prop] = boost_factor if g > 0.0 else 0.1
        else:
            weights[prop] = 1.0 + 5.0 * g
    return weights


# ---------------------------------------------------------------------------
# Per-task gradient computation
# ---------------------------------------------------------------------------
def _compute_task_gradients(
    z: torch.Tensor,
    engine: ScoringEngine,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    grads, scores = {}, {}
    for prop in TASKS:
        z_var = z.clone().detach().to(device).requires_grad_(True)
        all_scores = engine.get_all_scores(z_var)
        if prop not in all_scores:
            continue
        s = all_scores[prop]
        scores[prop] = _to_float(s)
        _score_to_loss(prop, s).backward()
        grads[prop] = z_var.grad.clone().detach()
    return grads, scores


# ---------------------------------------------------------------------------
# PCGrad conflict resolution (unchanged from pcgrad_only_selfies.py)
# ---------------------------------------------------------------------------
def _pcgrad_resolve(
    grads: Dict[str, torch.Tensor],
    props: List[str],
) -> List[torch.Tensor]:
    resolved: List[torch.Tensor] = []
    for i, prop in enumerate(props):
        g = grads[prop].clone()
        for j in range(i):
            g_prev = resolved[j]
            gf, gpf = g.flatten(), g_prev.flatten()
            dot = torch.dot(gf, gpf)
            if dot < 0:
                gp_sq = torch.dot(gpf, gpf) + 1e-8
                g = g - (dot / gp_sq) * g_prev
        resolved.append(g)
    return resolved


# ---------------------------------------------------------------------------
# Dynamic-PCGrad step
# ---------------------------------------------------------------------------
def _pcgrad_dynamic_step(
    z: torch.Tensor,
    engine: ScoringEngine,
    device: torch.device,
    lr: float,
    lambda_prior: float,
    boost_factor: float,
    deadlock_ratio: float,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
    grads, scores = _compute_task_gradients(z, engine, device)
    props = [p for p in TASKS if p in grads]
    if not props:
        return z, scores, {}

    weights = _priority_weights(scores, boost_factor, deadlock_ratio)

    # Scale each gradient by its priority weight before conflict resolution
    scaled = {p: grads[p] * weights.get(p, 1.0) for p in props}
    clean  = _pcgrad_resolve(scaled, props)
    update_dir = torch.stack(clean).sum(dim=0)
    final = torch.clamp(update_dir + lambda_prior * z.detach().clone(), -0.1, 0.1)
    return z.detach() - lr * final, scores, weights


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------
def pcgrad_dynamic_walk(
    vae: SelfiesVAE,
    engine: ScoringEngine,
    max_steps: int = 50_000,
    target_count: int = 0,
    lr: float = 0.01,
    lambda_prior: float = 3.0,
    latent_bounds: Tuple[float, float] = (-3.0, 3.0),
    log_every: int = 100,
    boost_factor: float = 8.0,
    deadlock_ratio: float = 0.75,
    out_file: str = "outputs/pcgrad_dynamic/passing_smiles_pcgrad_dynamic_selfies.csv",
) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  boost={boost_factor}  deadlock_ratio={deadlock_ratio}")

    try:
        _, z = vae.encode_molecule(GOLDEN_SCAFFOLD)
        z = z.to(device)
    except Exception as exc:
        logger.error(f"Encode failed ({exc}); falling back to random z.")
        _, z = vae.generate_molecule()
        z = z.to(device)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    cols = ["smiles", "step"] + list(TASKS.keys())
    passing, seen = [], set()

    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for step in range(max_steps):
            z, scores, weights = _pcgrad_dynamic_step(
                z, engine, device, lr, lambda_prior, boost_factor, deadlock_ratio
            )
            z = torch.clamp(z, latent_bounds[0], latent_bounds[1])

            try:
                smi, _ = vae.generate_molecule(z=z)
            except Exception:
                smi = None

            if _valid_smiles(smi) and _passes(scores) and smi not in seen:
                seen.add(smi)
                row = {"smiles": smi, "step": step}
                row.update({p: round(_to_float(scores[p]), 6)
                            for p in TASKS if p in scores})
                w.writerow(row)
                f.flush()
                passing.append(row)
                logger.info(f"  PASS #{len(passing):4d}  step={step}  -> {smi}")

            if step % log_every == 0:
                _log_step(step, scores, len(passing), weights)

            if target_count > 0 and len(passing) >= target_count:
                break

    logger.info(f"DONE  steps={step+1}  passing={len(passing)}  -> {out_file}")
    return passing


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--max_steps",     type=int,   default=50_000)
        p.add_argument("--target_count",  type=int,   default=0)
        p.add_argument("--lr",            type=float, default=0.01)
        p.add_argument("--lambda_prior",  type=float, default=3.0)
        p.add_argument("--boost_factor",  type=float, default=8.0,
                       help="Weight multiplier for lagging tasks in deadlock.")
        p.add_argument("--deadlock_ratio",type=float, default=0.75,
                       help="Pass-rate fraction that triggers deadlock mode.")
        p.add_argument("--log_every",     type=int,   default=100)
        p.add_argument("--out_file",      type=str,
                       default="outputs/pcgrad_dynamic/passing_smiles_pcgrad_dynamic_selfies.csv")
        return p.parse_args()

    a = parse_args()
    cfg = load_property_config(
        os.environ.get("SELFIES_PATHS_YAML", "configs/paths_selfies.yaml")
    )

    vae = SelfiesVAE(model_path=cfg["selfies_vae_model_path"])
    vae.load_model(vocab_base=cfg["selfies_vocab_path"])

    engine = ScoringEngine(
        activity_classifier_path=cfg["activity_classifier_model_path"],
        admet_model_path=cfg["admet_model_path"],
    )

    pcgrad_dynamic_walk(
        vae=vae,
        engine=engine,
        max_steps=a.max_steps,
        target_count=a.target_count,
        lr=a.lr,
        lambda_prior=a.lambda_prior,
        log_every=a.log_every,
        boost_factor=a.boost_factor,
        deadlock_ratio=a.deadlock_ratio,
        out_file=a.out_file,
    )
