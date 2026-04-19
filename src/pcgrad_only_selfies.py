"""
pcgrad_only_selfies.py
======================
Self-contained PCGrad latent-space walk over the SELFIES VAE.

Finds SMILES that pass all ADMET thresholds by optimising the VAE latent
`z` with pure PCGrad (Yu et al. 2020) — per-task gradients are
conflict-resolved by projecting out components anti-aligned with
higher-priority gradients, then summed with equal weight and applied to
`z`.

Sign convention (minimisation frame)
    high target (potency, Caco2, HLM, RLM)  →  loss = -score
    low  target (hERG, CYPs, BBBP, P-gp)    →  loss = +score

Output
------
passing_smiles_pcgrad_selfies.csv — one row per molecule that clears all
thresholds, with columns: smiles, step, <all property scores>.
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
# Constants
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


def _log_step(step: int, scores: Dict[str, float], n_found: int) -> None:
    lines = [f"  step={step:5d}  found={n_found}"]
    for prop in TASKS:
        if prop not in scores:
            continue
        v   = _to_float(scores[prop])
        thr = THRESHOLDS[prop]
        ok  = (v >= thr) if TASKS[prop] == "high" else (v <= thr)
        lines.append(f"    {prop:<22}  score={v:.3f}  thr={'+' if ok else '-'}{thr:.2f}")
    logger.info("\n".join(lines))


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
# PCGrad conflict resolution
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


def _pcgrad_step(
    z: torch.Tensor,
    engine: ScoringEngine,
    device: torch.device,
    lr: float,
    lambda_prior: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    grads, scores = _compute_task_gradients(z, engine, device)
    props = [p for p in TASKS if p in grads]
    if not props:
        return z, scores
    clean = _pcgrad_resolve(grads, props)
    update_dir = torch.stack(clean).sum(dim=0)
    final = torch.clamp(update_dir + lambda_prior * z.detach().clone(), -0.1, 0.1)
    return z.detach() - lr * final, scores


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------
def pcgrad_walk(
    vae: SelfiesVAE,
    engine: ScoringEngine,
    max_steps: int = 50_000,
    target_count: int = 0,
    lr: float = 0.01,
    lambda_prior: float = 3.0,
    latent_bounds: Tuple[float, float] = (-3.0, 3.0),
    log_every: int = 100,
    out_file: str = "outputs/pcgrad/passing_smiles_pcgrad_selfies.csv",
) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  Scaffold: {GOLDEN_SCAFFOLD}")

    try:
        _, z = vae.encode_molecule(GOLDEN_SCAFFOLD)
        z = z.to(device)
    except Exception as exc:
        logger.error(f"Encode failed ({exc}); falling back to random z.")
        _, z = vae.generate_molecule()
        z = z.to(device)

    cols = ["smiles", "step", "valid", "passes", "unique"] + list(TASKS.keys())
    passing, seen = [], set()

    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for step in range(max_steps):
            z, scores = _pcgrad_step(z, engine, device, lr, lambda_prior)
            z = torch.clamp(z, latent_bounds[0], latent_bounds[1])

            try:
                smi, _ = vae.generate_molecule(z=z)
            except Exception:
                smi = None

            if smi is None:
                if step % log_every == 0:
                    _log_step(step, scores, len(passing))
                continue

            is_valid  = _valid_smiles(smi)
            is_passes = is_valid and _passes(scores)
            is_unique = smi not in seen

            if is_unique:
                seen.add(smi)

            row = {"smiles": smi, "step": step,
                   "valid": is_valid, "passes": is_passes, "unique": is_unique}
            row.update({p: round(_to_float(scores[p]), 6) for p in TASKS if p in scores})
            w.writerow(row)

            if is_passes and is_unique:
                f.flush()
                passing.append(row)
                logger.info(f"  PASS #{len(passing):4d}  step={step}  -> {smi}")

            if step % log_every == 0:
                _log_step(step, scores, len(passing))

            if target_count > 0 and len(passing) >= target_count:
                break

    logger.info(f"DONE  steps={step+1}  passing={len(passing)}  -> {out_file}")
    return passing


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_property_config(
        os.environ.get("SELFIES_PATHS_YAML", "configs/paths_selfies.yaml")
    )

    vae = SelfiesVAE(model_path=cfg["selfies_vae_model_path"])
    vae.load_model(vocab_base=cfg["selfies_vocab_path"])

    engine = ScoringEngine(
        activity_classifier_path=cfg["activity_classifier_model_path"],
        admet_model_path=cfg["admet_model_path"],
    )
    from utils.GrowNetADMETClassifier import GrowNetScoringEngine
    engine = GrowNetScoringEngine(
        activity_classifier_path=cfg["activity_classifier_model_path"],
        grownet_ckpt_dir="./ckpts/grownet_admet",
        variant="pretrain",
    )

    from utils.AttentionADMETClassifier import AttentionScoringEngine
    engine = AttentionScoringEngine(
        activity_classifier_path=cfg["activity_classifier_model_path"],
        attention_ckpt_dir="./ckpts/attention_admet",
        variant="pretrain",   # or "pretrain"
    )


    pcgrad_walk(vae=vae, engine=engine)
