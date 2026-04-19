"""
pcgrad_epsilon_greedy_selfies.py
=================================
ε-Greedy PCGrad latent-space walk (7.3).

At each step:
  * With probability ε: take a random Gaussian perturbation of z
    (exploration — escapes local minima and over-optimised regions).
  * With probability 1-ε: take the standard PCGrad gradient step
    (exploitation — greedy optimisation towards all thresholds).

Inspired by ε-greedy exploration in Reinforcement Learning.

CLI additions over pcgrad_only_selfies.py:
    --epsilon      (default 0.10)  probability of random step
    --noise_scale  (default 0.05)  std of the Gaussian perturbation
"""

from __future__ import annotations

import csv
import logging
import os
import random
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


# ---------------------------------------------------------------------------
# Gradient / PCGrad (unchanged from pcgrad_only_selfies.py)
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
# Random (ε) step — forward pass only, no gradients
# ---------------------------------------------------------------------------
def _random_step(
    z: torch.Tensor,
    engine: ScoringEngine,
    device: torch.device,
    noise_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    z_new = z.detach() + torch.randn_like(z) * noise_scale
    z_new = z_new.to(device)
    with torch.no_grad():
        raw = engine.get_all_scores(z_new.clone())
    scores = {p: _to_float(raw[p]) for p in TASKS if p in raw}
    return z_new, scores


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------
def pcgrad_epsilon_walk(
    vae: SelfiesVAE,
    engine: ScoringEngine,
    max_steps: int = 50_000,
    target_count: int = 0,
    lr: float = 0.01,
    lambda_prior: float = 3.0,
    latent_bounds: Tuple[float, float] = (-3.0, 3.0),
    log_every: int = 100,
    epsilon: float = 0.10,
    noise_scale: float = 0.05,
    out_file: str = "outputs/pcgrad_eps/passing_smiles_pcgrad_eps_selfies.csv",
) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  epsilon={epsilon}  noise_scale={noise_scale}")

    try:
        _, z = vae.encode_molecule(GOLDEN_SCAFFOLD)
        z = z.to(device)
    except Exception as exc:
        logger.error(f"Encode failed ({exc}); falling back to random z.")
        _, z = vae.generate_molecule()
        z = z.to(device)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    cols = ["smiles", "step", "step_type"] + list(TASKS.keys())
    passing, seen = [], set()
    n_random, n_greedy = 0, 0

    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for step in range(max_steps):
            if random.random() < epsilon:
                z, scores = _random_step(z, engine, device, noise_scale)
                step_type = "random"
                n_random += 1
            else:
                z, scores = _pcgrad_step(z, engine, device, lr, lambda_prior)
                step_type = "greedy"
                n_greedy += 1

            z = torch.clamp(z, latent_bounds[0], latent_bounds[1])

            try:
                smi, _ = vae.generate_molecule(z=z)
            except Exception:
                smi = None

            if _valid_smiles(smi) and _passes(scores) and smi not in seen:
                seen.add(smi)
                row = {"smiles": smi, "step": step, "step_type": step_type}
                row.update({p: round(_to_float(scores[p]), 6)
                            for p in TASKS if p in scores})
                w.writerow(row)
                f.flush()
                passing.append(row)
                logger.info(
                    f"  PASS #{len(passing):4d}  step={step} ({step_type})  -> {smi}"
                )

            if step % log_every == 0:
                logger.info(
                    f"  step={step:5d}  found={len(passing)}"
                    f"  greedy={n_greedy}  random={n_random}"
                )

            if target_count > 0 and len(passing) >= target_count:
                break

    logger.info(f"DONE  steps={step+1}  passing={len(passing)}  -> {out_file}")
    logger.info(f"  greedy steps: {n_greedy}  random steps: {n_random}")
    return passing


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--max_steps",    type=int,   default=50_000)
        p.add_argument("--target_count", type=int,   default=0)
        p.add_argument("--lr",           type=float, default=0.01)
        p.add_argument("--lambda_prior", type=float, default=3.0)
        p.add_argument("--epsilon",      type=float, default=0.10,
                       help="Probability of taking a random step instead of PCGrad.")
        p.add_argument("--noise_scale",  type=float, default=0.05,
                       help="Std of Gaussian noise for the random step.")
        p.add_argument("--log_every",    type=int,   default=100)
        p.add_argument("--out_file",     type=str,
                       default="outputs/pcgrad_eps/passing_smiles_pcgrad_eps_selfies.csv")
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

    pcgrad_epsilon_walk(
        vae=vae,
        engine=engine,
        max_steps=a.max_steps,
        target_count=a.target_count,
        lr=a.lr,
        lambda_prior=a.lambda_prior,
        log_every=a.log_every,
        epsilon=a.epsilon,
        noise_scale=a.noise_scale,
        out_file=a.out_file,
    )
