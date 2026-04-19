"""
nash_epsilon_greedy_selfies.py
===============================
ε-Greedy Nash-MTL latent-space walk (7.3).

At each step:
  * With probability ε: take a random Gaussian perturbation of z
    (exploration — escapes local minima and over-optimised regions).
  * With probability 1-ε: take the standard Nash Bargaining MTL step
    (exploitation — fair-weighted gradient update across all tasks).

Inspired by ε-greedy exploration in Reinforcement Learning.

CLI additions over nash_mtl_walk_selfies.py:
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

import numpy as np
import torch
from rdkit import Chem

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError("Nash-MTL needs cvxpy: pip install cvxpy") from exc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Generators.SelfiesVAE import SelfiesVAE
from utils.ScoringEngine import ScoringEngine
from utils.utils import load_property_config


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (identical to nash_mtl_walk_selfies.py)
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


def _score_to_loss(prop: str, s: torch.Tensor) -> torch.Tensor:
    return -s if TASKS[prop] == "high" else s


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
# Nash solver (unchanged from nash_mtl_walk_selfies.py)
# ---------------------------------------------------------------------------
class NashSolver:
    def __init__(self, n_tasks: int, optim_niter: int = 20):
        self.n_tasks = n_tasks
        self.optim_niter = optim_niter
        self.prvs_alpha = np.ones(n_tasks, dtype=np.float64)
        self.normalization_factor = np.ones((1,))
        self._init_optim()

    def _init_optim(self) -> None:
        self.alpha   = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_p = cp.Parameter(shape=(self.n_tasks,), value=self.prvs_alpha)
        self.G_p     = cp.Parameter(shape=(self.n_tasks, self.n_tasks),
                                    value=np.eye(self.n_tasks))
        self.norm_p  = cp.Parameter(shape=(1,), value=np.array([1.0]))

        G_prvs  = self.G_p @ self.prvs_alpha_p
        phi_tag = 1.0 / self.prvs_alpha_p + (1.0 / G_prvs) @ self.G_p
        self.phi_alpha = phi_tag @ (self.alpha - self.prvs_alpha_p)

        G_alpha     = self.G_p @ self.alpha
        constraints = [
            -cp.log(self.alpha[i] * self.norm_p) - cp.log(G_alpha[i]) <= 0
            for i in range(self.n_tasks)
        ]
        obj = cp.Minimize(cp.sum(G_alpha) + self.phi_alpha / self.norm_p)
        self.prob = cp.Problem(obj, constraints)

    def _converged(self, gtg: np.ndarray, a: np.ndarray) -> bool:
        if self.alpha.value is None:
            return True
        residual = gtg @ a - 1.0 / (a + 1e-10)
        if np.linalg.norm(residual) < 1e-3:
            return True
        return np.linalg.norm(self.alpha.value - self.prvs_alpha_p.value) < 1e-6

    def solve(self, gtg: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(gtg)
        if n < 1e-12:
            return self.prvs_alpha.copy()
        self.normalization_factor = np.array([n])
        self.G_p.value    = gtg / n
        self.norm_p.value = self.normalization_factor

        a = self.prvs_alpha.copy()
        for _ in range(self.optim_niter):
            self.alpha.value      = a
            self.prvs_alpha_p.value = a
            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except cp.SolverError:
                self.alpha.value = self.prvs_alpha_p.value
            if self._converged(self.G_p.value, a):
                break
            if self.alpha.value is not None:
                a = self.alpha.value
        self.prvs_alpha = a
        return a.copy()


# ---------------------------------------------------------------------------
# Nash greedy step
# ---------------------------------------------------------------------------
def _nash_step(
    z: torch.Tensor,
    engine: ScoringEngine,
    solver: NashSolver,
    device: torch.device,
    lr: float,
    lambda_prior: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    grads, scores = _compute_task_gradients(z, engine, device)
    props = [p for p in TASKS if p in grads]
    if not props:
        return z, scores

    G   = torch.stack([grads[p].flatten() for p in props])
    gtg = (G @ G.t()).cpu().detach().numpy()

    if solver.n_tasks != len(props):
        solver = NashSolver(n_tasks=len(props), optim_niter=solver.optim_niter)

    alpha  = solver.solve(gtg)
    a      = torch.from_numpy(alpha).float().to(device)
    update = (a.unsqueeze(1) * G.to(device)).sum(dim=0).reshape(z.shape)
    final  = torch.clamp(update + lambda_prior * z.detach().clone(), -0.1, 0.1)
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
def nash_epsilon_walk(
    vae: SelfiesVAE,
    engine: ScoringEngine,
    max_steps: int = 50_000,
    target_count: int = 0,
    lr: float = 0.01,
    lambda_prior: float = 3.0,
    latent_bounds: Tuple[float, float] = (-3.0, 3.0),
    log_every: int = 100,
    optim_niter: int = 20,
    epsilon: float = 0.10,
    noise_scale: float = 0.05,
    out_file: str = "outputs/nash_eps/passing_smiles_nash_eps_selfies.csv",
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

    solver = NashSolver(n_tasks=len(TASKS), optim_niter=optim_niter)
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
                z, scores = _nash_step(z, engine, solver, device, lr, lambda_prior)
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
                       help="Probability of taking a random step instead of Nash-MTL.")
        p.add_argument("--noise_scale",  type=float, default=0.05,
                       help="Std of Gaussian noise for the random step.")
        p.add_argument("--optim_niter",  type=int,   default=20)
        p.add_argument("--log_every",    type=int,   default=100)
        p.add_argument("--out_file",     type=str,
                       default="outputs/nash_eps/passing_smiles_nash_eps_selfies.csv")
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

    nash_epsilon_walk(
        vae=vae,
        engine=engine,
        max_steps=a.max_steps,
        target_count=a.target_count,
        lr=a.lr,
        lambda_prior=a.lambda_prior,
        log_every=a.log_every,
        optim_niter=a.optim_niter,
        epsilon=a.epsilon,
        noise_scale=a.noise_scale,
        out_file=a.out_file,
    )
