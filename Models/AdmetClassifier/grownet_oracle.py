"""
grownet_oracle.py
=================
GrowNet (Badirli et al., arXiv:2002.07971) adapted as a multi-task latent-space
ADMET predictor that is a **drop-in replacement for `MultiHeadADMET`** in the
AdmetClassifier scripts (train_multitask.py, inference_multi_head_admet.py,
threshold_moving.py, test_admet.py, bootstrap_test.py, ...).

Drop-in contract
----------------
* Constructor: `GrowNetOracle(latent_dim=128, num_tasks=11)`  (same as MultiHeadADMET)
* `forward(z)` -> raw logits, shape `(B, num_tasks)`, compatible with
  `nn.BCEWithLogitsLoss` and external `torch.sigmoid(...)` calls.
* `state_dict()` / `load_state_dict()` work via the standard `nn.Module` API.
* Checkpoint format saved by `.save(path)` mirrors what train_multitask.py
  writes:  `{"model_state": ..., "task_names": ..., ...}`.

How it differs from a vanilla MLP
---------------------------------
The model is an ensemble of `n_learners` small 2-hidden-layer MLPs ("weak
learners"). Learner k receives `[z || penultimate_{k-1}]` as input -- the
"feature-stacking" trick from GrowNet. The final prediction is the
boost-rate-weighted sum of every learner's output.  All learners and boost
rates are trained jointly end-to-end with backprop (this is exactly the
"corrective step" of GrowNet, but used as the only training mode).

  pred(z) = sum_k  alpha_k * f_k( [z || h_{k-1}(z)] )

This gives you a deeper, residual-like model than a flat MLP while keeping
identical I/O semantics to MultiHeadADMET.

Usage in train_multitask.py
---------------------------
Just swap the model line:

    # from train_multitask import MultiHeadADMET
    from grownet_oracle import GrowNetOracle as MultiHeadADMET

    model = MultiHeadADMET(latent_dim=dataset.latent_dim,
                           num_tasks=len(dataset.tasks)).to(DEVICE)
    # ... rest of train() works unchanged ...

Usage in inference_multi_head_admet.py
--------------------------------------
    from grownet_oracle import GrowNetOracle
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GrowNetOracle(latent_dim=128, num_tasks=len(ckpt['task_names']))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    logits = model(z_batch)
    probs  = torch.sigmoid(logits)

Bonus: experience-buffer flow
-----------------------------
For the latent-walk use-case (`OracleAwareScoringEngine` + `ExperienceBuffer`),
`fit(zs, ys)` runs supervised end-to-end training with BCEWithLogitsLoss on
already-fully-labelled (z, y_vec) pairs and `predict(z)` returns a dict of
sigmoid-clamped scores keyed by task name -- compatible with the original
ScoringEngine interface.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "WeakLearner",
    "GrowNetOracle",
    "ExperienceBuffer",
    "OracleAwareScoringEngine",
    "build_oracle_engine",
]


# ---------------------------------------------------------------------------
# WeakLearner -- one shallow MLP in the GrowNet ensemble
# ---------------------------------------------------------------------------

class WeakLearner(nn.Module):
    """
    2-hidden-layer MLP that maps stacked input features -> n_tasks logits.

    forward(x) returns (logits, penultimate) so the next learner can stack
    `penultimate` onto the original z without relying on cached state.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_tasks: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(hidden_dim, n_tasks)

    @property
    def penultimate_dim(self) -> int:
        return self.fc2.out_features

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = F.silu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.silu(self.bn2(self.fc2(h)))
        return self.out(h), h  # (B, n_tasks), (B, hidden_dim)


# ---------------------------------------------------------------------------
# GrowNetOracle -- drop-in MultiHeadADMET replacement
# ---------------------------------------------------------------------------

class GrowNetOracle(nn.Module):
    """
    Multi-task GrowNet predictor with the same I/O signature as MultiHeadADMET.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input latent vector z.
    num_tasks : int, optional
        Number of ADMET tasks. Either this or `task_names` must be given.
    task_names : list[str], optional
        Ordered task labels (used by `predict()` and saved into checkpoints).
        Defaults to ["task_0", ..., f"task_{num_tasks-1}"] if not provided.
    n_learners : int
        Number of weak learners in the ensemble (depth of boosting chain).
    hidden_dim : int
        Hidden width of every weak learner.
    boost_rate : float
        Initial value for every learner's boost coefficient alpha_k. Stored
        in log-space as an `nn.Parameter` so it is learned end-to-end.
    dropout : float
        Dropout inside each weak learner.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_tasks: Optional[int] = 11,
        task_names: Optional[List[str]] = None,
        n_learners: int = 16,
        hidden_dim: int = 256,
        boost_rate: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if task_names is None and num_tasks is None:
            raise ValueError("Provide either `num_tasks` or `task_names`.")
        if task_names is None:
            task_names = [f"task_{i}" for i in range(num_tasks)]
        if num_tasks is None:
            num_tasks = len(task_names)
        if len(task_names) != num_tasks:
            raise ValueError(
                f"task_names has {len(task_names)} entries but num_tasks={num_tasks}"
            )

        self.latent_dim = latent_dim
        self.num_tasks = num_tasks
        self.task_names = list(task_names)
        self.n_learners = n_learners
        self.hidden_dim = hidden_dim

        self.learners = nn.ModuleList()
        for k in range(n_learners):
            in_dim = latent_dim if k == 0 else latent_dim + hidden_dim
            self.learners.append(
                WeakLearner(in_dim, hidden_dim, num_tasks, dropout=dropout)
            )

        # Learnable per-learner boost rates, parameterised in log-space so
        # they stay positive after gradient updates.
        init = torch.log(torch.tensor(max(boost_rate, 1e-6), dtype=torch.float32))
        self.log_boost_rates = nn.Parameter(init.expand(n_learners).clone())

        self.is_fitted = False

    # ------------------------------------------------------------------
    # Core forward (drop-in replacement for MultiHeadADMET.forward)
    # ------------------------------------------------------------------

    def forward(self, z: Tensor) -> Tensor:
        """
        Run the full GrowNet ensemble.

        Parameters
        ----------
        z : Tensor of shape (B, latent_dim)

        Returns
        -------
        logits : Tensor of shape (B, num_tasks)  -- raw, no sigmoid.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        alphas = torch.exp(self.log_boost_rates)
        pred = torch.zeros(
            z.size(0), self.num_tasks, device=z.device, dtype=z.dtype
        )

        x = z
        for k, learner in enumerate(self.learners):
            out, penu = learner(x)                    # (B, T), (B, H)
            pred = pred + alphas[k] * out
            if k + 1 < len(self.learners):
                x = torch.cat([z, penu], dim=1)       # stack for next learner
        return pred                                    # (B, num_tasks) logits

    # ------------------------------------------------------------------
    # Convenience inference helpers (used by buffer / oracle-engine flow)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_tensor(self, z: Tensor) -> Tensor:
        """
        Sigmoid-squashed prediction. Returns shape `(num_tasks,)` for a 1-D
        input or `(B, num_tasks)` for a batched input.
        """
        was_1d = z.dim() == 1
        if was_1d:
            z = z.unsqueeze(0)
        device = next(self.parameters()).device
        z = z.to(device).float()
        was_training = self.training
        self.eval()
        logits = self.forward(z)
        if was_training:
            self.train()
        probs = torch.sigmoid(logits)
        return probs[0] if was_1d else probs

    def predict(
        self, z: Tensor
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Drop-in for `ScoringEngine.get_all_scores(z)`. Returns a dict mapping
        task name -> scalar tensor in [0, 1] for a single z, or a list of such
        dicts for a batch.
        """
        scores = self.predict_tensor(z)
        if scores.dim() == 1:
            return {
                name: scores[i].clamp(0.0, 1.0)
                for i, name in enumerate(self.task_names)
            }
        return [
            {
                name: scores[b, i].clamp(0.0, 1.0)
                for i, name in enumerate(self.task_names)
            }
            for b in range(scores.size(0))
        ]

    # ------------------------------------------------------------------
    # Optional: end-to-end supervised fit on (z, y_vec) pairs
    # (used by ExperienceBuffer / OracleAwareScoringEngine, NOT by
    # train_multitask.py which has its own loop)
    # ------------------------------------------------------------------

    def fit(
        self,
        zs: Tensor,
        ys: Tensor,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        verbose: bool = False,
    ) -> None:
        """
        Train the whole ensemble end-to-end with BCEWithLogitsLoss.

        Parameters
        ----------
        zs : (N, latent_dim) float tensor
        ys : (N, num_tasks)  float tensor in [0, 1] (full label vectors)
        """
        device = next(self.parameters()).device
        zs = zs.to(device).float()
        ys = ys.to(device).float()
        N = zs.size(0)

        opt = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        crit = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            self.train()
            idx = torch.randperm(N, device=device)
            running = 0.0
            seen = 0
            for start in range(0, N, batch_size):
                b = idx[start : start + batch_size]
                if b.numel() < 2:        # BatchNorm1d needs at least 2
                    continue
                logits = self.forward(zs[b])
                loss = crit(logits, ys[b])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                opt.step()
                running += loss.item() * b.numel()
                seen += b.numel()
            if verbose and seen > 0:
                logger.info(
                    f"[GrowNet] fit epoch {epoch + 1}/{epochs}  "
                    f"loss={running / seen:.4f}"
                )

        self.is_fitted = True

    # ------------------------------------------------------------------
    # Persistence in the same format train_multitask.py uses
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save in the same checkpoint shape as train_multitask.py:
            {"model_state": state_dict, "task_names": [...], ...}
        """
        torch.save(
            {
                "model_state": self.state_dict(),
                "task_names": self.task_names,
                "latent_dim": self.latent_dim,
                "num_tasks": self.num_tasks,
                "n_learners": self.n_learners,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )
        logger.info(f"[GrowNet] saved -> {path}")

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
        map_location: Optional[Union[str, torch.device]] = "cpu",
    ) -> "GrowNetOracle":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        oracle = cls(
            latent_dim=ckpt.get("latent_dim", 128),
            num_tasks=ckpt.get("num_tasks", len(ckpt["task_names"])),
            task_names=ckpt["task_names"],
            n_learners=ckpt.get("n_learners", 8),
            hidden_dim=ckpt.get("hidden_dim", 256),
        )
        oracle.load_state_dict(ckpt["model_state"])
        oracle.is_fitted = True
        if device is not None:
            oracle.to(device)
        logger.info(f"[GrowNet] loaded <- {path}")
        return oracle


# ---------------------------------------------------------------------------
# ExperienceBuffer -- collects (z, scores) pairs during the latent walk
# ---------------------------------------------------------------------------

class ExperienceBuffer:
    """
    Ring buffer that accumulates (z, score_vector) pairs from a latent walk.

    Each call to `add(z, scores_dict)` stores one fully-labelled sample.
    `to_tensors(device)` returns the stacked (zs, ys) tensors ready for
    `GrowNetOracle.fit()`.
    """

    def __init__(self, task_names: List[str], max_size: int = 50_000) -> None:
        self.task_names = list(task_names)
        self.max_size = max_size
        self._zs: deque = deque(maxlen=max_size)
        self._ys: deque = deque(maxlen=max_size)
        self._n_added = 0

    def add(self, z: Tensor, scores: Dict[str, float]) -> None:
        y_vec = torch.tensor(
            [float(scores.get(t, 0.0)) for t in self.task_names],
            dtype=torch.float32,
        )
        self._zs.append(z.detach().cpu().float().flatten())
        self._ys.append(y_vec)
        self._n_added += 1

    def __len__(self) -> int:
        return len(self._zs)

    def ready(self, min_samples: int = 200, retrain_every: int = 100) -> bool:
        return (
            len(self) >= min_samples
            and self._n_added > 0
            and self._n_added % retrain_every == 0
        )

    def to_tensors(
        self, device: Optional[torch.device] = None
    ) -> Tuple[Tensor, Tensor]:
        zs = torch.stack(list(self._zs))
        ys = torch.stack(list(self._ys))
        if device is not None:
            zs = zs.to(device)
            ys = ys.to(device)
        return zs, ys


# ---------------------------------------------------------------------------
# OracleAwareScoringEngine -- transparent ScoringEngine wrapper
# ---------------------------------------------------------------------------

class OracleAwareScoringEngine:
    """
    Routes scoring requests to the GrowNetOracle once it is fitted, and to the
    ground-truth ScoringEngine otherwise. Each ground-truth call is buffered;
    when enough new samples have accumulated, the oracle is re-fit.
    """

    def __init__(
        self,
        scoring_engine,
        oracle: GrowNetOracle,
        buffer: ExperienceBuffer,
        use_oracle: bool = True,
        min_samples: int = 200,
        retrain_every: int = 100,
        fit_epochs: int = 5,
    ) -> None:
        self.scoring_engine = scoring_engine
        self.oracle = oracle
        self.buffer = buffer
        self.use_oracle = use_oracle
        self.min_samples = min_samples
        self.retrain_every = retrain_every
        self.fit_epochs = fit_epochs
        self._gt_calls = 0
        self._oracle_calls = 0

    def get_all_scores(self, z: Tensor) -> Dict[str, Tensor]:
        if self.use_oracle and self.oracle.is_fitted:
            self._oracle_calls += 1
            return self.oracle.predict(z)
        return self._ground_truth_call(z)

    def _ground_truth_call(self, z: Tensor) -> Dict[str, Tensor]:
        scores = self.scoring_engine.get_all_scores(z)
        scores_f = {
            k: float(v.item()) if isinstance(v, Tensor) else float(v)
            for k, v in scores.items()
        }
        self.buffer.add(z, scores_f)
        self._gt_calls += 1

        if self.buffer.ready(self.min_samples, self.retrain_every):
            device = next(self.oracle.parameters()).device
            zs, ys = self.buffer.to_tensors(device)
            logger.info(
                f"[OracleEngine] Retraining oracle on {len(self.buffer)} samples "
                f"(gt_calls={self._gt_calls})"
            )
            self.oracle.fit(zs, ys, epochs=self.fit_epochs)

        return scores

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "gt_calls": self._gt_calls,
            "oracle_calls": self._oracle_calls,
            "buffer_size": len(self.buffer),
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_oracle_engine(
    scoring_engine,
    task_names: List[str],
    latent_dim: int,
    n_learners: int = 8,
    hidden_dim: int = 256,
    min_samples: int = 200,
    retrain_every: int = 100,
    device: Optional[torch.device] = None,
) -> OracleAwareScoringEngine:
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    oracle = GrowNetOracle(
        latent_dim=latent_dim,
        task_names=task_names,
        n_learners=n_learners,
        hidden_dim=hidden_dim,
    ).to(device)
    buffer = ExperienceBuffer(task_names=task_names, max_size=50_000)
    return OracleAwareScoringEngine(
        scoring_engine=scoring_engine,
        oracle=oracle,
        buffer=buffer,
        min_samples=min_samples,
        retrain_every=retrain_every,
    )


# ---------------------------------------------------------------------------
# Quick self-test:
#   * shape compatibility with MultiHeadADMET
#   * BCEWithLogitsLoss + per-task indexing pattern from train_multitask.py
#   * checkpoint round-trip
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    logging.basicConfig(level=logging.INFO)

    LATENT_DIM = 128
    NUM_TASKS = 11
    BATCH = 64

    print("=== GrowNetOracle drop-in self-test ===")

    model = GrowNetOracle(latent_dim=LATENT_DIM, num_tasks=NUM_TASKS)
    print(model)

    z = torch.randn(BATCH, LATENT_DIM)
    logits = model(z)
    assert logits.shape == (BATCH, NUM_TASKS), logits.shape
    print(f"  forward(z) -> {tuple(logits.shape)}  OK")

    # Replicate the exact loss pattern from train_multitask.py:
    task_ids = torch.randint(0, NUM_TASKS, (BATCH,))
    labels = torch.randint(0, 2, (BATCH,)).float()
    target_preds = logits[torch.arange(BATCH), task_ids]
    smooth = labels * 0.9 + 0.05
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))(target_preds, smooth)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "no gradients flowed"
    print(f"  BCEWithLogitsLoss + backward  OK  (loss={loss.item():.4f})")

    # Inference path matches inference_multi_head_admet.py
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(z))
    assert probs.shape == (BATCH, NUM_TASKS)
    assert (probs >= 0).all() and (probs <= 1).all()
    print(f"  sigmoid(model(z)) -> [0,1]^{tuple(probs.shape)}  OK")

    # Checkpoint round-trip in train_multitask.py format
    tmp = os.path.join(tempfile.gettempdir(), "grownet_oracle_test.pt")
    torch.save(
        {"model_state": model.state_dict(),
         "task_names": [f"task_{i}" for i in range(NUM_TASKS)]},
        tmp,
    )
    ckpt = torch.load(tmp, weights_only=False)
    model2 = GrowNetOracle(latent_dim=LATENT_DIM, num_tasks=len(ckpt["task_names"]))
    model2.load_state_dict(ckpt["model_state"])
    model2.eval()
    with torch.no_grad():
        diff = (model(z) - model2(z)).abs().max().item()
    assert diff < 1e-6, diff
    print(f"  checkpoint round-trip max diff: {diff:.2e}  OK")
    os.remove(tmp)

    # Optional fit() path used by ExperienceBuffer flow
    zs = torch.randn(300, LATENT_DIM)
    ys = torch.rand(300, NUM_TASKS)
    model.fit(zs, ys, epochs=2, batch_size=64, verbose=True)
    d = model.predict(zs[0])
    assert set(d.keys()) == {f"task_{i}" for i in range(NUM_TASKS)}
    print(f"  fit() + predict() dict  OK  (keys={len(d)})")

    print("\nAll drop-in self-tests passed.")
