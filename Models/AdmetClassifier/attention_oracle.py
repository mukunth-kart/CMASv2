"""
attention_oracle.py
===================
Single-task Multi-Head Self-Attention predictor for ADMET properties.

Architecture
------------
The latent vector z (B, latent_dim) is treated as a short token sequence:

    z  →  patch-embed  →  [CLS | tok_0 | … | tok_{P-1}]  →  TransformerEncoder
       →  CLS output  →  LayerNorm  →  Linear  →  logit (scalar)

Patching splits z into P equal-length segments, each projected to `model_dim`.
A learnable [CLS] token and sinusoidal positional embeddings are prepended /
added. Pre-LN Transformer blocks (norm_first=True) give stable training.

Drop-in contract
----------------
* `AttentionOracle(latent_dim, task_name, ...)` — constructor.
* `forward(z)` → raw logit, shape `(B, 1)`, compatible with BCEWithLogitsLoss.
* `save(path)` / `load(path)` — checkpoint in the same format as GrowNetOracle
  (`{"model_state", "task_names", "latent_dim", …}`).
* `predict_tensor(z)` → sigmoid probability.
* `predict(z)` → dict {task_name: prob_tensor}.

Self-test:
    python Models/AdmetClassifier/attention_oracle.py
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["AttentionOracle"]


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

def _pick_n_patches(latent_dim: int, preferred: int = 8) -> int:
    """Largest divisor of latent_dim closest to `preferred`."""
    for candidate in [preferred, preferred * 2, preferred // 2, 4, 2, 1]:
        if candidate > 0 and latent_dim % candidate == 0:
            return candidate
    return 1


class PatchEmbedding(nn.Module):
    """Split z into P patches and project each to model_dim."""

    def __init__(self, latent_dim: int, n_patches: int, model_dim: int) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.patch_dim = latent_dim // n_patches
        self.proj = nn.Linear(self.patch_dim, model_dim)

    def forward(self, z: Tensor) -> Tensor:
        # z: (B, latent_dim)  →  (B, P, model_dim)
        B = z.size(0)
        patches = z.reshape(B, self.n_patches, self.patch_dim)
        return self.proj(patches)


# ---------------------------------------------------------------------------
# AttentionOracle
# ---------------------------------------------------------------------------

class AttentionOracle(nn.Module):
    """
    Single-task attention-based ADMET predictor.

    Parameters
    ----------
    latent_dim  : int   Input dimensionality of z.
    task_name   : str   Label stored in checkpoints (used by predict()).
    n_patches   : int   Number of tokens z is split into. Must divide latent_dim.
                        Pass 0 to auto-select.
    model_dim   : int   Internal transformer width (0 = auto).
    n_heads     : int   Attention heads (0 = auto).
    n_layers    : int   Transformer encoder depth (0 = auto).
    ffn_dim     : int   Feed-forward width inside each block (0 = 4*model_dim).
    dropout     : float Dropout inside transformer and patch projection.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        task_name: str = "task",
        n_patches: int = 0,
        model_dim: int = 0,
        n_heads: int = 0,
        n_layers: int = 2,
        ffn_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- auto-size unset params ---
        if n_patches == 0:
            n_patches = _pick_n_patches(latent_dim)
        assert latent_dim % n_patches == 0, (
            f"latent_dim ({latent_dim}) must be divisible by n_patches ({n_patches})"
        )
        patch_dim = latent_dim // n_patches
        if model_dim == 0:
            model_dim = max(32, min(128, patch_dim * 4))
            model_dim = (model_dim // 16) * 16  # round to multiple of 16
            model_dim = max(model_dim, 16)
        if n_heads == 0:
            n_heads = max(1, model_dim // 16)
        if ffn_dim == 0:
            ffn_dim = model_dim * 4

        # enforce divisibility: n_heads must divide model_dim
        while model_dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.latent_dim = latent_dim
        self.task_name  = task_name
        self.task_names = [task_name]   # list for API parity with GrowNetOracle
        self.n_patches  = n_patches
        self.model_dim  = model_dim
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.is_fitted  = False

        # Patch embedding
        self.patch_embed = PatchEmbedding(latent_dim, n_patches, model_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Sinusoidal positional embeddings (P+1 positions: CLS + patches)
        self.register_buffer(
            "pos_embed",
            self._sinusoidal_pe(n_patches + 1, model_dim),
        )

        # Transformer encoder (pre-LN for stability)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(model_dim)

        # Output head: CLS → logit
        self.head = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.head.bias)

        logger.debug(
            f"AttentionOracle({task_name})  "
            f"patches={n_patches}  model_dim={model_dim}  "
            f"n_heads={n_heads}  n_layers={n_layers}"
        )

    # ------------------------------------------------------------------
    # Positional encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d_model: int) -> Tensor:
        pe  = torch.zeros(1, seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        logit : (B, 1)   raw logit, no sigmoid.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.size(0)

        tokens = self.patch_embed(z)                           # (B, P, D)
        cls    = self.cls_token.expand(B, -1, -1)              # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)               # (B, P+1, D)
        tokens = tokens + self.pos_embed                       # add pos emb

        out = self.transformer(tokens)                         # (B, P+1, D)
        cls_out = self.norm(out[:, 0])                         # (B, D)
        return self.head(cls_out)                              # (B, 1)

    # ------------------------------------------------------------------
    # Inference helpers (API parity with GrowNetOracle)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_tensor(self, z: Tensor) -> Tensor:
        was_1d = z.dim() == 1
        if was_1d:
            z = z.unsqueeze(0)
        device = next(self.parameters()).device
        z = z.to(device).float()
        was_training = self.training
        self.eval()
        prob = torch.sigmoid(self.forward(z)).squeeze(1)       # (B,) or scalar
        if was_training:
            self.train()
        return prob[0] if was_1d else prob

    def predict(
        self, z: Tensor
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        scores = self.predict_tensor(z)
        if scores.dim() == 0:
            return {self.task_name: scores.clamp(0.0, 1.0)}
        return [
            {self.task_name: scores[b].clamp(0.0, 1.0)}
            for b in range(scores.size(0))
        ]

    # ------------------------------------------------------------------
    # Persistence (same checkpoint format as GrowNetOracle)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.state_dict(),
                "task_names":  self.task_names,
                "latent_dim":  self.latent_dim,
                "n_patches":   self.n_patches,
                "model_dim":   self.model_dim,
                "n_heads":     self.n_heads,
                "n_layers":    self.n_layers,
            },
            path,
        )
        logger.info(f"[AttentionOracle] saved -> {path}")

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
        map_location: Optional[Union[str, torch.device]] = "cpu",
    ) -> "AttentionOracle":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        task_name = ckpt["task_names"][0]
        oracle = cls(
            latent_dim=ckpt.get("latent_dim", 128),
            task_name=task_name,
            n_patches=ckpt.get("n_patches", 0),
            model_dim=ckpt.get("model_dim", 0),
            n_heads=ckpt.get("n_heads", 0),
            n_layers=ckpt.get("n_layers", 2),
        )
        oracle.load_state_dict(ckpt["model_state"])
        oracle.is_fitted = True
        if device is not None:
            oracle.to(device)
        logger.info(f"[AttentionOracle] loaded <- {path}")
        return oracle


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    logging.basicConfig(level=logging.INFO)
    LATENT_DIM, BATCH = 128, 32
    print("=== AttentionOracle self-test ===")

    model = AttentionOracle(latent_dim=LATENT_DIM, task_name="hERG_inhibition")
    print(model)

    z      = torch.randn(BATCH, LATENT_DIM)
    logits = model(z)
    assert logits.shape == (BATCH, 1), logits.shape
    print(f"  forward(z) -> {tuple(logits.shape)}  OK")

    # Gradient flow
    label  = torch.randint(0, 2, (BATCH, 1)).float()
    smooth = label * 0.9 + 0.05
    loss   = nn.BCEWithLogitsLoss()(logits, smooth)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "no gradients"
    print(f"  BCEWithLogitsLoss + backward  OK  (loss={loss.item():.4f})")

    # Checkpoint round-trip
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    model.save(tmp)
    model2 = AttentionOracle.load(tmp)
    model2.eval()
    model.eval()
    with torch.no_grad():
        diff = (model(z) - model2(z)).abs().max().item()
    assert diff < 1e-6, diff
    print(f"  checkpoint round-trip max diff: {diff:.2e}  OK")
    os.remove(tmp)

    # predict() API
    d = model.predict(z[0])
    assert "hERG_inhibition" in d
    print(f"  predict() dict  OK  ({d})")

    print("\nAll self-tests passed.")
