"""
AttentionADMETClassifier  +  AttentionScoringEngine
====================================================
Drop-in replacements for ADMETClassifier and ScoringEngine that route ADMET
scoring through the per-task AttentionOracle checkpoints trained by
Models/AdmetClassifier/train_attention_admet.py.

Each task has its own AttentionOracle checkpoint at:
    ckpts/attention_admet/{variant}/{task_name}.pt

Usage in any existing walk script (pcgrad, nash, dynamic, epsilon-greedy):

    # Replace this:
    engine = ScoringEngine(
        activity_classifier_path=cfg["activity_classifier_model_path"],
        admet_model_path=cfg["admet_model_path"],
    )

    # With this:
    from utils.AttentionADMETClassifier import AttentionScoringEngine
    engine = AttentionScoringEngine(
        activity_classifier_path=cfg["activity_classifier_model_path"],
        attention_ckpt_dir="./ckpts/attention_admet",
        variant="akt1",          # or "pretrain"
    )

The walk functions only call engine.get_all_scores(z), so they work with
AttentionScoringEngine without any further changes.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../Models/AdmetClassifier")
    ),
)

from attention_oracle import AttentionOracle
from utils.ActivityClassifier import ActivityClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AttentionADMETClassifier
# ---------------------------------------------------------------------------

class AttentionADMETClassifier:
    """
    Loads one AttentionOracle per ADMET task and provides the same
    classify_admet(z) interface as ADMETClassifier and GrowNetADMETClassifier.

    Parameters
    ----------
    ckpt_dir    : Root checkpoint directory, e.g. "./ckpts/attention_admet".
    variant     : "pretrain" or "akt1" — selects the sub-folder.
    temperature : Sigmoid temperature applied to raw logits (default 2.0).
    device      : "cuda" or "cpu".
    """

    def __init__(
        self,
        ckpt_dir: str,
        variant: str,
        temperature: float = 2.0,
        device: str = "cuda",
    ) -> None:
        self.device      = torch.device(device if torch.cuda.is_available() else "cpu")
        self.temperature = temperature

        model_dir = Path(ckpt_dir) / variant
        if not model_dir.exists():
            raise FileNotFoundError(
                f"AttentionOracle checkpoint dir not found: {model_dir}\n"
                "Run train_attention_admet.py first."
            )

        self.models: Dict[str, AttentionOracle] = {}
        for pt_path in sorted(model_dir.glob("*.pt")):
            task_name = pt_path.stem
            model = AttentionOracle.load(str(pt_path), device=self.device)
            model.eval()
            self.models[task_name] = model
            logger.info(f"  AttentionOracle loaded: {task_name} <- {pt_path.name}")

        if not self.models:
            raise RuntimeError(
                f"No .pt files found in {model_dir}. "
                "Train with train_attention_admet.py first."
            )

        self.task_names = list(self.models.keys())
        logger.info(
            f"AttentionADMETClassifier ready  tasks={self.task_names}  "
            f"variant={variant}  device={self.device}"
        )

    def classify_admet(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns a dict mapping task name -> differentiable probability tensor.
        Keeps the computational graph alive so gradients flow back to z.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device)

        scores: Dict[str, torch.Tensor] = {}
        for task_name, model in self.models.items():
            logits = model(z)                              # (1, 1)
            logits = torch.clamp(logits, -10.0, 10.0)
            prob   = torch.sigmoid(logits / self.temperature).squeeze()
            scores[task_name] = prob
        return scores

    def get_task_probability(self, z: torch.Tensor, task_name: str) -> torch.Tensor:
        if task_name not in self.models:
            raise ValueError(
                f"Task '{task_name}' not found in loaded AttentionOracle models."
            )
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device)
        logits = self.models[task_name](z)
        return torch.sigmoid(
            torch.clamp(logits, -10.0, 10.0) / self.temperature
        ).squeeze()


# ---------------------------------------------------------------------------
# AttentionScoringEngine
# ---------------------------------------------------------------------------

class AttentionScoringEngine:
    """
    Drop-in replacement for utils.ScoringEngine that scores ADMET properties
    via per-task AttentionOracle models instead of MultiHeadADMET.

    The activity (potency) classifier is unchanged.
    """

    def __init__(
        self,
        activity_classifier_path: str,
        attention_ckpt_dir: str,
        variant: str,
        temperature: float = 2.0,
    ) -> None:
        self.activity_classifier_model = ActivityClassifier(activity_classifier_path)
        self.admet_classifier_model    = AttentionADMETClassifier(
            ckpt_dir=attention_ckpt_dir,
            variant=variant,
            temperature=temperature,
        )

    def get_all_scores(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores: Dict[str, torch.Tensor] = {}
        scores.update(self.admet_classifier_model.classify_admet(z))
        scores["potency"] = self.activity_classifier_model.classify_activity(z)
        return scores
