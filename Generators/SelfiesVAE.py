"""
SelfiesVAE
==========
A drop-in replacement for `Generators.VAE.VAE` that operates on SELFIES
(Krenn et al. 2019, https://arxiv.org/abs/1905.13741,
https://github.com/aspuru-guzik-group/selfies) instead of raw SMILES.

The underlying sequence model is identical to the SMILES GRU-VAE
(`MolSelfiesVAE`) — only the tokenization alphabet changes.  Because SELFIES
is a robust string grammar, *every* decoded token sequence corresponds
to a chemically valid molecule, which dramatically improves the hit-rate
of downstream latent-space walks (PCGrad, Nash-MTL).

The public API (`encode_molecule`, `generate_molecule`, `fine_tune`,
`update_search_distribution`) matches `Generators.VAE.VAE`, so existing
search scripts (`src/pcgrad_only.py`, `src/nash_mtl_walk.py`) can swap
the import and run unchanged.

Inputs / outputs at the API boundary are still SMILES — the SMILES↔SELFIES
conversion is handled internally.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedTokenizerFast
from rdkit import Chem, RDLogger

try:
    import selfies as sf
except ImportError as exc:
    raise ImportError("Install `selfies`: pip install selfies") from exc

from Datasets.SELFIESDataset import BinarySELFIESDataset, SELFIESTextDataset
from .MolSelfiesVAE import MolSelfiesVAE
from Generators.metrics import token_reconstruction_accuracy

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def _pcgrad_resolve(grads_a, grads_b):
    """Per-parameter PCGrad conflict resolution (Yu et al. 2020).

    Projects each gradient onto the normal plane of the other when they point
    in conflicting directions (negative dot product).  Both projections use the
    *original* gradients so neither projection affects the other.

    grads_a, grads_b: lists aligned with model.parameters() — entries may be None.
    Returns (resolved_a, resolved_b).
    """
    out_a, out_b = [], []
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None:
            out_a.append(ga)
            out_b.append(gb)
            continue
        dot = (ga * gb).sum()
        if dot < 0:
            gb_norm_sq = gb.norm().pow(2).clamp(min=1e-12)
            ga_norm_sq = ga.norm().pow(2).clamp(min=1e-12)
            # Both projections must use original ga and gb — compute before any mutation
            ga_new = ga - (dot / gb_norm_sq) * gb
            gb_new = gb - (dot / ga_norm_sq) * ga
            ga, gb = ga_new, gb_new
        out_a.append(ga)
        out_b.append(gb)
    return out_a, out_b


class SelfiesVAE:
    """GRU-VAE over the SELFIES alphabet.  SMILES-level API."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model: Optional[MolSelfiesVAE] = None
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.search_mean = None
        self.search_std  = None

        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Model / tokenizer loading
    # ------------------------------------------------------------------
    def load_model(self, vocab_base: str) -> None:
        """
        Load the SELFIES tokenizer (a HF tokenizer JSON produced by
        `data/smiles_to_selfies.py`) and the GRU-VAE backbone.
        """
        if not vocab_base or not vocab_base.endswith(".json"):
            raise ValueError("SelfiesVAE expects a tokenizer .json path "
                             "(produced by data/smiles_to_selfies.py).")

        if self.local_rank == 0:
            print(f"Loading SELFIES tokenizer from {vocab_base}")

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=vocab_base,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

        vocab_size = len(self.tokenizer)
        self.model = MolSelfiesVAE(vocab_size=vocab_size).to(self.device)

        if self.model_path and Path(self.model_path).exists():
            sd = torch.load(self.model_path, map_location=self.device)
            if sd["embedding.weight"].shape[0] != vocab_size:
                if self.local_rank == 0:
                    print(f"[warn] vocab mismatch — weights have "
                          f"{sd['embedding.weight'].shape[0]}, "
                          f"model has {vocab_size}.  Skipping load.")
            else:
                self.model.load_state_dict(sd)
                if self.local_rank == 0:
                    print("Loaded SELFIES-VAE weights.")

    # ------------------------------------------------------------------
    # Search-distribution steering (same contract as VAE.VAE)
    # ------------------------------------------------------------------
    def update_search_distribution(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.search_mean = mean.to(self.device)
        self.search_std  = std.to(self.device)

    # ------------------------------------------------------------------
    # SMILES <-> SELFIES conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _smiles_to_selfies(smiles: str) -> str:
        return sf.encoder(smiles)

    @staticmethod
    def _selfies_to_smiles(selfies_str: str) -> str:
        try:
            return sf.decoder(selfies_str)
        except Exception:
            return ""

    def _tokenize_selfies(self, selfies_str: str) -> torch.Tensor:
        tokens = list(sf.split_selfies(selfies_str))
        spaced = " ".join(tokens)
        final  = f"{self.tokenizer.bos_token} {spaced} {self.tokenizer.eos_token}"
        enc = self.tokenizer(final, return_tensors="pt", truncation=True, max_length=512)
        return enc["input_ids"].to(self.device)

    # ------------------------------------------------------------------
    # Inference: encode / generate
    # ------------------------------------------------------------------
    def encode_molecule(self, smiles: str) -> Tuple[str, torch.Tensor]:
        """SMILES -> SELFIES tokens -> deterministic mu latent."""
        if self.model is None:
            raise RuntimeError("Call load_model() first.")

        model = self.model.module if hasattr(self.model, "module") else self.model
        model.eval()

        selfies_str = self._smiles_to_selfies(smiles)
        input_ids = self._tokenize_selfies(selfies_str)

        with torch.no_grad():
            embedded = model.embedding(input_ids)
            _, h_n = model.encoder_gru(embedded)
            h_cat = torch.cat((h_n[0], h_n[1]), dim=1)
            z = model.fc_mu(h_cat).detach().clone()

        return smiles, z

    def generate_molecule(
        self,
        z: Optional[torch.Tensor] = None,
        exploration_rate: float = 0.3,
        max_len: int = 100,
        temperature: float = 0.8,
        max_retries: int = 10,
        retry_noise: float = 0.05,
    ) -> Tuple[str, torch.Tensor]:
        """Decode z -> SELFIES token ids -> SELFIES string -> SMILES.

        If decoding produces an invalid molecule, z is lightly perturbed and
        retried up to max_retries times so callers (PCGrad, Nash-MTL) always
        receive a chemically valid SMILES.  retry_noise controls the std of
        the perturbation; keeping it small (0.02–0.1) stays near the original
        latent point without drifting far.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first.")

        self.model.eval()

        # Special tokens to strip from the decoded sequence
        skip_ids = {
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,  # <-- unk breaks sf.decoder
        }

        if z is None:
            z = torch.randn(1, self.model.latent_dim).to(self.device)
            if self.search_mean is not None and torch.rand(1).item() > exploration_rate:
                z = z * self.search_std + self.search_mean
        else:
            z = z.to(self.device)

        origin_z = z.clone()  # keep for perturbation reference

        for attempt in range(max_retries):
            with torch.no_grad():
                token_ids = self.model.sample(
                    max_len=max_len,
                    start_token_idx=self.tokenizer.bos_token_id,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    z=z,
                    temp=temperature,
                )

            clean = [t for t in token_ids if t not in skip_ids]

            if clean:
                selfies_str = (
                    self.tokenizer.decode(clean, skip_special_tokens=True)
                    .replace(" ", "")
                )
                smiles = self._selfies_to_smiles(selfies_str)
                if smiles and Chem.MolFromSmiles(smiles) is not None:
                    return smiles, z

            # Invalid — perturb slightly around the original z and retry
            z = origin_z + retry_noise * torch.randn_like(origin_z)

        # All retries failed; return empty string so callers can log/skip
        return "", origin_z

    # ------------------------------------------------------------------
    # Training / fine-tuning
    # ------------------------------------------------------------------
    def fine_tune(
        self,
        dataset_path: str,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        start_epoch: int = 0,
        save_dir: str = "./trained_selfies_vae",
        max_beta: float = 0.05,
        kl_ramp: int = 150,
        min_kl_weight: float = 0.0001,
        word_dropout: float = 0.25,
        free_bits: float = 1.0,
        # --- Optimisation knobs ---
        patience: int = 15,
        min_delta: float = 0.001,
        lr_sched: str = "plateau",       # "none" | "plateau" | "cosine"
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        grad_surgery: str = "none",      # "none" | "pcgrad"
    ) -> str:
        """
        Train the SELFIES GRU-VAE with Early Stopping and LR Scheduling.
        """
        is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if is_distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        ds_path = Path(dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(ds_path)

        # ------- dataset -------
        pad_id = self.tokenizer.pad_token_id or 0
        if str(ds_path).endswith(".npy"):
            if self.local_rank == 0:
                print(f"Loading BINARY SELFIES dataset: {ds_path}")
            dataset = BinarySELFIESDataset(str(ds_path), pad_token_id=pad_id)
            num_workers, prefetch = 4, 2
        else:
            if self.local_rank == 0:
                print(f"Loading TEXT dataset: {ds_path}")
            dataset = SELFIESTextDataset(ds_path, self.tokenizer)
            num_workers, prefetch = 8, 4

        sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch,
        )

        # ------- model -------
        self.model.to(self.device)
        if is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             output_device=self.local_rank)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        scaler    = GradScaler("cuda")
        
        # LR scheduler — selected by lr_sched argument
        if lr_sched == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=lr_factor, patience=lr_patience,
            )
        elif lr_sched == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr * 0.01,
            )
        else:  # "none"
            scheduler = None

        # Early Stopping State
        best_acc = -1.0
        epochs_no_improve = 0

        def kl_weight(epoch: int) -> float:
            # Always keep a tiny floor so the encoder never trains
            # with zero KL pressure (prevents posterior explosion with z-injection).
            ramp_progress = min(1.0, epoch / max(1, kl_ramp))
            return min_kl_weight + ramp_progress * (max_beta - min_kl_weight)

        unk_id = self.tokenizer.unk_token_id or 0

        for epoch in range(start_epoch, epochs):
            if is_distributed:
                sampler.set_epoch(epoch)

            self.model.train()
            tot_loss = torch.zeros(1, device=self.device)
            tot_kl   = torch.zeros(1, device=self.device)
            tot_acc  = torch.zeros(1, device=self.device)
            steps = 0

            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)

                # word dropout
                masked = input_ids.clone()
                if word_dropout > 0:
                    drop = torch.rand(masked.shape, device=self.device) < word_dropout
                    masked[drop] = unk_id

                # safety: out-of-vocab ids
                actual_model = self.model.module if is_distributed else self.model
                if input_ids.max().item() >= actual_model.fc_out.out_features:
                    continue

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", dtype=torch.float16):
                    recon_logits, mu, logvar = self.model(masked)
                    recon_logits = torch.clamp(recon_logits, -20.0, 20.0)
                    mu, logvar = mu.float(), logvar.float()

                    logits  = recon_logits.reshape(-1, recon_logits.size(-1))
                    targets = labels[:, 1:].reshape(-1)

                    recon_loss = F.cross_entropy(logits, targets, ignore_index=-100)

                    kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
                    kl_div = -0.5 * torch.sum(kl_element, dim=1).mean()
                    kl_clamped = torch.max(
                        kl_div, torch.tensor(free_bits, device=self.device)
                    )

                if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                    continue

                w = kl_weight(epoch)

                if grad_surgery == "pcgrad":
                    # --- PCGrad: two backward passes, single unscale at the end ---
                    # Gradients from scaler are uniformly scaled — PCGrad dot products
                    # are scale-invariant, so we resolve on scaled grads and unscale once.
                    scaler.scale(recon_loss).backward(retain_graph=True)
                    g_recon = [p.grad.clone() if p.grad is not None else None
                               for p in actual_model.parameters()]

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(kl_clamped).backward()
                    g_kl = [p.grad.clone() if p.grad is not None else None
                            for p in actual_model.parameters()]

                    g_recon, g_kl = _pcgrad_resolve(g_recon, g_kl)

                    for p, gr, gk in zip(actual_model.parameters(), g_recon, g_kl):
                        if gr is not None and gk is not None:
                            p.grad = gr + w * gk
                        elif gr is not None:
                            p.grad = gr
                        elif gk is not None:
                            p.grad = w * gk
                        else:
                            p.grad = None

                else:
                    # --- standard single-pass ---
                    loss = recon_loss + w * kl_clamped
                    scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                loss_for_log = (recon_loss + w * kl_clamped).detach()
                tot_loss += loss_for_log
                tot_kl   += kl_div.detach()
                with torch.no_grad():
                    pred_ids = torch.argmax(recon_logits, dim=-1)
                    tot_acc += token_reconstruction_accuracy(
                        pred_ids, labels[:, 1:], pad_token_id=-100
                    )
                steps += 1

            if is_distributed:
                for t in (tot_loss, tot_kl, tot_acc):
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
            
            world = dist.get_world_size() if is_distributed else 1
            gsteps = steps * world
            if gsteps == 0:
                continue

            avg_loss = tot_loss.item() / gsteps
            avg_kl   = tot_kl.item()   / gsteps
            avg_acc  = tot_acc.item()  / gsteps
            
            # Step the scheduler
            if scheduler is not None:
                if lr_sched == "plateau":
                    scheduler.step(avg_acc)
                else:
                    scheduler.step()

            if self.local_rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1} | Loss {avg_loss:.4f} | "
                      f"KL {avg_kl:.2f} | KLw {kl_weight(epoch):.5f} | "
                      f"Acc {avg_acc:.3f} | LR {current_lr:.2e}")

                # Early Stopping Logic
                if avg_acc > best_acc + min_delta:
                    best_acc = avg_acc
                    epochs_no_improve = 0
                    
                    # Save best model
                    model_to_save = self.model.module if is_distributed else self.model
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    torch.save(model_to_save.state_dict(), save_path / "selfies_vae_weights.pt")
                    self.tokenizer.save_pretrained(save_path)
                else:
                    epochs_no_improve += 1

                if (epoch + 1) % 50 == 0:
                    ckpt = save_path / f"checkpoint_epoch_{epoch+1}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    torch.save(model_to_save.state_dict(), ckpt / "selfies_vae_weights.pt")

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.")
                    # In distributed mode, we need a way to tell other ranks to stop
                    stop_signal = torch.tensor(1.0, device=self.device)
                else:
                    stop_signal = torch.tensor(0.0, device=self.device)
            else:
                stop_signal = torch.tensor(0.0, device=self.device)

            # Sync stop signal across all processes
            if is_distributed:
                dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
                if stop_signal.item() > 0:
                    break
            elif self.local_rank == 0 and epochs_no_improve >= patience:
                break

            if is_distributed:
                dist.barrier()

        if is_distributed:
            dist.destroy_process_group()

        return str(Path(save_dir).resolve())
