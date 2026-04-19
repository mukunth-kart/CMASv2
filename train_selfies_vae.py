"""
SELFIES-VAE Training Script
===========================
Two-phase workflow:

1. PRETRAIN on the full pChEMBL corpus (large, general drug-like coverage)
2. FINE-TUNE on the AKT1 bio-activity dataset (focused chemical neighborhood)

Both phases reuse the *same* SELFIES tokenizer (built once during
preprocessing), so latent spaces remain compatible across phases and
with downstream PCGrad / Nash-MTL walkers.

Examples
--------
    # Build vocab + tokenize ChEMBL (one-time)
    python data/smiles_to_selfies.py \
        --input ./data/ChemBL_Smiles.txt \
        --out_npy ./data/ChemBL_Selfies.npy \
        --vocab_out ./selfies_vocab.json --build_vocab

    # Tokenize AKT1 with the SAME vocab
    python data/smiles_to_selfies.py \
        --input ./data/AKT1_smiles.txt \
        --out_npy ./data/AKT1_Selfies.npy \
        --vocab_out ./selfies_vocab.json

    # Phase 1 — pretrain
    python train_selfies_vae.py --mode pretrain \
        --dataset ./data/ChemBL_Selfies.npy \
        --vocab   ./selfies_vocab.json \
        --save_dir ./trained_selfies_vae \
        --epochs 300 --batch_size 2048 --lr 1e-3

    # Phase 2 — fine-tune on AKT1
    python train_selfies_vae.py --mode finetune \
        --dataset ./data/AKT1_Selfies.npy \
        --vocab   ./selfies_vocab.json \
        --weights ./trained_selfies_vae/selfies_vae_weights.pt \
        --save_dir ./trained_selfies_vae_akt1 \
        --epochs 100 --batch_size 256 --lr 1e-4
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Generators.SelfiesVAE import SelfiesVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser("Train / fine-tune the SELFIES GRU-VAE.")
    p.add_argument("--mode", choices=["pretrain", "finetune"], required=True)
    p.add_argument("--dataset", required=True,
                   help="SELFIES corpus — .npy (preprocessed) or .txt (SMILES/SELFIES).")
    p.add_argument("--vocab", required=True,
                   help="SELFIES tokenizer JSON (from smiles_to_selfies.py).")
    p.add_argument("--weights", default=None,
                   help="Optional pretrained .pt to resume / fine-tune from.")
    p.add_argument("--save_dir", default="./trained_selfies_vae")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--start_epoch", type=int, default=0)
    
    # Early stopping / LR scheduler
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (epochs without improvement).")
    p.add_argument("--min_delta", type=float, default=0.001,
                   help="Minimum accuracy gain to count as improvement for early stopping.")
    p.add_argument("--grad_surgery", choices=["none", "pcgrad"], default="none",
                   help="'pcgrad': resolve KL/recon gradient conflicts via PCGrad per parameter. "
                        "~1.8x slower per batch; may reduce need for aggressive KL annealing.")
    p.add_argument("--lr_sched", choices=["none", "plateau", "cosine"], default="plateau",
                   help="LR scheduler: 'none' = fixed LR, 'plateau' = ReduceLROnPlateau, "
                        "'cosine' = CosineAnnealingLR.")
    p.add_argument("--lr_factor", type=float, default=0.5,
                   help="(plateau only) Factor by which the learning rate will be reduced.")
    p.add_argument("--lr_patience", type=int, default=5,
                   help="(plateau only) Patience for LR scheduler before reducing LR.")
    # KL annealing (floor + ramp, no dead warmup)
    p.add_argument("--kl_ramp", type=int, default=150,
                   help="Epochs over which KL weight linearly ramps from min_kl_weight to max_beta.")
    p.add_argument("--max_beta", type=float, default=0.05,
                   help="Maximum KL weight after ramp.")
    p.add_argument("--min_kl_weight", type=float, default=0.0001,
                   help="KL weight floor — never drops to zero, prevents encoder posterior explosion.")
    p.add_argument("--free_bits", type=float, default=1.0,
                   help="Minimum KL per batch (free bits) — prevents full KL collapse.")
    
    return p.parse_args()


# Sensible per-mode defaults
DEFAULTS = {
    "pretrain": dict(epochs=300, batch_size=2048, lr=1e-3),
    "finetune": dict(epochs=100, batch_size=256,  lr=1e-4),
}


def main():
    args = parse_args()
    defaults = DEFAULTS[args.mode]
    epochs     = args.epochs     or defaults["epochs"]
    batch_size = args.batch_size or defaults["batch_size"]
    lr         = args.lr         or defaults["lr"]

    if not Path(args.dataset).exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)
    if not Path(args.vocab).exists():
        logger.error(f"Vocab not found: {args.vocab}")
        sys.exit(1)
    if args.mode == "finetune" and not args.weights:
        logger.error("--weights is required in finetune mode.")
        sys.exit(1)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info("=" * 60)
        logger.info(f"  SELFIES-VAE {args.mode.upper()}")
        logger.info("=" * 60)
        logger.info(f"  Dataset    : {args.dataset}")
        logger.info(f"  Vocab      : {args.vocab}")
        logger.info(f"  Weights    : {args.weights or '(scratch)'}")
        logger.info(f"  Epochs     : {epochs}")
        logger.info(f"  Batch      : {batch_size}")
        logger.info(f"  LR         : {lr}")
        logger.info(f"  Patience   : {args.patience} (Early Stopping, min_delta={args.min_delta})")
        logger.info(f"  KL sched   : floor={args.min_kl_weight} ramp={args.kl_ramp} max_beta={args.max_beta} free_bits={args.free_bits}")
        logger.info(f"  Grad surg  : {args.grad_surgery}")
        logger.info(f"  LR sched   : {args.lr_sched}"
                    + (f" (factor={args.lr_factor}, patience={args.lr_patience})"
                       if args.lr_sched == "plateau" else ""))
        logger.info(f"  Save dir   : {args.save_dir}")
        logger.info("=" * 60)

    vae = SelfiesVAE(model_path=args.weights)
    vae.load_model(vocab_base=args.vocab)

    vae.fine_tune(
        dataset_path=args.dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        start_epoch=args.start_epoch,
        save_dir=args.save_dir,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_surgery=args.grad_surgery,
        lr_sched=args.lr_sched,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        kl_ramp=args.kl_ramp,
        max_beta=args.max_beta,
        min_kl_weight=args.min_kl_weight,
        free_bits=args.free_bits,
    )

    if local_rank == 0:
        logger.info(f"Done.  Weights at: {args.save_dir}")


if __name__ == "__main__":
    main()