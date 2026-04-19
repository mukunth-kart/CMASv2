"""
eval_finetuned.py
=================
Generate molecules from a fine-tuned SELFIES VAE, check RDKit validity,
and compute Tanimoto similarity against a reference set (e.g. AKT1).

Sample modes
------------
  prior      : sample z ~ N(0, I)  [default, but misses AKT1 region]
  encoded    : encode ref molecules -> perturb their mu vectors -> decode
               (targets the actual AKT1 region of latent space)
  interpolate: random linear interpolations between pairs of encoded ref molecules

Usage
-----
    # prior sampling (baseline)
    python eval_finetuned.py --weights ./ckpts/akt1_v3/selfies_vae_weights.pt \
        --vocab ./ckpts/akt1_v3/tokenizer.json --ref ./data/AKT1_smiles.txt \
        --n 500 --temp 1.0 --sample_mode prior

    # sample near actual AKT1 encodings (recommended to test fine-tuning quality)
    python eval_finetuned.py --weights ./ckpts/akt1_v3/selfies_vae_weights.pt \
        --vocab ./ckpts/akt1_v3/tokenizer.json --ref ./data/AKT1_smiles.txt \
        --n 500 --temp 0.7 --sample_mode encoded --noise 0.5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from Generators.SelfiesVAE import SelfiesVAE

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mol_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def max_tanimoto_to_set(query_fp, ref_fps):
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps)
    return max(sims) if sims else 0.0


def encode_smiles_set(vae, smiles_list, max_encode=500):
    """Encode a list of SMILES -> list of mu tensors (on CPU)."""
    mus = []
    for smi in smiles_list[:max_encode]:
        try:
            _, mu = vae.encode_molecule(smi)
            mus.append(mu.cpu())
        except Exception:
            pass
    return mus


def sample_near_encodings(vae, mus, n, noise_std, temp, max_len):
    """
    For each sample: pick a random reference mu, add Gaussian noise, decode.
    noise_std=0 -> reconstruct reference; noise_std=1 -> prior-like.
    """
    results = []
    device = vae.device
    for _ in range(n):
        base_mu = mus[np.random.randint(len(mus))].to(device)
        z = base_mu + noise_std * torch.randn_like(base_mu)
        try:
            smi, _ = vae.generate_molecule(z=z, temperature=temp, max_len=max_len)
            results.append(smi)
        except Exception:
            results.append("")
    return results


def sample_interpolations(vae, mus, n, temp, max_len):
    """
    Sample along random linear interpolations between pairs of encoded ref mols.
    """
    results = []
    device = vae.device
    for _ in range(n):
        i, j = np.random.choice(len(mus), size=2, replace=False)
        alpha = np.random.uniform(0.0, 1.0)
        z = (alpha * mus[i] + (1 - alpha) * mus[j]).to(device)
        try:
            smi, _ = vae.generate_molecule(z=z, temperature=temp, max_len=max_len)
            results.append(smi)
        except Exception:
            results.append("")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(generated_smiles, ref_fps, label):
    valid_smiles = []
    seen = set()
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            valid_smiles.append(canonical)
            seen.add(canonical)

    validity   = len(valid_smiles) / max(len(generated_smiles), 1) * 100
    uniqueness = len(seen) / max(len(valid_smiles), 1) * 100

    print(f"\n{'='*58}")
    print(f"  Mode        : {label}")
    print(f"  Generated   : {len(generated_smiles)}")
    print(f"  Valid       : {len(valid_smiles)}  ({validity:.1f} %)")
    print(f"  Unique valid: {len(seen)}  ({uniqueness:.1f} % of valid)")

    if not valid_smiles:
        print("  No valid molecules.")
        print(f"{'='*58}\n")
        return

    scores = []
    for smi in valid_smiles:
        fp = mol_fp(smi)
        if fp is not None:
            scores.append(max_tanimoto_to_set(fp, ref_fps))

    if scores:
        arr = np.array(scores)
        print(f"\n  Max-Tanimoto vs. reference set:")
        print(f"    mean   {arr.mean():.4f}   median {np.median(arr):.4f}"
              f"   std {arr.std():.4f}   max {arr.max():.4f}")

        bins   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
        counts, _ = np.histogram(arr, bins=bins)
        width = max(counts)
        print(f"\n  Similarity distribution:")
        for lbl, cnt in zip(labels, counts):
            bar = "#" * int(cnt / width * 36)
            print(f"    {lbl}  {cnt:4d}  {bar}")

    print(f"\n  Sample valid SMILES (up to 8):")
    for k, smi in enumerate(list(seen)[:8], 1):
        fp = mol_fp(smi)
        sim = f"  sim={max_tanimoto_to_set(fp, ref_fps):.3f}" if fp else ""
        print(f"    {k:2d}. {smi}{sim}")

    print(f"{'='*58}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Evaluate fine-tuned SELFIES VAE.")
    p.add_argument("--weights", required=True)
    p.add_argument("--vocab",   required=True)
    p.add_argument("--ref",     required=True, help="Reference SMILES (one per line).")
    p.add_argument("--n",       type=int,   default=500)
    p.add_argument("--temp",    type=float, default=1.0)
    p.add_argument("--max_len", type=int,   default=120)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--sample_mode", choices=["prior", "encoded", "interpolate", "all"],
                   default="all",
                   help="'prior': N(0,I); 'encoded': perturb ref encodings; "
                        "'interpolate': latent interpolations; 'all': run all three.")
    p.add_argument("--noise", type=float, default=0.5,
                   help="Noise std added to ref encodings in 'encoded' mode (0=reconstruct, 1=prior-like).")
    p.add_argument("--max_encode", type=int, default=500,
                   help="Max reference molecules to encode (for encoded/interpolate modes).")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vae = SelfiesVAE(model_path=args.weights)
    vae.load_model(vocab_base=args.vocab)
    vae.model.eval()

    # Reference set
    ref_smiles = [s.strip() for s in Path(args.ref).read_text().splitlines() if s.strip()]
    ref_fps = [fp for smi in ref_smiles if (fp := mol_fp(smi)) is not None]
    print(f"Reference set : {len(ref_smiles)} SMILES  ({len(ref_fps)} valid fps)")

    run_modes = ["prior", "encoded", "interpolate"] if args.sample_mode == "all" else [args.sample_mode]

    # Pre-encode ref molecules once if needed
    mus = None
    if any(m in run_modes for m in ("encoded", "interpolate")):
        print(f"Encoding up to {args.max_encode} reference molecules ...")
        mus = encode_smiles_set(vae, ref_smiles, max_encode=args.max_encode)
        print(f"  Encoded {len(mus)} molecules successfully.")
        if len(mus) < 2:
            print("  Not enough encodable reference molecules — skipping encoded/interpolate modes.")
            run_modes = [m for m in run_modes if m == "prior"]

    for mode in run_modes:
        print(f"\n[{mode.upper()}] Generating {args.n} molecules ...")

        if mode == "prior":
            smiles = []
            for _ in range(args.n):
                try:
                    smi, _ = vae.generate_molecule(temperature=args.temp, max_len=args.max_len)
                    smiles.append(smi)
                except Exception:
                    smiles.append("")

        elif mode == "encoded":
            smiles = sample_near_encodings(
                vae, mus, args.n, noise_std=args.noise,
                temp=args.temp, max_len=args.max_len,
            )

        else:  # interpolate
            smiles = sample_interpolations(
                vae, mus, args.n, temp=args.temp, max_len=args.max_len,
            )

        label = (f"{mode}  temp={args.temp}"
                 + (f"  noise={args.noise}" if mode == "encoded" else ""))
        print_report(smiles, ref_fps, label)


if __name__ == "__main__":
    main()
