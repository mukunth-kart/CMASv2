# CMAS 2.0 — HGT-LDM

**Hierarchical Graph Transformer VAE with Latent Diffusion Prior** for the CMAS drug discovery framework.

## Overview

HGT-LDM replaces the GRU-VAE backbone with a modern graph-native architecture:

1. **GPS++ Graph Transformer Encoder** — 6-layer graph transformer with MPNN++ local + multi-head global attention, random walk positional encoding, and a virtual node readout for graph-level embeddings.
2. **Junction-Tree Decoder** — GRU-based tree generation + fragment assembly with real RDKit valence masking.
3. **Conditional Latent Diffusion Prior** — DDPM on latent vectors conditioned on 10 property targets, using a Transformer noise network with Adaptive Layer Normalisation (AdaLN).
4. **Differentiable Oracles** — Activity and 9-task ADMET property oracles that support autograd for gradient-based optimisation.
5. **CMAS Multi-Agent Loop** — Hunter (sample + filter) and Medic (constrained gradient ascent repair with diffusion score regulariser) agents.

## Project Structure

```
cmas-hgt-ldm/
├── configs/           # Hyperparameters (default.yaml, paths.yaml)
├── src/
│   ├── data/          # Featuriser, vocabulary, dataset, transforms
│   ├── models/        # Encoder, decoder, diffusion, VAE, oracles
│   ├── agents/        # Blackboard, HallOfFame, Hunter, Medic
│   ├── pipeline/      # Training loops, CMAS loop
│   └── utils/         # Chemistry helpers, metrics, scheduling, logging
├── tests/             # Pytest test suite
└── scripts/           # CLI entry points
```

## Setup

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in paths:

```bash
cp .env.example .env
```

## Usage

### 1. Preprocess ChEMBL Data

```bash
python scripts/preprocess_chembl.py \
    --smiles_file data/chembl/smiles.txt \
    --output_dir  data/processed
```

### 2. Train the VAE

```bash
python scripts/train_vae.py \
    --config configs/default.yaml \
    --data_dir data/processed \
    --device cuda
```

### 3. Train the Oracles

```bash
python scripts/train_oracles.py \
    --config configs/default.yaml \
    --vae_checkpoint outputs/vae.pt \
    --device cuda
```

### 4. Train the Diffusion Prior

```bash
python scripts/train_diffusion.py \
    --config configs/default.yaml \
    --vae_checkpoint outputs/vae.pt \
    --device cuda
```

### 5. Run CMAS Optimisation

```bash
python scripts/run_cmas.py \
    --config         configs/default.yaml \
    --vae_checkpoint outputs/vae.pt \
    --diffusion_ckpt outputs/diffusion.pt \
    --oracle_ckpt    outputs/oracles.pt \
    --device         cuda \
    --generations    50 \
    --condition      "[0.9,0.1,0.8,0.1,0.1,0.1,0.8,0.1,0.8,0.1]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Key Design Decisions

- **Virtual node**: Added to each graph and connected to all real nodes; its representation after 6 GPS++ layers serves as the graph-level embedding.
- **Attention masking**: PyG `batch` vector is used to prevent cross-graph attention in batched mode.
- **Valence masking**: Uses `rdkit.Chem.GetPeriodicTable().GetDefaultValence()` for hard chemical validity guarantees during decoding.
- **Diffusion score regulariser**: The NEW contribution in L_medic — adds t=5% noise to z, runs one denoising step, and penalises `||z - z_denoised||²` to keep gradient ascent on the learned manifold.
- **Free-bits KL**: Per-dimension KL clamped to ≥ `free_bits` nats to prevent posterior collapse.

## Architecture Details

| Component | Details |
|-----------|---------|
| Latent dim | 256 |
| GPS++ layers | 6 |
| Hidden dim | 256 |
| Attention heads | 8 |
| RWPE dim | 16 |
| Diffusion T | 1000 |
| Noise net layers | 4 (Transformer) |
| ADMET tasks | 9 (hERG, CYP3A4, BBBP, CYP1A2, CYP2C19, CYP2C9, HLM, P-gp, RLM) |
