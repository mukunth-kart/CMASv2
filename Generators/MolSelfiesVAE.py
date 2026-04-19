"""
MolSelfiesVAE
=============
Bidirectional-GRU VAE for SELFIES token sequences (Krenn et al. 2019).

The sequence-model architecture is alphabet-agnostic and therefore
structurally identical to `Generators.MolGRUVAE.MolGRUVAE`; this file
exists as a dedicated, self-contained copy so the SELFIES pipeline can
evolve independently of the SMILES stack (different latent sizes,
additional tokens, etc.) without touching the original.

Only `MolSelfiesVAE` is exported.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["MolSelfiesVAE"]


class MolSelfiesVAE(nn.Module):
    """Bidirectional GRU encoder + unidirectional GRU decoder VAE."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        z_proj_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.z_proj_dim = z_proj_dim

        # --- Encoder ---
        self.embedding   = nn.Embedding(vocab_size, embed_dim)
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu  = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)

        # --- Decoder ---
        # z is projected and concatenated with each decoder input step
        self.z_proj        = nn.Linear(latent_dim, z_proj_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru   = nn.GRU(embed_dim + z_proj_dim, hidden_dim, batch_first=True)
        self.fc_out        = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, word_dropout_rate: float = 0.0):
        embedded = self.embedding(input_ids)

        _, h_n = self.encoder_gru(embedded)
        h_cat = torch.cat((h_n[0], h_n[1]), dim=1)

        mu     = self.fc_mu(h_cat)
        logvar = torch.clamp(self.fc_var(h_cat), max=10.0)
        z      = self.reparameterize(mu, logvar)

        h_dec = self.decoder_input(z).unsqueeze(0)
        dec_in = embedded[:, :-1, :]

        if self.training and word_dropout_rate > 0:
            mask = (torch.rand(dec_in.shape[:2], device=input_ids.device)
                    > word_dropout_rate).unsqueeze(2)
            dec_in = dec_in * mask

        # Inject z at every decoder step by concatenating projected z
        z_proj = self.z_proj(z)                                          # (B, z_proj_dim)
        z_expanded = z_proj.unsqueeze(1).expand(-1, dec_in.size(1), -1)  # (B, T, z_proj_dim)
        dec_in = torch.cat([dec_in, z_expanded], dim=-1)                 # (B, T, embed+z_proj)

        decoder_out, _ = self.decoder_gru(dec_in, h_dec)
        logits = self.fc_out(decoder_out)
        return logits, mu, logvar

    # ------------------------------------------------------------------
    def sample(
        self,
        max_len: int,
        start_token_idx: int,
        tokenizer,
        device,
        temp: float = 1.0,
        z: torch.Tensor | None = None,
    ):
        """Autoregressive greedy-multinomial decoding from (optional) z."""
        if z is None:
            z = torch.randn(1, self.latent_dim, device=device)

        h_dec = self.decoder_input(z).unsqueeze(0)
        z_proj = self.z_proj(z)  # (1, z_proj_dim) — reused every step
        curr = torch.tensor([[start_token_idx]], device=device)
        out_ids = []

        for _ in range(max_len):
            embed = self.embedding(curr)
            embed = torch.cat([embed, z_proj.unsqueeze(1)], dim=-1)
            step_out, h_dec = self.decoder_gru(embed, h_dec)
            logits = self.fc_out(step_out.squeeze(1))
            probs = F.softmax(logits / temp, dim=-1)
            nxt = torch.multinomial(probs, 1)
            tid = nxt.item()
            if tid == tokenizer.eos_token_id:
                break
            out_ids.append(tid)
            curr = nxt
        return out_ids
