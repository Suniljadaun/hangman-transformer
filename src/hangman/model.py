"""The guessing network: a masked-word transformer with two prediction heads.

The board is a sequence of characters, most of them hidden. Head A is a
masked-language-model head that predicts the letter at each hidden position;
its per-position distributions are combined by noisy-OR into "does letter L
appear anywhere still hidden". Head B answers that question directly from a
pooled representation. The two disagree in useful ways, so we fuse them.

The network is conditioned on the letters already tried -- crucially including
the ones known to be *absent*, which is a large part of the information a real
player has and which most implementations discard.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import MASK_ID, MAX_LIVES, N_LETTERS, PAD_ID, VOCAB_SIZE


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 32

    @property
    def n_params_estimate(self) -> int:
        return self.n_layers * (4 * self.d_model**2 + 2 * self.d_model * self.d_ff)


class HangmanNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.token_emb = nn.Embedding(VOCAB_SIZE, d, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        self.len_emb = nn.Embedding(cfg.max_len + 1, d)
        self.lives_emb = nn.Embedding(MAX_LIVES + 1, d)
        # Two 26-dim views of the guess history: what was tried, and what of
        # that is known to be absent. Their difference encodes the hits.
        self.history_proj = nn.Linear(2 * N_LETTERS, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d)

        self.head_position = nn.Linear(d, N_LETTERS)          # head A
        self.head_word = nn.Sequential(                        # head B
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_LETTERS)
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, tokens, guessed, absent, lives, lengths):
        """
        tokens   (b, L) long
        guessed  (b, 26) float
        absent   (b, 26) float
        lives    (b,) long
        lengths  (b,) long
        returns  position_logits (b, L, 26), word_logits (b, 26)
        """
        b, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(b, L)

        context = (
            self.history_proj(torch.cat([guessed, absent], dim=-1))
            + self.lives_emb(lives)
            + self.len_emb(lengths)
        ).unsqueeze(1)

        x = self.token_emb(tokens) + self.pos_emb(pos) + context
        pad_mask = tokens.eq(PAD_ID)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        position_logits = self.head_position(x)

        hidden = tokens.eq(MASK_ID).unsqueeze(-1).float()
        pooled = (x * hidden).sum(1) / hidden.sum(1).clamp(min=1.0)
        word_logits = self.head_word(pooled)
        return position_logits, word_logits

    @torch.no_grad()
    def score(self, tokens, guessed, absent, lives, lengths, fusion: float = 0.5):
        """Fused log-scores over the 26 letters, with tried letters suppressed.

        Head A is aggregated by noisy-OR across hidden positions:
            P(letter appears) = 1 - prod_i (1 - p_i(letter))
        computed in log space for stability.
        """
        position_logits, word_logits = self.forward(
            tokens, guessed, absent, lives, lengths
        )
        hidden = tokens.eq(MASK_ID)                             # (b, L)
        log_p = F.log_softmax(position_logits, dim=-1)          # (b, L, 26)
        log_not = torch.log1p(-log_p.exp().clamp(max=1 - 1e-6))  # log(1 - p)
        log_none = (log_not * hidden.unsqueeze(-1)).sum(1)       # (b, 26)

        # logit of P(letter appears somewhere hidden) = log(1-e^x) - x
        log_none = log_none.clamp(max=-1e-6)
        noisy_or_logit = torch.log(-torch.expm1(log_none)) - log_none

        fused = fusion * noisy_or_logit + (1.0 - fusion) * word_logits
        return fused.masked_fill(guessed.bool(), float("-inf"))


def losses(model_out, targets, hidden_mask, present_mask):
    """Head A: masked-LM cross entropy. Head B: multi-label BCE."""
    position_logits, word_logits = model_out
    b, L, _ = position_logits.shape

    flat_logits = position_logits.reshape(b * L, N_LETTERS)
    flat_targets = targets.reshape(b * L)
    flat_valid = hidden_mask.reshape(b * L)
    loss_a = F.cross_entropy(
        flat_logits[flat_valid], flat_targets[flat_valid], label_smoothing=0.02
    )
    loss_b = F.binary_cross_entropy_with_logits(word_logits, present_mask.float())
    return loss_a, loss_b
