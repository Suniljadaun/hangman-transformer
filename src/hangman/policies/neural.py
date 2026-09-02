"""The learned policy: run HangmanNet over a batch of boards."""
from __future__ import annotations

import numpy as np
import torch

from ..simulator import Observation
from .base import Policy


class NeuralPolicy(Policy):
    def __init__(self, model, device: str = "cpu", fusion: float = 0.5, chunk: int = 8192):
        self.model = model.to(device).eval()
        self.device = device
        self.fusion = fusion
        self.chunk = chunk

    @torch.no_grad()
    def scores(self, obs: Observation) -> np.ndarray:
        out = np.empty((len(obs), 26), dtype=np.float32)
        for start in range(0, len(obs), self.chunk):
            end = min(start + self.chunk, len(obs))
            sl = slice(start, end)
            width = int((obs.tokens[sl] != 0).sum(axis=1).max())
            width = max(width, 1)
            tokens = torch.from_numpy(obs.tokens[sl, :width]).long().to(self.device)
            guessed = torch.from_numpy(obs.guessed[sl]).float().to(self.device)
            absent = torch.from_numpy(obs.absent[sl]).float().to(self.device)
            lives = torch.from_numpy(obs.lives_left[sl]).long().to(self.device)
            lengths = torch.from_numpy(obs.lengths[sl]).long().to(self.device)
            s = self.model.score(tokens, guessed, absent, lives, lengths, self.fusion)
            out[sl] = s.float().cpu().numpy()
        return out
