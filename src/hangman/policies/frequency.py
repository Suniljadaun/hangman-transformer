"""Length-conditioned positional letter frequency -- the statistical floor."""
from __future__ import annotations

import numpy as np

from ..encoding import N_LETTERS
from ..simulator import Observation
from .base import Policy


class LengthFrequencyPolicy(Policy):
    """P(letter occurs in the word | word length), estimated on the train list.

    Deliberately ignores the board. It exists to establish a floor, to seed the
    first round of on-policy state generation, and to back up the dictionary
    policy when its candidate set empties out.
    """

    def __init__(self, words: list[str], max_len: int = 32, smoothing: float = 1.0):
        counts = np.full((max_len + 1, N_LETTERS), smoothing)
        totals = np.full(max_len + 1, 2.0 * smoothing)
        for word in words:
            n = min(len(word), max_len)
            totals[n] += 1.0
            for ch in set(word):
                idx = ord(ch) - 97
                if 0 <= idx < N_LETTERS:
                    counts[n, idx] += 1.0
        self.table = counts / totals[:, None]
        # Length-agnostic fallback for lengths unseen in training.
        self.global_rate = counts.sum(0) / totals.sum()
        self.max_len = max_len

    def scores(self, obs: Observation) -> np.ndarray:
        lengths = np.clip(obs.lengths, 0, self.max_len)
        return self.table[lengths]
