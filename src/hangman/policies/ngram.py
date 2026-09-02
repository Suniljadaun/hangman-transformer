"""Character n-gram back-off model.

The dictionary policy collapses the moment the true word is not in the training
vocabulary -- which, here, is always. An n-gram model degrades gracefully
instead: it never needs to have seen the word, only its neighbourhoods. For
each hidden slot it asks "what letter usually follows -ati- and precedes -n?"
and backs off to shorter contexts when the long ones are unseen.

Per-slot letter distributions are combined across the word by noisy-OR, the
same aggregation the network uses, so the two are directly comparable and
easy to ensemble.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..encoding import LETTER_OFFSET, MASK_ID, N_LETTERS, OTHER_ID, PAD_ID
from ..simulator import Observation
from .base import Policy

BOUNDARY = 26  # word-edge symbol
UNKNOWN = 27   # a still-hidden neighbour


class NGramPolicy(Policy):
    def __init__(self, words: list[str], orders=(4, 3, 2, 1), alpha: float = 0.4):
        self.orders = tuple(sorted(orders, reverse=True))
        self.alpha = alpha
        self.left: dict[bytes, np.ndarray] = {}
        self.right: dict[bytes, np.ndarray] = {}
        left_counts = defaultdict(lambda: np.zeros(N_LETTERS, dtype=np.float32))
        right_counts = defaultdict(lambda: np.zeros(N_LETTERS, dtype=np.float32))
        prior = np.zeros(N_LETTERS, dtype=np.float64)

        for word in words:
            codes = [ord(c) - 97 if "a" <= c <= "z" else -1 for c in word]
            padded = [BOUNDARY] * max(self.orders) + codes + [BOUNDARY] * max(self.orders)
            offset = max(self.orders)
            for i, code in enumerate(codes):
                if code < 0:
                    continue
                prior[code] += 1
                p = offset + i
                for n in self.orders:
                    lkey = bytes(padded[p - n : p])
                    rkey = bytes(padded[p + 1 : p + 1 + n])
                    if all(0 <= c <= BOUNDARY for c in lkey):
                        left_counts[bytes([n]) + lkey][code] += 1
                    if all(0 <= c <= BOUNDARY for c in rkey):
                        right_counts[bytes([n]) + rkey][code] += 1

        self.left = {k: v / v.sum() for k, v in left_counts.items() if v.sum() > 3}
        self.right = {k: v / v.sum() for k, v in right_counts.items() if v.sum() > 3}
        self.prior = prior / prior.sum()

    # ------------------------------------------------------------------ #
    def _slot_distribution(self, codes: list[int], i: int) -> np.ndarray:
        pad = max(self.orders)
        padded = [BOUNDARY] * pad + codes + [BOUNDARY] * pad
        p = pad + i

        dist = self.prior.copy()
        for table, key_fn in (
            (self.left, lambda n: bytes([n]) + bytes(padded[p - n : p])),
            (self.right, lambda n: bytes([n]) + bytes(padded[p + 1 : p + 1 + n])),
        ):
            for n in self.orders:
                window = padded[p - n : p] if table is self.left else padded[p + 1 : p + 1 + n]
                if any(c == UNKNOWN for c in window):
                    continue
                hit = table.get(key_fn(n))
                if hit is not None:
                    dist = self.alpha * dist + (1 - self.alpha) * hit
                    break
        return dist

    def scores(self, obs: Observation) -> np.ndarray:
        out = np.zeros((len(obs), N_LETTERS), dtype=np.float64)
        for b in range(len(obs)):
            row = obs.tokens[b]
            codes: list[int] = []
            hidden: list[int] = []
            for j, tok in enumerate(row):
                if tok == PAD_ID:
                    break
                if tok == MASK_ID:
                    codes.append(UNKNOWN)
                    hidden.append(j)
                elif tok == OTHER_ID:
                    codes.append(BOUNDARY)
                else:
                    codes.append(int(tok) - LETTER_OFFSET)

            log_none = np.zeros(N_LETTERS)
            for j in hidden:
                dist = self._slot_distribution(codes, j)
                log_none += np.log1p(-np.clip(dist, 0, 1 - 1e-9))
            out[b] = -np.expm1(log_none)          # P(letter appears somewhere)
            out[b][obs.absent[b]] = -1.0
        return out
