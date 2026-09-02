"""Pattern-filtered dictionary voting.

Keeps every training word that is still consistent with the board and votes on
the next letter by how many of those candidates contain it. Consistency is
stricter than simple pattern matching: because a correct guess reveals *all*
occurrences of a letter, a candidate must place each already-guessed letter in
exactly the revealed positions and nowhere else.

Distinct games collide heavily on early turns -- every eight-letter word starts
from the same blank board -- so results are memoised on the board state. That
turns the expensive opening turns into a handful of computations and leaves
only the cheap, already-narrow endgame states to be solved individually.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..encoding import (
    LETTER_OFFSET,
    MASK_ID,
    N_LETTERS,
    OTHER_ID,
    PAD_ID,
    encode_words,
)
from ..simulator import Observation
from .base import Policy
from .frequency import LengthFrequencyPolicy


class DictionaryPolicy(Policy):
    def __init__(self, words: list[str], max_len: int = 32, cache_size: int = 400_000):
        self.fallback = LengthFrequencyPolicy(words, max_len=max_len)
        by_length: dict[int, list[str]] = defaultdict(list)
        for word in set(words):
            if len(word) <= max_len:
                by_length[len(word)].append(word)

        self.buckets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for length, group in by_length.items():
            chars, _, present = encode_words(group, max_len=length)
            self.buckets[length] = (chars.astype(np.int16), present)

        self._cache: dict[tuple, np.ndarray | None] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------------ #
    def _candidate_votes(self, tokens_row: np.ndarray, absent: np.ndarray):
        """Letter vote vector for one board, or None when nothing matches."""
        length = int((tokens_row != PAD_ID).sum())
        bucket = self.buckets.get(length)
        if bucket is None:
            return None

        key = (length, tokens_row[:length].tobytes(), np.packbits(absent).tobytes())
        if key in self._cache:
            return self._cache[key]

        chars, present = bucket
        keep = np.ones(chars.shape[0], dtype=bool)

        for pos in range(length):
            tok = int(tokens_row[pos])
            if tok == MASK_ID or tok == PAD_ID:
                continue
            if tok == OTHER_ID:
                keep &= chars[:, pos] < 0
            else:
                keep &= chars[:, pos] == (tok - LETTER_OFFSET)
            if not keep.any():
                break

        if keep.any():
            # Revealed letters must not also sit under a still-hidden slot,
            # and absent letters must not appear at all.
            revealed_letters = {
                int(t) - LETTER_OFFSET
                for t in tokens_row[:length]
                if LETTER_OFFSET <= int(t) < OTHER_ID
            }
            hidden_positions = np.flatnonzero(tokens_row[:length] == MASK_ID)
            if len(hidden_positions):
                sub = chars[:, hidden_positions]
                for letter in revealed_letters:
                    keep &= ~(sub == letter).any(axis=1)
            for letter in np.flatnonzero(absent):
                keep &= ~present[:, letter]

        votes = present[keep].sum(axis=0).astype(np.float64) if keep.any() else None
        if votes is not None:
            votes /= max(keep.sum(), 1)

        if len(self._cache) < self._cache_size:
            self._cache[key] = votes
        return votes

    # ------------------------------------------------------------------ #
    def scores(self, obs: Observation) -> np.ndarray:
        out = self.fallback.scores(obs).copy()
        for i in range(len(obs)):
            votes = self._candidate_votes(obs.tokens[i], obs.absent[i])
            if votes is not None:
                # Offset keeps a live candidate set strictly above the prior.
                out[i] = votes + 1.0
        return out

    def candidate_counts(self, obs: Observation) -> np.ndarray:
        counts = np.zeros(len(obs), dtype=np.int32)
        for i in range(len(obs)):
            votes = self._candidate_votes(obs.tokens[i], obs.absent[i])
            counts[i] = 0 if votes is None else 1
        return counts
