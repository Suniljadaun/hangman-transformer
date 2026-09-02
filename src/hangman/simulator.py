"""Batched, no-peeking Hangman simulator.

Every game in a batch advances in lockstep: at each step the policy receives
only what a real player would see (the masked board, the letters already
guessed, which of them were absent, and lives remaining) and returns one letter
per active game. The secret word lives exclusively inside this module and is
never handed to a policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding import MAX_LIVES, N_LETTERS, board_tokens, encode_words


@dataclass(frozen=True)
class Observation:
    """Everything a policy is allowed to see."""

    tokens: np.ndarray      # (b, max_len) int64 board rendering
    guessed: np.ndarray     # (b, 26) bool  every letter tried so far
    absent: np.ndarray      # (b, 26) bool  tried and known not to occur
    lives_left: np.ndarray  # (b,) int32
    lengths: np.ndarray     # (b,) int32

    def __len__(self) -> int:
        return self.tokens.shape[0]


@dataclass
class GameResult:
    guesses: list[str]      # chronological guess string per word
    won: np.ndarray         # (n,) bool
    wrong: np.ndarray       # (n,) int32  strikes accrued before termination

    @property
    def win_rate(self) -> float:
        return float(self.won.mean() * 100.0)

    @property
    def total_wrong(self) -> int:
        return int(self.wrong.sum())

    def summary(self) -> str:
        return (
            f"win_rate={self.win_rate:.3f}%  "
            f"mean_wrong={self.wrong.mean():.3f}  "
            f"total_wrong={self.total_wrong}"
        )


def play(words, policy, max_len: int | None = None, on_step=None) -> GameResult:
    """Play one game per word and return the guess sequences and outcomes.

    `policy` must expose ``guess(Observation) -> (b,) int array`` of letter
    indices, and must never return a letter already marked in ``guessed``.
    """
    chars, lengths, present = encode_words(list(words), max_len=max_len)
    n = chars.shape[0]

    revealed = np.zeros((n, N_LETTERS), dtype=bool)
    guessed = np.zeros((n, N_LETTERS), dtype=bool)
    wrong = np.zeros(n, dtype=np.int32)
    sequences: list[list[str]] = [[] for _ in range(n)]

    # A word made entirely of non-letters is won before it starts.
    solved = ~(present & ~revealed).any(axis=1)
    active = ~solved

    while active.any():
        idx = np.flatnonzero(active)
        obs = Observation(
            tokens=board_tokens(chars[idx], revealed[idx]),
            guessed=guessed[idx],
            absent=guessed[idx] & ~revealed[idx],
            lives_left=(MAX_LIVES - wrong[idx]).astype(np.int32),
            lengths=lengths[idx],
        )
        letters = np.asarray(policy.guess(obs), dtype=np.intp)
        if letters.shape != idx.shape:
            raise ValueError("policy returned the wrong number of guesses")
        if guessed[idx, letters].any():
            raise ValueError("policy repeated a letter, which would cost a life")

        if on_step is not None:
            on_step(idx, obs, letters)

        hit = present[idx, letters]
        guessed[idx, letters] = True
        revealed[idx[hit], letters[hit]] = True
        wrong[idx[~hit]] += 1

        for k, i in enumerate(idx):
            sequences[i].append(chr(ord("a") + letters[k]))

        solved[idx] = ~(present[idx] & ~revealed[idx]).any(axis=1)
        active[idx] = ~solved[idx] & (wrong[idx] < MAX_LIVES)

    return GameResult(
        guesses=["".join(s) for s in sequences],
        won=solved,
        wrong=wrong,
    )
