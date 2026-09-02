"""On-policy training-state collection (the DAgger loop).

A board state is fully determined by the secret word plus the set of letters
already guessed, so we store nine bytes per state -- a word index, a 26-bit
guess mask and the lives remaining -- and reconstruct tokens and labels on the
fly. Two million states cost about twenty megabytes.

Why on-policy: masking letters at random produces boards that no real game ever
reaches. Training on them teaches the network to solve a different problem. We
instead play games with the current policy, record the boards it actually
lands on, and aggregate those across rounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .encoding import MASK_ID, N_LETTERS, board_tokens, encode_words
from .simulator import play


@dataclass
class StateBuffer:
    word_index: np.ndarray   # (m,) int32
    guess_mask: np.ndarray   # (m,) uint32  bit i set => letter i was tried
    lives: np.ndarray        # (m,) int8

    def __len__(self) -> int:
        return len(self.word_index)

    @staticmethod
    def concat(buffers: list["StateBuffer"]) -> "StateBuffer":
        return StateBuffer(
            word_index=np.concatenate([b.word_index for b in buffers]),
            guess_mask=np.concatenate([b.guess_mask for b in buffers]),
            lives=np.concatenate([b.lives for b in buffers]),
        )

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            word_index=self.word_index,
            guess_mask=self.guess_mask,
            lives=self.lives,
        )

    @staticmethod
    def load(path: str) -> "StateBuffer":
        z = np.load(path)
        return StateBuffer(z["word_index"], z["guess_mask"], z["lives"])


_BITS = (1 << np.arange(N_LETTERS)).astype(np.uint32)


def collect(words: list[str], policy, max_len: int = 32, chunk: int = 20_000):
    """Play every word with `policy` and record every board state it visits."""
    parts: list[StateBuffer] = []
    results = []
    for start in range(0, len(words), chunk):
        batch = words[start : start + chunk]
        rows_idx, rows_mask, rows_lives = [], [], []

        def on_step(idx, obs, letters, _offset=start):
            rows_idx.append(idx.astype(np.int32) + _offset)
            rows_mask.append((obs.guessed.astype(np.uint32) * _BITS).sum(axis=1))
            rows_lives.append(obs.lives_left.astype(np.int8))

        results.append(play(batch, policy, max_len=max_len, on_step=on_step))
        parts.append(
            StateBuffer(
                np.concatenate(rows_idx),
                np.concatenate(rows_mask),
                np.concatenate(rows_lives),
            )
        )

    buffer = StateBuffer.concat(parts)
    won = np.concatenate([r.won for r in results])
    wrong = np.concatenate([r.wrong for r in results])
    return buffer, {"win_rate": float(won.mean() * 100), "mean_wrong": float(wrong.mean())}


class BoardBatcher:
    """Vectorised batch construction straight from the compact state buffer.

    A per-item Dataset would rebuild one board at a time in Python and starve
    the GPU; this builds a whole batch with numpy in one go. Batches are
    length-bucketed so that padding stays near zero -- word lengths here run
    from 1 to 29, so naive batching would waste most of the sequence budget.
    """

    def __init__(
        self,
        words: list[str],
        buffer: StateBuffer,
        max_len: int = 32,
        batch_size: int = 512,
        bucket: bool = True,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.chars, self.lengths, self.present = encode_words(words, max_len=max_len)
        self.buffer = buffer
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        order = np.arange(len(buffer))
        if bucket:
            order = order[np.argsort(self.lengths[buffer.word_index], kind="stable")]
        n_batches = len(order) // batch_size if drop_last else -(-len(order) // batch_size)
        self.batches = [
            order[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        order = self.rng.permutation(len(self.batches)) if self.shuffle else range(len(self.batches))
        for b in order:
            yield self.build(self.batches[b])

    def build(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        w = self.buffer.word_index[rows]
        guessed = (self.buffer.guess_mask[rows, None] & _BITS) > 0     # (b, 26)
        present = self.present[w]
        revealed = guessed & present

        lengths = self.lengths[w]
        width = max(int(lengths.max()), 1)
        chars = self.chars[w, :width]
        tokens = board_tokens(chars, revealed)
        hidden = tokens == MASK_ID
        labels = np.where(hidden, np.maximum(chars, 0), -100).astype(np.int64)

        return {
            "tokens": torch.from_numpy(tokens),
            "guessed": torch.from_numpy(guessed.astype(np.float32)),
            "absent": torch.from_numpy((guessed & ~present).astype(np.float32)),
            "lives": torch.from_numpy(self.buffer.lives[rows].astype(np.int64)),
            "length": torch.from_numpy(lengths.astype(np.int64)),
            "labels": torch.from_numpy(labels),
            "hidden": torch.from_numpy(hidden),
            "target_present": torch.from_numpy(present & ~revealed),
        }
