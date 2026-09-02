"""Vocabulary loading, auditing and splitting."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_words(path: str | Path, lower: bool = True) -> list[str]:
    """Read a one-word-per-line list, preserving order and dropping blanks.

    Order matters: `word_id` in the submission is the line index of test.txt,
    so this must never sort, deduplicate or reorder.
    """
    words: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            word = line.strip()
            if lower:
                word = word.lower()
            if word:
                words.append(word)
    return words


@dataclass
class VocabAudit:
    n_words: int
    n_unique: int
    min_len: int
    max_len: int
    mean_len: float
    charset: str
    non_letter_words: int
    length_hist: dict[int, int]

    def __str__(self) -> str:
        return (
            f"words={self.n_words} unique={self.n_unique} "
            f"len={self.min_len}..{self.max_len} (mean {self.mean_len:.2f})\n"
            f"charset={self.charset!r}\n"
            f"words containing non a-z characters: {self.non_letter_words}"
        )


def audit(words: list[str]) -> VocabAudit:
    lengths = [len(w) for w in words]
    charset = sorted({c for w in words for c in w})
    non_letter = sum(1 for w in words if not w.isalpha() or not w.isascii())
    return VocabAudit(
        n_words=len(words),
        n_unique=len(set(words)),
        min_len=min(lengths),
        max_len=max(lengths),
        mean_len=float(np.mean(lengths)),
        charset="".join(charset),
        non_letter_words=non_letter,
        length_hist=dict(sorted(Counter(lengths).items())),
    )


def overlap(train: list[str], test: list[str]) -> dict[str, float]:
    """How much of the test list is memorisable from the training list."""
    train_set = set(train)
    hits = sum(1 for w in test if w in train_set)
    return {
        "test_words_in_train": hits,
        "fraction": hits / max(len(test), 1),
    }


def split_holdout(words: list[str], n_holdout: int = 10_000, seed: int = 0):
    """Disjoint train / holdout split.

    The holdout exists because test.txt is a validation list we are forbidden
    to train on; the holdout gives an honest generalisation signal that is
    also disjoint from every state we generate during training.
    """
    rng = np.random.default_rng(seed)
    unique = sorted(set(words))
    order = rng.permutation(len(unique))
    holdout = [unique[i] for i in order[:n_holdout]]
    train = [unique[i] for i in order[n_holdout:]]
    return train, holdout


def find_competition_dir(filenames=("train.txt", "test.txt")) -> Path:
    """Locate the competition data wherever Kaggle mounted it.

    Kaggle exposes competition data under /kaggle/input/<slug>/ on some
    kernels and /kaggle/input/competitions/<slug>/ on others, and the local
    checkout keeps a copy in ./data. Searching beats hard-coding a path that
    fails on the first run.
    """
    roots = [
        Path("/kaggle/input"),
        Path("/kaggle/input/competitions"),
        Path("data"),
        Path("."),
    ]
    for root in roots:
        if not root.exists():
            continue
        if all((root / name).exists() for name in filenames):
            return root
        for child in sorted(root.iterdir()):
            if child.is_dir() and all((child / name).exists() for name in filenames):
                return child
    raise FileNotFoundError(
        f"could not find {filenames} under any of {[str(r) for r in roots]}"
    )
