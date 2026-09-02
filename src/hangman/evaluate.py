"""Evaluation and error analysis."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .simulator import play


def evaluate(words: list[str], policy, max_len: int = 32, chunk: int = 20_000):
    """Play every word and report the competition metrics plus a length breakdown."""
    won, wrong, guesses = [], [], []
    for start in range(0, len(words), chunk):
        result = play(words[start : start + chunk], policy, max_len=max_len)
        won.append(result.won)
        wrong.append(result.wrong)
        guesses.extend(result.guesses)

    won = np.concatenate(won)
    wrong = np.concatenate(wrong)

    by_length: dict[int, list[bool]] = defaultdict(list)
    for word, w in zip(words, won):
        by_length[len(word)].append(bool(w))

    return {
        "win_rate": float(won.mean() * 100),
        "mean_wrong": float(wrong.mean()),
        "total_wrong": int(wrong.sum()),
        "leaderboard_score": float(won.mean() * 100 - wrong.sum() / 1e8),
        "by_length": {
            k: round(100 * float(np.mean(v)), 2) for k, v in sorted(by_length.items())
        },
        "won": won,
        "wrong": wrong,
        "guesses": guesses,
    }


def print_report(metrics: dict) -> None:
    print(f"win rate        : {metrics['win_rate']:.3f}%")
    print(f"mean strikes    : {metrics['mean_wrong']:.3f}")
    print(f"total strikes   : {metrics['total_wrong']:,}")
    print(f"leaderboard     : {metrics['leaderboard_score']:.6f}")
    print("win rate by word length:")
    for length, rate in metrics["by_length"].items():
        print(f"  {length:>2} : {rate:6.2f}%")
