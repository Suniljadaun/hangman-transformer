"""Correctness checks on the learning path.

The wiring check that matters is whether the network can learn the states it is
actually shown: after training on a handful of words, the letter it picks must
be a letter the word still needs, on those same states, essentially every time.
A shape bug, a label misalignment or a broken fusion in `score` fails this.

Game win rate on the same words is deliberately *not* the assertion. A model
trained on one policy's trajectories and then asked to follow its own visits
states it never saw, and on twelve words there is nothing to generalise from --
that gap is the whole reason the training loop aggregates on-policy states, and
folding it into a unit test would make the test measure the wrong thing.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from hangman.model import ModelConfig
from hangman.policies import LengthFrequencyPolicy
from hangman.states import BoardBatcher, collect
from hangman.train import TrainConfig, train

WORDS = [
    "hangman", "transformer", "letter", "guessing", "vocabulary", "policy",
    "network", "sequence", "training", "position", "gradient", "encoder",
]


def _fit_tiny_model():
    torch.manual_seed(0)
    buffer, _ = collect(WORDS, LengthFrequencyPolicy(WORDS, max_len=16), max_len=16)
    assert len(buffer) > len(WORDS), "state collection produced nothing"

    batcher = BoardBatcher(WORDS, buffer, max_len=16, batch_size=32,
                           bucket=False, drop_last=False, seed=0)
    cfg = TrainConfig(epochs=200, batch_size=32, lr=1e-3, amp=False,
                      log_every=10_000, warmup_frac=0.1,
                      model=ModelConfig(d_model=64, n_layers=2, n_heads=4,
                                        d_ff=128, dropout=0.0, max_len=16))
    return train(batcher, cfg, device="cpu").eval(), batcher


def _top1_accuracy(model, batcher, fusion):
    """Fraction of states where the chosen letter is one the word still needs."""
    hits = total = 0
    with torch.no_grad():
        for rows in batcher.batches:
            b = batcher.build(rows)
            scores = model.score(b["tokens"], b["guessed"], b["absent"],
                                 b["lives"], b["length"], fusion=fusion)
            picked = scores.argmax(1)
            hits += int(b["target_present"][torch.arange(len(picked)), picked].sum())
            total += len(picked)
    return hits / total


def test_learns_the_states_it_is_shown():
    model, batcher = _fit_tiny_model()
    for fusion in (0.0, 0.5, 1.0):
        accuracy = _top1_accuracy(model, batcher, fusion)
        assert accuracy > 0.95, f"fusion={fusion}: only {accuracy:.1%} top-1"


def test_score_never_returns_an_already_guessed_letter():
    model, batcher = _fit_tiny_model()
    with torch.no_grad():
        for rows in batcher.batches:
            b = batcher.build(rows)
            scores = model.score(b["tokens"], b["guessed"], b["absent"],
                                 b["lives"], b["length"], fusion=0.5)
            picked = scores.argmax(1)
            repeated = b["guessed"][torch.arange(len(picked)), picked]
            assert not bool(repeated.any()), "score() would repeat a letter"
