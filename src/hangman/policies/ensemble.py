"""Combining policies."""
from __future__ import annotations

import numpy as np

from ..simulator import Observation
from .base import Policy


class FallbackPolicy(Policy):
    """Use `primary` where it is confident, otherwise `secondary`.

    Scores are z-normalised per row before blending so that policies on
    different scales (probabilities vs logits) can be mixed.
    """

    def __init__(self, primary: Policy, secondary: Policy, weight: float = 0.5):
        self.primary = primary
        self.secondary = secondary
        self.weight = weight

    def scores(self, obs: Observation) -> np.ndarray:
        a = _standardise(np.asarray(self.primary.scores(obs), dtype=np.float64))
        b = _standardise(np.asarray(self.secondary.scores(obs), dtype=np.float64))
        return self.weight * a + (1.0 - self.weight) * b


class EpsilonMixPolicy(Policy):
    """Follow `policy` most of the time, explore with `explorer` otherwise.

    Used only when generating training states: a policy that never deviates
    visits a thin slice of the state space, and the network then has no idea
    what to do on the boards it will actually meet after an unlucky guess.
    """

    def __init__(
        self,
        policy: Policy,
        explorer: Policy,
        epsilon: float = 0.02,
        noise: float = 0.05,
        seed: int = 0,
    ):
        self.policy = policy
        self.explorer = explorer
        self.epsilon = epsilon   # probability of deferring to the explorer
        self.noise = noise       # Gumbel jitter on the standardised scores
        self.rng = np.random.default_rng(seed)

    def scores(self, obs: Observation) -> np.ndarray:
        main = _standardise(np.asarray(self.policy.scores(obs), dtype=np.float64))
        alt = _standardise(np.asarray(self.explorer.scores(obs), dtype=np.float64))
        noise = self.rng.gumbel(size=main.shape) * self.noise
        take_alt = self.rng.random(len(obs)) < self.epsilon
        out = np.where(take_alt[:, None], alt, main) + noise
        return out


def _standardise(x: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(x), x, np.nan)
    mean = np.nanmean(finite, axis=1, keepdims=True)
    std = np.nanstd(finite, axis=1, keepdims=True)
    return (x - mean) / np.clip(std, 1e-6, None)
