"""Policy interface.

A policy maps a batch of observations to one letter index per game. It has no
access to the secret word -- that invariant is enforced by the simulator, which
never puts the word into an Observation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..simulator import Observation


class Policy(ABC):
    @abstractmethod
    def scores(self, obs: Observation) -> np.ndarray:
        """Return (b, 26) preference scores. Higher is a better guess."""

    def guess(self, obs: Observation) -> np.ndarray:
        s = np.array(self.scores(obs), dtype=np.float64, copy=True)
        s[obs.guessed] = -np.inf          # a repeated letter costs a life
        return s.argmax(axis=1)
