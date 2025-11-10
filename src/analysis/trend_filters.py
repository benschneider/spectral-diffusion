"""Trend filters for smoothing diagnostic signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EWMA:
    """Exponentially-weighted moving average with slope estimation."""

    beta: float
    value: Optional[float] = None
    _prev_value: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must be in (0, 1)")

    def update(self, observation: float) -> float:
        observation = float(observation)
        if self.value is None:
            self.value = observation
            self._prev_value = observation
            return observation
        smoothed = self.beta * self.value + (1.0 - self.beta) * observation
        self._prev_value = self.value
        self.value = smoothed
        return smoothed

    @property
    def slope(self) -> float:
        if self.value is None or self._prev_value is None:
            return 0.0
        return float(self.value - self._prev_value)
