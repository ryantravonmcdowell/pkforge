"""Residual error models: the variance of an observation around its prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = {
    "additive": ("sigma_add",),
    "proportional": ("sigma_prop",),
    "combined": ("sigma_add", "sigma_prop"),
}


@dataclass(frozen=True)
class ErrorModel:
    kind: str = "proportional"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"error model must be one of {sorted(KINDS)}")

    @property
    def names(self) -> tuple[str, ...]:
        return KINDS[self.kind]

    def variance(self, f: np.ndarray, sigma, floor: float = 0.0) -> np.ndarray:
        s = np.asarray(sigma, dtype=float)
        if self.kind == "additive":
            v = np.full_like(f, s[0] ** 2, dtype=float)
        elif self.kind == "proportional":
            v = (s[0] * f) ** 2
        else:
            v = s[0] ** 2 + (s[1] * f) ** 2
        return np.maximum(v, floor)

    def dvariance(self, f: np.ndarray, sigma, floor: float = 0.0) -> np.ndarray:
        """d variance / d prediction (zero where the variance floor is active)."""
        s = np.asarray(sigma, dtype=float)
        if self.kind == "additive":
            return np.zeros_like(f, dtype=float)
        sp = s[0] if self.kind == "proportional" else s[1]
        dv = 2 * sp**2 * f
        return np.where(self.variance(f, sigma) > floor, dv, 0.0)

    def d2variance(self, f: np.ndarray, sigma, floor: float = 0.0) -> np.ndarray:
        """Second derivative of the variance with respect to the prediction."""
        s = np.asarray(sigma, dtype=float)
        if self.kind == "additive":
            return np.zeros_like(f, dtype=float)
        sp = s[0] if self.kind == "proportional" else s[1]
        return np.where(self.variance(f, sigma) > floor, 2 * sp**2, 0.0)
