from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import N


_PURPOSES = {
    "K_L": 11,
    "K_C": 12,
    "K_A": 13,
    "K_sc": 21,
    "K_dh": 22,
    "auth": 23,
    "coef_L": 31,
    "coef_C": 32,
    "coef_A": 33,
    "fill": 41,
}


@dataclass(frozen=True)
class KeyBundle:
    """Stable, purpose-separated keys derived from one master seed."""

    master_seed: int

    def _rng(self, purpose: str, index: int = 0) -> np.random.Generator:
        if purpose not in _PURPOSES:
            raise KeyError(purpose)
        sequence = np.random.SeedSequence([int(self.master_seed), _PURPOSES[purpose], int(index)])
        return np.random.default_rng(sequence)

    def _evaluation_keys(self, purpose: str, length: int) -> np.ndarray:
        if length < 0:
            raise ValueError("key length cannot be negative")
        return self._rng(purpose).integers(1, 257 - N, size=length, dtype=np.uint16).astype(np.uint8)

    def lr_keys(self, length: int) -> np.ndarray:
        return self._evaluation_keys("K_L", length)

    def code_keys(self, length: int) -> np.ndarray:
        return self._evaluation_keys("K_C", length)

    def msa_keys(self, length: int) -> np.ndarray:
        return self._evaluation_keys("K_A", length)

    def stream_seed(self, name: str = "K_sc", index: int = 0) -> int:
        return int(self._rng(name, index).integers(0, 2**32, dtype=np.uint64))

    def dh_seed(self, share_index: int = 0) -> int:
        return self.stream_seed("K_dh", share_index)

    def auth_seed(self, share_index: int = 0) -> int:
        return self.stream_seed("auth", share_index)

    def rng_coeff(self, payload: str, share_index: int = 0) -> np.random.Generator:
        return self._rng(f"coef_{payload}", share_index)

    def rng_fill(self, share_index: int = 0) -> np.random.Generator:
        return self._rng("fill", share_index)
