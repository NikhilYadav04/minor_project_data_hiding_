from __future__ import annotations

import numpy as np

from . import gf256


def _check_keys(K: np.ndarray, length: int) -> np.ndarray:
    keys = np.asarray(K, dtype=np.uint8).reshape(-1)
    if len(keys) != length:
        raise ValueError("evaluation-key length does not match pair count")
    if np.any(keys == 0):
        raise ValueError("evaluation points must be nonzero")
    return keys


def share_pairs(pairs, K, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.uint8)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (P, 2)")
    if not 2 <= k <= n:
        raise ValueError("require 2 <= k <= n")
    keys = _check_keys(K, len(pairs))
    P = len(pairs)
    coefficients = np.asarray(rng.integers(0, 256, size=(max(0, k - 2), P), dtype=np.uint8))
    out = np.empty((n, P), dtype=np.uint8)
    for share_index in range(n):
        x = (keys.astype(np.uint16) + share_index).astype(np.uint8)
        value = pairs[:, 0].copy()
        value ^= np.asarray(gf256.mul(pairs[:, 1], x), dtype=np.uint8)
        power = x.copy()
        for degree in range(2, k):
            power = np.asarray(gf256.mul(power, x), dtype=np.uint8)
            value ^= np.asarray(gf256.mul(coefficients[degree - 2], power), dtype=np.uint8)
        out[share_index] = value
    return out


def recover_pairs(shares, idx, K, k: int) -> np.ndarray:
    values = np.asarray(shares, dtype=np.uint8)
    if values.ndim != 2 or values.shape[0] != k:
        raise ValueError("shares must have shape (k, P)")
    indices = [int(i) for i in idx]
    if len(indices) != k or len(set(indices)) != k:
        raise ValueError("idx must contain k unique share indices")
    keys = _check_keys(K, values.shape[1])
    constant = np.zeros(values.shape[1], dtype=np.uint8)
    slope = np.zeros(values.shape[1], dtype=np.uint8)
    for j, share_index in enumerate(indices):
        xj = (keys.astype(np.uint16) + share_index).astype(np.uint8)
        p0 = np.ones(values.shape[1], dtype=np.uint8)
        p1 = np.zeros(values.shape[1], dtype=np.uint8)
        denominator = np.ones(values.shape[1], dtype=np.uint8)
        for m, other_index in enumerate(indices):
            if m == j:
                continue
            xm = (keys.astype(np.uint16) + other_index).astype(np.uint8)
            old_p0 = p0
            p0 = np.asarray(gf256.mul(old_p0, xm), dtype=np.uint8)
            p1 = old_p0 ^ np.asarray(gf256.mul(p1, xm), dtype=np.uint8)
            denominator = np.asarray(gf256.mul(denominator, xj ^ xm), dtype=np.uint8)
        l0 = np.asarray(gf256.div(p0, denominator), dtype=np.uint8)
        l1 = np.asarray(gf256.div(p1, denominator), dtype=np.uint8)
        constant ^= np.asarray(gf256.mul(values[j], l0), dtype=np.uint8)
        slope ^= np.asarray(gf256.mul(values[j], l1), dtype=np.uint8)
    return np.stack((constant, slope), axis=1)
