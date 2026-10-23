"""Vectorized arithmetic over GF(2^8)."""

from __future__ import annotations

import numpy as np


def _mul_scalar(a: int, b: int, poly: int) -> int:
    result = 0
    aa = a & 0xFF
    bb = b & 0xFF
    for _ in range(8):
        if bb & 1:
            result ^= aa
        carry = aa & 0x80
        aa = (aa << 1) & 0xFF
        if carry:
            aa ^= poly & 0xFF
        bb >>= 1
    return result & 0xFF


def _pow_scalar(a: int, exponent: int, poly: int) -> int:
    result = 1
    base = a & 0xFF
    e = exponent
    while e:
        if e & 1:
            result = _mul_scalar(result, base, poly)
        base = _mul_scalar(base, base, poly)
        e >>= 1
    return result


def find_primitive(poly: int) -> int:
    """Find a generator of the nonzero multiplicative group."""
    for candidate in range(2, 256):
        if all(_pow_scalar(candidate, 255 // factor, poly) != 1 for factor in (3, 5, 17)):
            return candidate
    raise ValueError(f"no primitive element found for polynomial {poly:#x}")


def build_tables(poly: int) -> tuple[np.ndarray, np.ndarray, int]:
    generator = find_primitive(poly)
    exp = np.zeros(510, dtype=np.uint8)
    log = np.full(256, -1, dtype=np.int16)
    value = 1
    for i in range(255):
        exp[i] = value
        log[value] = i
        value = _mul_scalar(value, generator, poly)
    if value != 1:
        raise ValueError("candidate generator did not have order 255")
    exp[255:] = exp[:255]
    return exp, log, generator


class GF256Field:
    def __init__(self, poly: int = 0x11D):
        self.poly = poly
        self.exp, self.log, self.generator = build_tables(poly)

    @staticmethod
    def _scalar_or_array(value: np.ndarray) -> np.ndarray | np.uint8:
        return value.item() if value.ndim == 0 else value

    def mul(self, a, b):
        aa, bb = np.broadcast_arrays(np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8))
        out = np.zeros(aa.size, dtype=np.uint8)
        av = aa.reshape(-1)
        bv = bb.reshape(-1)
        mask = (av != 0) & (bv != 0)
        if np.any(mask):
            indices = (self.log[av[mask]].astype(np.int32) + self.log[bv[mask]].astype(np.int32)) % 255
            out[mask] = self.exp[indices]
        return self._scalar_or_array(out.reshape(aa.shape))

    def inv(self, a):
        aa = np.asarray(a, dtype=np.uint8)
        if np.any(aa == 0):
            raise ZeroDivisionError("zero has no multiplicative inverse in GF(256)")
        out = self.exp[255 - self.log[aa].astype(np.int32)]
        return self._scalar_or_array(np.asarray(out, dtype=np.uint8))

    def div(self, a, b):
        return self.mul(a, self.inv(b))

    def pow(self, a, exponent: int):
        if exponent < 0:
            return self.pow(self.inv(a), -exponent)
        aa = np.asarray(a, dtype=np.uint8)
        if exponent == 0:
            return self._scalar_or_array(np.ones(aa.shape, dtype=np.uint8))
        result = np.ones(aa.shape, dtype=np.uint8)
        base = aa.copy()
        e = int(exponent)
        while e:
            if e & 1:
                result = np.asarray(self.mul(result, base), dtype=np.uint8)
            base = np.asarray(self.mul(base, base), dtype=np.uint8)
            e >>= 1
        return self._scalar_or_array(result)


DEFAULT_FIELD = GF256Field(0x11D)


def mul(a, b):
    return DEFAULT_FIELD.mul(a, b)


def inv(a):
    return DEFAULT_FIELD.inv(a)


def div(a, b):
    return DEFAULT_FIELD.div(a, b)


def pow(a, exponent: int):
    return DEFAULT_FIELD.pow(a, exponent)
