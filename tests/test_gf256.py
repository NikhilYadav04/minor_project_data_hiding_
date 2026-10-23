import numpy as np
import pytest

from pss.gf256 import GF256Field


def test_paper_example_under_11d():
    field = GF256Field(0x11D)
    assert [int(field.mul(73, 1) ^ field.mul(76, x) ^ field.mul(106, field.pow(x, 2))) for x in (110, 111, 112, 113)] == [236, 202, 53, 19]
    assert [int(field.mul(77, 1) ^ field.mul(78, x) ^ field.mul(184, field.pow(x, 2))) for x in (6, 7, 8, 9)] == [97, 151, 38, 208]


def test_known_11b_product():
    assert int(GF256Field(0x11B).mul(0x53, 0xCA)) == 0x01


@pytest.mark.parametrize("poly", [0x11B, 0x11D])
def test_inverse_and_distributivity(poly):
    field = GF256Field(poly)
    rng = np.random.default_rng(4)
    a = rng.integers(1, 256, 10000, dtype=np.uint8)
    b = rng.integers(0, 256, 10000, dtype=np.uint8)
    c = rng.integers(0, 256, 10000, dtype=np.uint8)
    assert np.array_equal(field.mul(a, field.inv(a)), np.ones_like(a))
    assert np.array_equal(field.mul(a, b ^ c), field.mul(a, b) ^ field.mul(a, c))
