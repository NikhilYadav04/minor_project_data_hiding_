import numpy as np

from pss.huffman import compress_residuals, decompress_residuals, pack_bits, pad_bits


def test_huffman_random_round_trip():
    rng = np.random.default_rng(6)
    values = np.clip(rng.laplace(0, 5, 3000).round(), -255, 255).astype(np.int32)
    encoded = compress_residuals(values)
    assert np.array_equal(decompress_residuals(encoded.bits, len(values)), values)


def test_huffman_constant_and_canonical():
    values = np.full(100, -3, dtype=np.int32)
    for canonical in (False, True):
        encoded = compress_residuals(values, canonical=canonical)
        assert np.array_equal(decompress_residuals(encoded.bits, len(values)), values)


def test_huffman_all_symbols():
    values = np.arange(-255, 256, dtype=np.int32)
    encoded = compress_residuals(values)
    assert np.array_equal(decompress_residuals(encoded.bits, len(values)), values)
