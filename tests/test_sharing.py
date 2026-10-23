import numpy as np

from pss.keys import KeyBundle
from pss.sharing import recover_pairs, share_pairs


def test_three_of_four_all_combinations():
    rng = np.random.default_rng(2)
    pairs = rng.integers(0, 256, size=(10000, 2), dtype=np.uint8)
    keys = rng.integers(1, 253, size=len(pairs), dtype=np.uint8)
    shares = share_pairs(pairs, keys, 3, 4, np.random.default_rng(3))
    for idx in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
        assert np.array_equal(recover_pairs(shares[idx], idx, keys, 3), pairs)


def test_key_bundle_is_order_independent():
    a = KeyBundle(123)
    b = KeyBundle(123)
    assert np.array_equal(a.lr_keys(100), b.lr_keys(100))
    assert np.array_equal(a.code_keys(100), b.code_keys(100))
    assert a.stream_seed() == b.stream_seed()
