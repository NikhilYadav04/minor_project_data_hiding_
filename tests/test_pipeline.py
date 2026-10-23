import itertools

import numpy as np

from pss.keys import KeyBundle
from pss.pipeline import owner_encode, recover_stage1, recover_stage2, recover_stage3


def test_pipeline_all_stages_and_share_combinations(image64):
    keys = KeyBundle(42)
    output = owner_encode(image64, (10, 12, 15, 17), keys, "edge_directed")
    for idx in itertools.combinations(range(4), 3):
        low = recover_stage1([output.final_shares[i] for i in idx], list(idx), keys, "edge_directed")
        assert np.array_equal(low, output.masked[0::2, 0::2])
    assert np.array_equal(recover_stage2(output.final_shares, keys, "edge_directed"), output.masked)
    assert np.array_equal(recover_stage3(output.final_shares, keys, "edge_directed"), image64)


def test_pipeline_without_msa_and_bilinear(image64):
    keys = KeyBundle(99)
    output = owner_encode(image64, None, keys, "bilinear")
    assert np.array_equal(recover_stage2(output.final_shares, keys, "bilinear"), image64)
    assert np.array_equal(recover_stage3(output.final_shares, keys, "bilinear"), image64)
