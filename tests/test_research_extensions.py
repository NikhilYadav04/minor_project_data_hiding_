import numpy as np
import pytest

from pss.exceptions import InsufficientSharesError
from pss.keys import KeyBundle
from pss.msa import select_adaptive_rect
from pss.pipeline import owner_encode, recover_available


def test_adaptive_msa_selectors_are_valid_and_reproducible(image64):
    for mode in ("texture", "edge"):
        rect = select_adaptive_rect(image64, mode, height=12, width=10, stride=4)
        top, left, height, width = rect
        assert 0 <= top <= image64.shape[0] - height
        assert 0 <= left <= image64.shape[1] - width
        assert rect == select_adaptive_rect(image64, mode, height=12, width=10, stride=4)


def test_adaptive_owner_mode_records_selected_region(image64):
    output = owner_encode(image64, (0, 0, 12, 10), KeyBundle(707), "med", msa_mode="auto_texture")
    assert output.stats["msa_mode"] == "auto_texture"
    assert output.stats["msa_height"] == 12
    assert output.stats["msa_width"] == 10


def test_resilient_recovery_works_with_one_tampered_or_missing_share(image64):
    keys = KeyBundle(708)
    output = owner_encode(image64, (8, 8, 12, 12), keys, "edge_directed")

    tampered = list(output.final_shares)
    tampered[1] = tampered[1].copy()
    tampered[1][10, 10] ^= 0xFF
    for stage, target in ((2, output.masked), (3, image64)):
        recovered, valid = recover_available(tampered, keys, "edge_directed", stage)
        assert valid == [0, 2, 3]
        assert np.array_equal(recovered, target)

    missing_shares = [output.final_shares[index] for index in (0, 2, 3)]
    recovered, valid = recover_available(missing_shares, keys, "edge_directed", 3, indexes=[0, 2, 3])
    assert valid == [0, 2, 3]
    assert np.array_equal(recovered, image64)

    twice_tampered = list(output.final_shares)
    for index in (0, 1):
        twice_tampered[index] = twice_tampered[index].copy()
        twice_tampered[index][10, 10] ^= 0xFF
    with pytest.raises(InsufficientSharesError):
        recover_available(twice_tampered, keys, "edge_directed", 3)
