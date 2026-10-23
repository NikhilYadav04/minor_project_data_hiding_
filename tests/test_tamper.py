import numpy as np
import pytest

from pss.auth import find_valid_shares, verify_share
from pss.embed import hider_embed
from pss.exceptions import InvalidShareError
from pss.keys import KeyBundle
from pss.pipeline import owner_encode, recover_stage2, recover_stage3


def test_tampered_share_is_detected_and_clean_recovery_works(image64):
    keys = KeyBundle(515)
    output = owner_encode(image64, (8, 8, 12, 12), keys, "edge_directed")
    assert find_valid_shares(output.final_shares, keys) == [0, 1, 2, 3]
    tampered = list(output.final_shares)
    tampered[1] = tampered[1].copy()
    tampered[1][10, 10] ^= 0xFF
    assert not verify_share(tampered[1], 1, keys.auth_seed(1))
    assert find_valid_shares(tampered, keys) == [0, 2, 3]
    with pytest.raises(InvalidShareError):
        recover_stage2(tampered, keys)
    assert np.array_equal(recover_stage3(output.final_shares, keys), image64)


def test_data_hider_area_does_not_break_integrity_tag(image64):
    keys = KeyBundle(516)
    output = owner_encode(image64, None, keys, "edge_directed")
    embedded = hider_embed(output.final_shares[0], b"integrity-preserving payload", keys.dh_seed(0))
    assert verify_share(embedded, 0, keys.auth_seed(0))
