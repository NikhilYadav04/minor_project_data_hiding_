import numpy as np
import pytest

from pss.exceptions import CapacityError, InsufficientSharesError, InvalidKeyError
from pss.keys import KeyBundle
from pss.pipeline import owner_encode, recover_stage2, recover_stage3


def test_negative_stage_requirements_and_wrong_key(image64):
    keys = KeyBundle(4)
    output = owner_encode(image64, (0, 0, 12, 12), keys)
    with pytest.raises(InsufficientSharesError):
        recover_stage2(output.final_shares[:3], keys)
    wrong = KeyBundle(5)
    with pytest.raises((InvalidKeyError, ValueError)):
        recover_stage2(output.final_shares, wrong)
    with pytest.raises((InvalidKeyError, ValueError)):
        recover_stage3(output.final_shares, wrong)


def test_oversized_msa_is_rejected(image64):
    image = np.random.default_rng(123).integers(0, 256, size=(64, 64), dtype=np.uint8)
    with pytest.raises(CapacityError):
        owner_encode(image, (0, 0, 64, 64), KeyBundle(8))
