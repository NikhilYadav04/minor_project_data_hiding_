import numpy as np
import pytest

from pss.embed import embed_core, extract_core, expand_share, extract_shared, hider_embed, hider_extract
from pss.exceptions import CapacityError, ExtractionError


def test_expansion_and_hider_round_trip():
    low = np.arange(256, dtype=np.uint8).reshape(16, 16)
    expanded = expand_share(low, np.random.default_rng(8))
    assert np.array_equal(extract_shared(expanded), low)
    final = embed_core(expanded, np.arange(20, dtype=np.uint8), np.arange(10, dtype=np.uint8), 143)
    code, msa, header = extract_core(final)
    assert np.array_equal(code, np.arange(20, dtype=np.uint8))
    assert np.array_equal(msa, np.arange(10, dtype=np.uint8))
    hidden = hider_embed(final, b"hello", 77)
    assert hider_extract(hidden, 77) == b"hello"
    with pytest.raises(ExtractionError):
        hider_extract(hidden, 78)
