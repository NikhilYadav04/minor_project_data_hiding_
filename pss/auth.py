"""Keyed integrity tags for detecting modified final shares."""

from __future__ import annotations

import hashlib
import struct

import numpy as np

from .embed import extract_core, extract_shared, set_auth_tag
from .exceptions import InvalidShareError


def _auth_message(share: np.ndarray, share_index: int) -> bytes:
    s_c, s_a, info = extract_core(share)
    lr = extract_shared(np.asarray(share, dtype=np.uint8))
    return b"PSS-AUTH-v1" + struct.pack(">I", int(share_index)) + struct.pack(">III", info["LEN_SC"], info["LEN_SA"], info["BITLEN_C"]) + lr.tobytes() + s_c.tobytes() + s_a.tobytes()


def auth_tag(share: np.ndarray, share_index: int, auth_key: int) -> int:
    key = struct.pack(">Q", int(auth_key) & 0xFFFFFFFFFFFFFFFF)
    digest = hashlib.blake2s(_auth_message(share, share_index), key=key, digest_size=4).digest()
    return struct.unpack(">I", digest)[0]


def stamp_share(share: np.ndarray, share_index: int, auth_key: int) -> np.ndarray:
    result = np.asarray(share, dtype=np.uint8).copy()
    return set_auth_tag(result, auth_tag(result, share_index, auth_key))


def verify_share(share: np.ndarray, share_index: int, auth_key: int) -> bool:
    try:
        _, _, info = extract_core(share)
        return int(info["reserved"]) == auth_tag(share, share_index, auth_key)
    except (ValueError, TypeError, struct.error):
        return False


def require_valid_share(share: np.ndarray, share_index: int, auth_key: int) -> None:
    if not verify_share(share, share_index, auth_key):
        raise InvalidShareError(f"share {share_index + 1} failed integrity verification")


def find_valid_shares(shares, keys, indexes=None) -> list[int]:
    """Return original share indexes that pass integrity verification.

    ``indexes`` lets callers validate a subset after a share is missing while
    preserving the original evaluation-point identity.
    """
    actual_indexes = list(range(len(shares))) if indexes is None else [int(index) for index in indexes]
    if len(actual_indexes) != len(shares):
        raise ValueError("indexes must have one entry per share")
    return [index for index, share in zip(actual_indexes, shares) if verify_share(share, index, keys.auth_seed(index))]
