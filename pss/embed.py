from __future__ import annotations

import struct
import zlib

import numpy as np

from .config import HEADER_BYTES
from .exceptions import CapacityError, ExtractionError


def _blocks_from_image(image: np.ndarray) -> np.ndarray:
    m, n = image.shape
    return image.reshape(m // 2, 2, n // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 8)


def _image_from_blocks(blocks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m, n = shape
    return blocks.reshape(m // 2, n // 4, 2, 4).transpose(0, 2, 1, 3).reshape(m, n)


def expand_share(shared: np.ndarray, rng_fill: np.random.Generator) -> np.ndarray:
    low = np.asarray(shared, dtype=np.uint8)
    if low.ndim != 2 or low.shape[0] < 1 or low.shape[1] < 1:
        raise ValueError("shared image must be two-dimensional")
    blocks = rng_fill.integers(0, 256, size=(low.size, 8), dtype=np.uint8)
    blocks[:, 0] = low.reshape(-1)
    return _image_from_blocks(blocks, (2 * low.shape[0], 4 * low.shape[1]))


def extract_shared(expanded: np.ndarray) -> np.ndarray:
    image = np.asarray(expanded, dtype=np.uint8)
    blocks = _blocks_from_image(image)
    return blocks[:, 0].reshape(image.shape[0] // 2, image.shape[1] // 4)


def _redundant_view(share: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    image = np.asarray(share, dtype=np.uint8)
    blocks = _blocks_from_image(image)
    return blocks[:, 1:].reshape(-1), image.shape


def _write_redundant(blocks: np.ndarray, offset: int, payload: np.ndarray) -> None:
    """Write through the non-contiguous redundant-block slice safely."""
    redundant = blocks[:, 1:].copy()
    flat = redundant.reshape(-1)
    flat[offset : offset + len(payload)] = np.asarray(payload, dtype=np.uint8).reshape(-1)
    blocks[:, 1:] = redundant


def parse_header(share: np.ndarray) -> dict[str, int]:
    redundant, shape = _redundant_view(share)
    capacity = len(redundant)
    if capacity < HEADER_BYTES:
        raise CapacityError("share has no room for the core header")
    length_sc, length_sa, bitlen_c, reserved = struct.unpack(">IIII", redundant[:HEADER_BYTES].tobytes())
    free_offset = HEADER_BYTES + length_sc + length_sa
    if free_offset > capacity:
        raise CapacityError("share header declares payload beyond redundant capacity")
    # S_C contains one byte per shared 16-bit pair; recovery reconstructs two
    # bytes per S_C element.
    if bitlen_c > length_sc * 16:
        raise CapacityError("compressed-code bit length exceeds its shared length")
    return {
        "LEN_SC": length_sc,
        "LEN_SA": length_sa,
        "BITLEN_C": bitlen_c,
        "reserved": reserved,
        "free_offset": free_offset,
        "capacity": capacity,
        "shape_m": shape[0],
        "shape_n": shape[1],
    }


def set_auth_tag(share: np.ndarray, tag: int) -> np.ndarray:
    result = np.asarray(share, dtype=np.uint8).copy()
    blocks = _blocks_from_image(result)
    _write_redundant(blocks, 12, np.frombuffer(struct.pack(">I", int(tag) & 0xFFFFFFFF), dtype=np.uint8))
    return _image_from_blocks(blocks, result.shape)


def embed_core(expanded: np.ndarray, s_c: np.ndarray, s_a: np.ndarray, bitlen_c: int) -> np.ndarray:
    result = np.asarray(expanded, dtype=np.uint8).copy()
    blocks = _blocks_from_image(result)
    redundant = blocks[:, 1:].copy().reshape(-1)
    code = np.asarray(s_c, dtype=np.uint8).reshape(-1)
    msa = np.asarray(s_a, dtype=np.uint8).reshape(-1)
    header = struct.pack(">IIII", len(code), len(msa), int(bitlen_c), 0)
    payload = np.frombuffer(header, dtype=np.uint8)
    payload = np.concatenate((payload, code, msa))
    if len(payload) > len(redundant):
        raise CapacityError(
            f"core payload needs {len(payload)} redundant bytes but only {len(redundant)} are available"
        )
    _write_redundant(blocks, 0, payload)
    return _image_from_blocks(blocks, result.shape)


def extract_core(share: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    info = parse_header(share)
    redundant, _ = _redundant_view(share)
    start_sc = HEADER_BYTES
    start_sa = start_sc + info["LEN_SC"]
    s_c = redundant[start_sc:start_sa].copy()
    s_a = redundant[start_sa : start_sa + info["LEN_SA"]].copy()
    return s_c, s_a, info


def _keystream(seed: int, length: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).integers(0, 256, size=length, dtype=np.uint8)


def hider_embed(final_share: np.ndarray, data: bytes, k_dh: int) -> np.ndarray:
    result = np.asarray(final_share, dtype=np.uint8).copy()
    info = parse_header(result)
    raw = bytes(data)
    capacity = info["capacity"] - info["free_offset"]
    if len(raw) + 8 > capacity:
        raise CapacityError(f"data payload needs {len(raw) + 8} bytes but only {capacity} are available")
    payload = struct.pack(">II", len(raw), zlib.crc32(raw) & 0xFFFFFFFF) + raw
    encrypted = np.frombuffer(payload, dtype=np.uint8) ^ _keystream(k_dh, len(payload))
    blocks = _blocks_from_image(result)
    _write_redundant(blocks, info["free_offset"], encrypted)
    return _image_from_blocks(blocks, result.shape)


def hider_extract(embedded_share: np.ndarray, k_dh: int) -> bytes:
    info = parse_header(embedded_share)
    redundant, _ = _redundant_view(np.asarray(embedded_share, dtype=np.uint8))
    available = info["capacity"] - info["free_offset"]
    if available < 8:
        raise ExtractionError("share has no data-hider header room")
    first = redundant[info["free_offset"] : info["free_offset"] + 4] ^ _keystream(k_dh, 4)
    length = struct.unpack(">I", first.tobytes())[0]
    if length + 8 > available:
        raise ExtractionError("data-hider length is invalid for this share")
    encrypted = redundant[info["free_offset"] : info["free_offset"] + length + 8]
    payload = encrypted ^ _keystream(k_dh, len(encrypted))
    stored_length, stored_crc = struct.unpack(">II", payload[:8].tobytes())
    data = payload[8 : 8 + stored_length].tobytes()
    if stored_length != length or zlib.crc32(data) & 0xFFFFFFFF != stored_crc:
        raise ExtractionError("data-hider CRC mismatch")
    return data
