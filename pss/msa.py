from __future__ import annotations

import struct
from dataclasses import dataclass

import cv2
import numpy as np

from .exceptions import InvalidKeyError


@dataclass(frozen=True)
class MSAData:
    masked: np.ndarray
    pairs: np.ndarray
    rect: tuple[int, int, int, int] | None


def validate_rect(shape: tuple[int, int], rect: tuple[int, int, int, int] | None):
    if rect is None:
        return None
    if len(rect) != 4:
        raise ValueError("rect must be (top, left, height, width)")
    top, left, height, width = (int(v) for v in rect)
    m, n = shape
    if top < 0 or left < 0 or height < 1 or width < 1:
        raise ValueError("MSA rectangle must have positive area and nonnegative origin")
    if top + height > m or left + width > n:
        raise ValueError("MSA rectangle must lie inside the image")
    if top + height > 65535 or left + width > 65535:
        raise ValueError("MSA coordinates must fit in uint16 headers")
    return top, left, height, width


def _keystream(seed: int, length: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).integers(0, 256, size=length, dtype=np.uint8)


def _pairs_to_bytes(pairs: np.ndarray) -> np.ndarray:
    values = np.asarray(pairs, dtype=np.uint8)
    if values.size == 0:
        return np.empty(0, dtype=np.uint8)
    return values.reshape(-1, 2).reshape(-1)


def prepare_msa(img: np.ndarray, rect: tuple[int, int, int, int] | None, k_sc: int) -> MSAData:
    image = np.asarray(img, dtype=np.uint8)
    validated = validate_rect(image.shape, rect)
    if validated is None:
        return MSAData(image.copy(), np.empty((0, 2), dtype=np.uint8), None)
    top, left, height, width = validated
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[top : top + height, left : left + width] = 255
    masked = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    outside = mask == 0
    if not np.array_equal(masked[outside], image[outside]):
        raise AssertionError("inpainting changed pixels outside the MSA")
    bottom, right = top + height, left + width
    header = struct.pack(">HHHH", top, left, bottom, right)
    raw = np.frombuffer(header + image[top:bottom, left:right].tobytes(), dtype=np.uint8).copy()
    if len(raw) % 2:
        raw = np.concatenate((raw, np.zeros(1, dtype=np.uint8)))
    encrypted = raw ^ _keystream(k_sc, len(raw))
    return MSAData(masked, encrypted.reshape(-1, 2), validated)


def restore_msa(masked: np.ndarray, pairs: np.ndarray, k_sc: int) -> np.ndarray:
    image = np.asarray(masked, dtype=np.uint8)
    encrypted = _pairs_to_bytes(pairs)
    if len(encrypted) == 0:
        return image.copy()
    raw = encrypted ^ _keystream(k_sc, len(encrypted))
    if len(raw) < 8:
        raise InvalidKeyError("MSA payload is shorter than its coordinate header")
    try:
        top, left, bottom, right = struct.unpack(">HHHH", raw[:8].tobytes())
    except struct.error as exc:
        raise InvalidKeyError("invalid MSA coordinate header") from exc
    m, n = image.shape
    if not (0 <= top < bottom <= m and 0 <= left < right <= n):
        raise InvalidKeyError("MSA coordinates are outside the image")
    area = (bottom - top) * (right - left)
    payload_len = len(raw) - 8
    if payload_len not in (area, area + 1):
        raise InvalidKeyError("MSA byte count does not match its rectangle")
    if payload_len == area + 1 and raw[-1] != 0:
        raise InvalidKeyError("invalid MSA padding byte")
    out = image.copy()
    out[top:bottom, left:right] = raw[8 : 8 + area].reshape(bottom - top, right - left)
    return out


def select_adaptive_rect(
    img: np.ndarray,
    mode: str = "texture",
    height: int = 60,
    width: int = 40,
    stride: int = 16,
) -> tuple[int, int, int, int]:
    """Select a fixed-size MSA window from image content.

    ``texture`` maximizes local variance and ``edge`` maximizes Sobel
    gradient magnitude. Fixed dimensions keep the payload budget comparable
    with the manual selector while the location adapts to the image.
    """
    image = np.asarray(img, dtype=np.uint8)
    m, n = image.shape
    height, width, stride = int(height), int(width), int(stride)
    if mode not in {"texture", "edge"}:
        raise ValueError("adaptive MSA mode must be 'texture' or 'edge'")
    if height < 1 or width < 1 or height > m or width > n:
        raise ValueError("adaptive MSA dimensions must fit inside the image")
    if stride < 1:
        raise ValueError("adaptive MSA stride must be positive")

    values = image.astype(np.float32)
    if mode == "texture":
        mean = cv2.boxFilter(values, cv2.CV_32F, (width, height), normalize=True)
        mean_sq = cv2.boxFilter(values * values, cv2.CV_32F, (width, height), normalize=True)
        score_map = np.maximum(mean_sq - mean * mean, 0.0)
    else:
        gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
        score_map = cv2.boxFilter(np.abs(gx) + np.abs(gy), cv2.CV_32F, (width, height), normalize=True)

    best: tuple[float, int, int] | None = None
    for top in range(0, m - height + 1, stride):
        for left in range(0, n - width + 1, stride):
            score = float(np.mean(score_map[top : top + height, left : left + width]))
            candidate = (score, -top, -left)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise ValueError("could not select an adaptive MSA window")
    return -best[1], -best[2], height, width
