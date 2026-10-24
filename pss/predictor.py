from __future__ import annotations

import cv2
import numpy as np

from .deep_models import predict_rcan_style


def get_lr(masked: np.ndarray) -> np.ndarray:
    image = np.asarray(masked, dtype=np.uint8)
    if image.ndim != 2 or image.shape[0] % 2 or image.shape[1] % 2:
        raise ValueError("masked image must have even dimensions")
    return image[0::2, 0::2].copy()


def _padded(L: np.ndarray) -> np.ndarray:
    return np.pad(L, ((0, 1), (0, 1)), mode="edge").astype(np.int32)


def predict_bilinear(L: np.ndarray) -> np.ndarray:
    low = np.asarray(L, dtype=np.uint8)
    h, w = low.shape
    p = _padded(low)
    a, b = p[:-1, :-1], p[:-1, 1:]
    c, e = p[1:, :-1], p[1:, 1:]
    high = np.empty((2 * h, 2 * w), dtype=np.uint8)
    high[0::2, 0::2] = low
    high[0::2, 1::2] = ((a + b + 1) >> 1).astype(np.uint8)
    high[1::2, 0::2] = ((a + c + 1) >> 1).astype(np.uint8)
    high[1::2, 1::2] = ((a + b + c + e + 2) >> 2).astype(np.uint8)
    return high


def predict_edge_directed(L: np.ndarray) -> np.ndarray:
    low = np.asarray(L, dtype=np.uint8)
    h, w = low.shape
    p = _padded(low)
    a, b = p[:-1, :-1], p[:-1, 1:]
    c, e = p[1:, :-1], p[1:, 1:]
    high = np.empty((2 * h, 2 * w), dtype=np.uint8)
    high[0::2, 0::2] = low
    centers = np.where(np.abs(a - e) <= np.abs(b - c), (a + e + 1) >> 1, (b + c + 1) >> 1).astype(np.uint8)
    high[1::2, 1::2] = centers

    upper = centers.copy()
    if h > 1:
        upper[1:] = centers[:-1]
    horizontal = np.where(
        np.abs(a - b) <= np.abs(upper.astype(np.int32) - centers.astype(np.int32)),
        (a + b + 1) >> 1,
        (upper.astype(np.int32) + centers.astype(np.int32) + 1) >> 1,
    )
    high[0::2, 1::2] = horizontal.astype(np.uint8)

    left = centers.copy()
    if w > 1:
        left[:, 1:] = centers[:, :-1]
    vertical = np.where(
        np.abs(a - c) <= np.abs(left.astype(np.int32) - centers.astype(np.int32)),
        (a + c + 1) >> 1,
        (left.astype(np.int32) + centers.astype(np.int32) + 1) >> 1,
    )
    high[1::2, 0::2] = vertical.astype(np.uint8)
    return high


def predict_med(L: np.ndarray) -> np.ndarray:
    """Median edge detector predictor used as a lightweight paper baseline."""
    low = np.asarray(L, dtype=np.uint8)
    h, w = low.shape
    p = np.pad(low, ((1, 1), (1, 1)), mode="edge").astype(np.int32)
    # For each LR location: a=upper-left, b=upper, c=left.
    a = p[:-2, :-2]
    b = p[:-2, 1:-1]
    c = p[1:-1, :-2]
    prediction = np.where(c <= np.minimum(a, b), np.maximum(a, b), np.where(c >= np.maximum(a, b), np.minimum(a, b), a + b - c))
    high = np.empty((2 * h, 2 * w), dtype=np.uint8)
    high[0::2, 0::2] = low
    high[0::2, 1::2] = ((low.astype(np.int32) + np.pad(low[:, 1:], ((0, 0), (0, 1)), mode="edge") + 1) >> 1).astype(np.uint8)
    high[1::2, 0::2] = ((low.astype(np.int32) + np.pad(low[1:, :], ((0, 1), (0, 0)), mode="edge") + 1) >> 1).astype(np.uint8)
    high[1::2, 1::2] = np.clip(prediction, 0, 255).astype(np.uint8)
    return high


def predict_cv2_bicubic(L: np.ndarray) -> np.ndarray:
    low = np.asarray(L, dtype=np.uint8)
    high = cv2.resize(low, (2 * low.shape[1], 2 * low.shape[0]), interpolation=cv2.INTER_CUBIC)
    high[0::2, 0::2] = low
    return high.astype(np.uint8)


def predict(L: np.ndarray, method: str = "edge_directed") -> np.ndarray:
    if method == "bilinear":
        return predict_bilinear(L)
    if method == "edge_directed":
        return predict_edge_directed(L)
    if method == "med":
        return predict_med(L)
    if method == "cv2_bicubic":
        return predict_cv2_bicubic(L)
    if method in {"rcan_style", "rcan_lite"}:
        return predict_rcan_style(L)
    raise ValueError(f"unknown predictor: {method}")


def residual(masked: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    value = np.asarray(masked, dtype=np.int32) - np.asarray(prediction, dtype=np.int32)
    if np.any(value < -255) or np.any(value > 255):
        raise ValueError("residual is outside the supported range")
    if np.any(value[0::2, 0::2] != 0):
        raise AssertionError("block-first residuals must be zero")
    return value


def residual_sequence(E: np.ndarray) -> np.ndarray:
    m, n = E.shape
    blocks = np.asarray(E).reshape(m // 2, 2, n // 2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
    return blocks[:, 1:].reshape(-1).astype(np.int32)


def rebuild_residual(sequence: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m, n = shape
    seq = np.asarray(sequence, dtype=np.int32).reshape(-1)
    expected = 3 * m * n // 4
    if len(seq) != expected:
        raise ValueError(f"expected {expected} residual symbols, received {len(seq)}")
    blocks = np.zeros((m // 2, n // 2, 4), dtype=np.int32)
    blocks[:, :, 1:] = seq.reshape(m // 2, n // 2, 3)
    return blocks.reshape(m // 2, n // 2, 2, 2).transpose(0, 2, 1, 3).reshape(m, n)
