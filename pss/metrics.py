from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(structural_similarity(np.asarray(reference), np.asarray(candidate), data_range=255))


def mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("reference and candidate must have the same shape")
    return float(np.mean((a - b) ** 2))


def mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("reference and candidate must have the same shape")
    return float(np.mean(np.abs(a - b)))


def quality_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | str]:
    """Return presentation-ready fidelity metrics for equal-sized images."""
    error_mse = mse(reference, candidate)
    value_psnr = psnr(reference, candidate)
    return {
        "psnr": "∞" if math.isinf(value_psnr) else round(float(value_psnr), 4),
        "ssim": round(ssim(reference, candidate), 6),
        "mse": round(error_mse, 6),
        "mae": round(mae(reference, candidate), 6),
    }


def region_quality_metrics(reference: np.ndarray, candidate: np.ndarray, rect: tuple[int, int, int, int] | None) -> dict[str, float | str] | None:
    if rect is None:
        return None
    top, left, height, width = (int(value) for value in rect)
    return quality_metrics(
        np.asarray(reference)[top : top + height, left : left + width],
        np.asarray(candidate)[top : top + height, left : left + width],
    )


def entropy(image: np.ndarray) -> float:
    counts = np.bincount(np.asarray(image, dtype=np.uint8).reshape(-1), minlength=256)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def adjacent_correlation(image: np.ndarray, axis: int = 1, sample: int = 10000, seed: int = 0) -> float:
    values = np.asarray(image, dtype=np.float64)
    if axis == 1:
        left, right = values[:, :-1].reshape(-1), values[:, 1:].reshape(-1)
    elif axis == 0:
        left, right = values[:-1, :].reshape(-1), values[1:, :].reshape(-1)
    else:
        raise ValueError("axis must be 0 or 1")
    if len(left) > sample:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(left), size=sample, replace=False)
        left, right = left[selected], right[selected]
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(abs(np.corrcoef(left, right)[0, 1]))


def npcr(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean(np.asarray(reference) != np.asarray(candidate)) * 100)


def uaci(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(reference, dtype=np.int32) - np.asarray(candidate, dtype=np.int32))) / 255 * 100)
