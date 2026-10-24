"""Deterministic deep-model-inspired components for paper-aligned ablations.

The paper's exact GICA/RCAN weights are not distributed with this repository.
These implementations therefore provide reproducible, fixed-point analogues
for comparison, while the report continues to distinguish them from the
original neural networks.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np


RCAN_STYLE_CONFIG = {
    "name": "rcan_style_fixed_point",
    "residual_groups": 5,
    "blocks_per_group": 5,
    "channels": 1,
    "scale": 2,
    "loss": "not trained; deterministic fixed-point analogue",
}


def model_metadata() -> dict[str, object]:
    config = repr(sorted(RCAN_STYLE_CONFIG.items())).encode("utf-8")
    return {**RCAN_STYLE_CONFIG, "artifact_sha256": hashlib.sha256(config).hexdigest()}


def predict_rcan_style(low: np.ndarray) -> np.ndarray:
    """RCAN-inspired fixed-point 2x predictor.

    The routine mirrors the useful structural ideas of RCAN—long skip,
    repeated residual refinement, and a content-dependent channel gate—using
    deterministic OpenCV filters.  It is not a trained RCAN network.
    """

    source = np.asarray(low, dtype=np.uint8)
    if source.ndim != 2:
        raise ValueError("low-resolution input must be a grayscale 2-D array")
    high = cv2.resize(source, (source.shape[1] * 2, source.shape[0] * 2), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    long_skip = high.copy()
    for _ in range(RCAN_STYLE_CONFIG["residual_groups"]):
        group_skip = high.copy()
        for _ in range(RCAN_STYLE_CONFIG["blocks_per_group"]):
            smooth = cv2.GaussianBlur(high, (3, 3), 0)
            detail = high - smooth
            local_energy = cv2.GaussianBlur(np.abs(detail), (5, 5), 0)
            gate = np.clip(0.45 + local_energy / 64.0, 0.45, 1.0)
            high = high + 0.08 * detail * gate
            high = np.clip(high, 0, 255)
        high = 0.75 * high + 0.25 * group_skip
    high = 0.8 * high + 0.2 * long_skip
    result = np.clip(np.rint(high), 0, 255).astype(np.uint8)
    result[0::2, 0::2] = source
    return result
