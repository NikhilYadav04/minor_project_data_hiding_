"""Deterministic inpainting backends used by the PSS research pipeline.

The original paper uses a pretrained GICA model.  This project does not ship
those weights, so the contextual backend below is deliberately described as
GICA-inspired rather than exact GICA.  It combines multi-scale context,
Navier--Stokes diffusion, and boundary-preserving compositing while keeping
known pixels byte-identical.
"""

from __future__ import annotations

import cv2
import numpy as np


def _validate(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(image, dtype=np.uint8)
    binary = np.asarray(mask, dtype=np.uint8)
    if values.ndim != 2 or binary.shape != values.shape:
        raise ValueError("image and mask must be equal-sized grayscale arrays")
    return values, np.where(binary > 0, 255, 0).astype(np.uint8)


def telea_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values, binary = _validate(image, mask)
    if not np.any(binary):
        return values.copy()
    result = cv2.inpaint(values, binary, 3, cv2.INPAINT_TELEA)
    result[binary == 0] = values[binary == 0]
    return result.astype(np.uint8)


def contextual_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Use multi-scale contextual evidence to fill a masked region.

    This is a deterministic, dependency-light approximation of the paper's
    coarse-to-fine/contextual-attention idea.  It is useful as a stronger
    reproducible fallback when GICA weights are unavailable:

    1. fill at 1/4 and 1/2 scale to propagate broad context,
    2. refine at full resolution using Navier--Stokes diffusion,
    3. preserve every known pixel exactly.
    """

    values, binary = _validate(image, mask)
    if not np.any(binary):
        return values.copy()

    current = values.copy()
    # Coarse-to-fine context propagation.  Nearest-neighbour mask resizing
    # keeps the hole conservative at each scale.
    for divisor in (4, 2):
        height = max(2, values.shape[0] // divisor)
        width = max(2, values.shape[1] // divisor)
        small = cv2.resize(current, (width, height), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        filled = cv2.inpaint(small, small_mask, 3, cv2.INPAINT_NS)
        enlarged = cv2.resize(filled, (values.shape[1], values.shape[0]), interpolation=cv2.INTER_CUBIC)
        current[binary > 0] = enlarged[binary > 0]

    refined = cv2.inpaint(current, binary, 5, cv2.INPAINT_NS)
    # A very light contextual blend suppresses hard seams while avoiding a
    # second alteration of the known image area.
    blurred = cv2.GaussianBlur(refined, (5, 5), 0)
    result = refined.astype(np.float32)
    result[binary > 0] = 0.75 * refined[binary > 0] + 0.25 * blurred[binary > 0]
    result[binary == 0] = values[binary == 0]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def inpaint(image: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
    if method == "telea":
        return telea_inpaint(image, mask)
    if method in {"contextual", "gica_inspired"}:
        return contextual_inpaint(image, mask)
    raise ValueError("inpainting method must be 'telea' or 'contextual'")
