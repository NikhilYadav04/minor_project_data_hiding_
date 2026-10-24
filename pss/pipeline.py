from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import N, R, validate_image_shape
from .embed import embed_core, extract_core, expand_share, extract_shared
from .auth import find_valid_shares, require_valid_share, stamp_share
from .deep_models import model_metadata
from .exceptions import InsufficientSharesError, InvalidKeyError
from .huffman import compress_residuals, decompress_residuals, pack_bits, pad_bits
from .keys import KeyBundle
from .msa import prepare_msa, restore_msa, select_adaptive_rect
from .predictor import get_lr, predict, rebuild_residual, residual, residual_sequence
from .sharing import recover_pairs, share_pairs


@dataclass
class OwnerOutput:
    final_shares: list[np.ndarray]
    expanded_shares: list[np.ndarray]
    masked: np.ndarray
    stats: dict


PREDICTOR_CANDIDATES = ("bilinear", "edge_directed", "med")


def _select_predictor(masked: np.ndarray, low: np.ndarray) -> tuple[str, list[dict[str, float | str]]]:
    """Choose the predictor with the smallest compressed residual stream."""
    trials: list[dict[str, float | str]] = []
    for order, candidate in enumerate(PREDICTOR_CANDIDATES):
        prediction = predict(low, candidate)
        errors = residual(masked, prediction)
        sequence = residual_sequence(errors)
        compressed = compress_residuals(sequence)
        trials.append({
            "method": candidate,
            "bit_length": int(compressed.bit_length),
            "mean_abs_residual": float(np.mean(np.abs(sequence))),
            "order": order,
        })
    best = min(trials, key=lambda row: (int(row["bit_length"]), float(row["mean_abs_residual"]), int(row["order"])))
    return str(best["method"]), trials


def _check_share_shapes(shares) -> tuple[int, int]:
    if not shares:
        raise InsufficientSharesError("no shares were supplied")
    arrays = [np.asarray(share, dtype=np.uint8) for share in shares]
    shape = arrays[0].shape
    validate_image_shape(shape)
    if any(array.shape != shape for array in arrays):
        raise ValueError("all shares must have the same shape")
    return shape


def owner_encode(
    img: np.ndarray,
    rect: tuple[int, int, int, int] | None,
    keys: KeyBundle,
    method: str = "edge_directed",
    r: int = R,
    n: int = N,
    msa_mode: str = "manual",
    inpainting_method: str = "telea",
) -> OwnerOutput:
    image = np.asarray(img, dtype=np.uint8)
    validate_image_shape(image.shape)
    if (r, n) != (R, N):
        raise ValueError("this implementation is configured for (r, n) = (3, 4)")
    timings = {}
    if msa_mode == "manual":
        selected_rect = rect
    elif msa_mode in {"auto_texture", "auto_edge"}:
        if rect is None:
            dimensions = (60, 40)
        else:
            dimensions = (int(rect[2]), int(rect[3]))
        selected_rect = select_adaptive_rect(image, msa_mode.removeprefix("auto_"), *dimensions)
    else:
        raise ValueError("msa_mode must be 'manual', 'auto_texture', or 'auto_edge'")
    start = time.perf_counter()
    msa = prepare_msa(image, selected_rect, keys.stream_seed("K_sc"), inpainting_method)
    timings["t_inpaint"] = time.perf_counter() - start

    requested_method = method
    start = time.perf_counter()
    low = get_lr(msa.masked)
    predictor_candidates: list[dict[str, float | str]] = []
    if method == "auto":
        method, predictor_candidates = _select_predictor(msa.masked, low)
    high = predict(low, method)
    errors = residual(msa.masked, high)
    sequence = residual_sequence(errors)
    timings["t_predict"] = time.perf_counter() - start

    start = time.perf_counter()
    compressed = compress_residuals(sequence)
    C_bits = pad_bits(compressed.bits, 16)
    C_bytes = pack_bits(C_bits)
    timings["t_compress"] = time.perf_counter() - start

    start = time.perf_counter()
    lr_pairs = low.reshape(-1, 2)
    sc_pairs = C_bytes.reshape(-1, 2)
    sa_pairs = msa.pairs
    shared_lr = share_pairs(lr_pairs, keys.lr_keys(len(lr_pairs)), r, n, keys.rng_coeff("L"))
    # The extension makes the code and MSA streams recoverable from any
    # threshold subset, so one damaged or missing share does not block
    # masked/original recovery.
    shared_sc = share_pairs(sc_pairs, keys.code_keys(len(sc_pairs)), r, n, keys.rng_coeff("C"))
    shared_sa = share_pairs(sa_pairs, keys.msa_keys(len(sa_pairs)), r, n, keys.rng_coeff("A")) if len(sa_pairs) else np.empty((n, 0), dtype=np.uint8)
    timings["t_share"] = time.perf_counter() - start

    start = time.perf_counter()
    expanded_shares = []
    final_shares = []
    low_shape = (image.shape[0] // 2, image.shape[1] // 4)
    for share_index in range(n):
        low_share = shared_lr[share_index].reshape(low_shape)
        expanded = expand_share(low_share, keys.rng_fill(share_index))
        final = embed_core(expanded, shared_sc[share_index], shared_sa[share_index], compressed.bit_length)
        final = stamp_share(final, share_index, keys.auth_seed(share_index))
        expanded_shares.append(expanded)
        final_shares.append(final)
    timings["t_embed"] = time.perf_counter() - start

    total_pixels = image.size
    capacity = 7 * total_pixels // 8
    len_sc = len(shared_sc[0])
    len_sa = len(shared_sa[0])
    free_bytes = capacity - 16 - len_sc - len_sa
    timings.update(
        {
            "LEN_SC": len_sc,
            "LEN_SA": len_sa,
            "BITLEN_C": compressed.bit_length,
            "bpp_paper": (7 * total_pixels - 8 * len_sc - 8 * len_sa) / total_pixels,
            "bpp_free": max(0, 8 * (free_bytes - 8) / total_pixels),
            "free_bytes": free_bytes,
            "method": method,
            "requested_method": requested_method,
            "predictor_selection": "automatic" if requested_method == "auto" else "manual",
            "predictor_candidates": predictor_candidates,
            "seed": keys.master_seed,
            "msa_mode": msa_mode,
            "inpainting_method": inpainting_method,
            "predictor_backend": "integer" if method in {"bilinear", "edge_directed", "med"} else "rcan_style_fixed_point" if method in {"rcan_style", "rcan_lite"} else "adaptive_integer",
            "model_metadata": model_metadata() if method in {"rcan_style", "rcan_lite"} else None,
            "msa_top": -1 if selected_rect is None else selected_rect[0],
            "msa_left": -1 if selected_rect is None else selected_rect[1],
            "msa_height": 0 if selected_rect is None else selected_rect[2],
            "msa_width": 0 if selected_rect is None else selected_rect[3],
        }
    )
    return OwnerOutput(final_shares, expanded_shares, msa.masked, timings)


def recover_stage1(shares, idx, keys: KeyBundle, method: str = "edge_directed") -> np.ndarray:
    if len(shares) < R:
        raise InsufficientSharesError("Stage 1 requires three shares")
    selected = list(shares) if len(shares) == len(idx) else [shares[i] for i in idx]
    shape = _check_share_shapes(selected)
    for share, share_index in zip(selected, idx):
        require_valid_share(share, share_index, keys.auth_seed(share_index))
    arrays = [extract_shared(np.asarray(share, dtype=np.uint8)) for share in selected]
    pairs = recover_pairs(np.stack([array.reshape(-1) for array in arrays]), idx, keys.lr_keys(arrays[0].size), R)
    return pairs.reshape(-1).reshape(shape[0] // 2, shape[1] // 2)


def _recover_code(shares, indexes, keys: KeyBundle) -> tuple[np.ndarray, dict]:
    shape = _check_share_shapes(shares)
    if len(shares) != R or len(indexes) != R:
        raise InsufficientSharesError("code recovery requires three valid shares")
    for share, index in zip(shares, indexes):
        require_valid_share(share, index, keys.auth_seed(index))
    core = [extract_core(share) for share in shares]
    infos = [entry[2] for entry in core]
    if any(info["LEN_SC"] != infos[0]["LEN_SC"] or info["BITLEN_C"] != infos[0]["BITLEN_C"] for info in infos):
        raise InvalidKeyError("shares do not agree on compressed-code metadata")
    sc = np.stack([entry[0] for entry in core])
    pairs = recover_pairs(sc, list(indexes), keys.code_keys(len(sc[0])), R)
    return pairs.reshape(-1), infos[0]


def _recover_stage2_selected(shares, indexes, keys: KeyBundle, method: str = "edge_directed") -> np.ndarray:
    shape = _check_share_shapes(shares)
    low = recover_stage1(shares, list(indexes), keys, method)
    code_bytes, info = _recover_code(shares, list(indexes), keys)
    bits = np.unpackbits(code_bytes, bitorder="big")[: info["BITLEN_C"]]
    sequence = decompress_residuals(bits, 3 * shape[0] * shape[1] // 4)
    prediction = predict(low, method)
    errors = rebuild_residual(sequence, shape)
    masked = prediction.astype(np.int32) + errors
    if np.any(masked < 0) or np.any(masked > 255):
        raise InvalidKeyError("recovered residuals produce invalid pixels")
    return masked.astype(np.uint8)


def recover_stage2(shares, keys: KeyBundle, method: str = "edge_directed") -> np.ndarray:
    if len(shares) != N:
        raise InsufficientSharesError("Stage 2 requires all four shares")
    for index, share in enumerate(shares):
        require_valid_share(share, index, keys.auth_seed(index))
    return _recover_stage2_selected(shares[:R], list(range(R)), keys, method)


def recover_stage3(shares, keys: KeyBundle, method: str = "edge_directed") -> np.ndarray:
    if len(shares) != N:
        raise InsufficientSharesError("Stage 3 requires all four shares")
    for index, share in enumerate(shares):
        require_valid_share(share, index, keys.auth_seed(index))
    return _recover_stage3_selected(shares[:R], list(range(R)), keys, method)


def _recover_stage3_selected(shares, indexes, keys: KeyBundle, method: str = "edge_directed") -> np.ndarray:
    masked = _recover_stage2_selected(shares, indexes, keys, method)
    core = [extract_core(share) for share in shares]
    len_sa = core[0][2]["LEN_SA"]
    if len_sa == 0:
        return masked
    if any(entry[2]["LEN_SA"] != len_sa for entry in core):
        raise InvalidKeyError("shares do not agree on MSA metadata")
    sa = np.stack([entry[1] for entry in core])
    pairs = recover_pairs(sa, list(indexes), keys.msa_keys(len(sa[0])), R)
    return restore_msa(masked, pairs, keys.stream_seed("K_sc"))


def recover_available(shares, keys: KeyBundle, method: str = "edge_directed", stage: int = 3, indexes=None) -> tuple[np.ndarray, list[int]]:
    """Recover using the first valid threshold subset and report usable shares."""
    arrays = list(shares)
    actual_indexes = list(range(len(arrays))) if indexes is None else [int(index) for index in indexes]
    valid_indexes = find_valid_shares(arrays, keys, actual_indexes)
    if len(valid_indexes) < R:
        raise InsufficientSharesError(f"need {R} valid shares; found {len(valid_indexes)}")
    selected_indexes = valid_indexes[:R]
    selected = [arrays[actual_indexes.index(index)] for index in selected_indexes]
    if stage == 1:
        image = recover_stage1(selected, selected_indexes, keys, method)
    elif stage == 2:
        image = _recover_stage2_selected(selected, selected_indexes, keys, method)
    elif stage == 3:
        image = _recover_stage3_selected(selected, selected_indexes, keys, method)
    else:
        raise ValueError("stage must be 1, 2, or 3")
    return image, valid_indexes
