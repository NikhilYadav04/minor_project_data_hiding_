"""Reproducible Tier-1 experiments for the simplified implementation."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pss.cli import load_gray512
from pss.auth import verify_share
from pss.embed import hider_embed, hider_extract
from pss.keys import KeyBundle
from pss.metrics import adjacent_correlation, entropy, npcr, psnr, uaci
from pss.pipeline import owner_encode, recover_available, recover_stage1, recover_stage2, recover_stage3


IMAGE_NAMES = ["airplane", "baboon", "boat", "couple", "elaine", "man", "peppers", "sailboat"]
METHODS = ["bilinear", "edge_directed", "med"]
QUOTED_BPP = {
    "airplane": 5.4867,
    "baboon": 5.1466,
    "boat": 5.0487,
    "couple": 5.1721,
    "elaine": 5.1708,
    "man": 5.4330,
    "peppers": 5.0868,
    "sailboat": 5.2857,
}


def synthetic(name: str, shape=(128, 128)) -> np.ndarray:
    y, x = np.indices(shape)
    salt = sum(ord(c) for c in name)
    return ((x * (7 + salt % 11) + y * (13 + salt % 17) + x * y % 31) % 256).astype(np.uint8)


def load_images() -> dict[str, np.ndarray]:
    result = {}
    for name in IMAGE_NAMES:
        path = ROOT / "data" / "images" / f"{name}.png"
        result[name] = load_gray512(path) if path.exists() else synthetic(name)
    return result


def image_source(name: str) -> str:
    return "bundled_public_substitute" if (ROOT / "data" / "images" / f"{name}.png").exists() else "deterministic_synthetic_fallback"


def save_csv(rows: list[dict], name: str) -> None:
    pd.DataFrame(rows).to_csv(ROOT / "results" / f"{name}.csv", index=False)


def e1(images):
    rows = []
    for name, image in images.items():
        rect = (image.shape[0] // 4, image.shape[1] // 4, image.shape[0] // 8, image.shape[1] // 8)
        for method in METHODS:
            keys = KeyBundle(2026)
            output = owner_encode(image, rect, keys, method)
            stage2 = recover_stage2(output.final_shares, keys, method)
            stage3 = recover_stage3(output.final_shares, keys, method)
            for idx in itertools.combinations(range(4), 3):
                recovered = recover_stage1([output.final_shares[i] for i in idx], idx, keys, method)
                rows.append({"image": name, "method": method, "shares": "".join(str(i + 1) for i in idx), "stage1_pass": bool(np.array_equal(recovered, output.masked[0::2, 0::2])), "stage2_pass": bool(np.array_equal(stage2, output.masked)), "stage3_pass": bool(np.array_equal(stage3, image))})
    save_csv(rows, "E1")


def e2(images):
    rows = []
    for name, image in images.items():
        rect = None
        for method in METHODS:
            output = owner_encode(image, rect, KeyBundle(2026), method)
            rows.append({"image": name, "source": image_source(name), "method": method, "bpp_paper": output.stats["bpp_paper"], "bpp_free": output.stats["bpp_free"], "quoted_paper_bpp": QUOTED_BPP[name]})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "E2.csv", index=False)
    pivot = frame.pivot(index="image", columns="method", values="bpp_free")
    pivot.plot(kind="bar", title="Measured practical free-room embedding rate")
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "E2_bpp.png")
    plt.close()


def e3(images):
    rows = []
    for name, image in images.items():
        output = owner_encode(image, None, KeyBundle(2026), "edge_directed")
        for index, share in enumerate(output.final_shares, 1):
            rows.append({"image": name, "share": index, "entropy": entropy(share), "horizontal_correlation": adjacent_correlation(share, 1), "vertical_correlation": adjacent_correlation(share, 0), "npcr_vs_original": npcr(image, share), "uaci_vs_original": uaci(image, share)})
    save_csv(rows, "E3")


def e4(images):
    name, image = next(iter(images.items()))
    output = owner_encode(image, (image.shape[0] // 4, image.shape[1] // 4, image.shape[0] // 8, image.shape[1] // 8), KeyBundle(2026), "edge_directed")
    panels = [image, output.masked, output.final_shares[0], output.final_shares[1], output.final_shares[2], output.final_shares[3], recover_stage3(output.final_shares, KeyBundle(2026), "edge_directed")]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3))
    for axis, panel, title in zip(axes, panels, ["original", "masked", "share 1", "share 2", "share 3", "share 4", "stage 3"]):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "E4_overview.png", dpi=150)
    plt.close()


def e5(images):
    rows = []
    for name, image in images.items():
        keys = KeyBundle(2026)
        output = owner_encode(image, None, keys)
        wrong = KeyBundle(2027)
        try:
            recover_stage2(output.final_shares, wrong)
            wrong_key = False
        except Exception:
            wrong_key = True
        recovered = recover_stage3(output.final_shares, keys)
        rows.append({"image": name, "wrong_key_rejected": wrong_key, "stage3_exact": bool(np.array_equal(recovered, image)), "stage3_psnr": psnr(image, recovered)})
    save_csv(rows, "E5")


def e6(images):
    rows = []
    for name, image in images.items():
        for method in METHODS:
            output = owner_encode(image, None, KeyBundle(2026), method)
            rows.append({"image": name, "method": method, **{key: value for key, value in output.stats.items() if key.startswith("t_")}})
    save_csv(rows, "E6")


def e7(images):
    name, image = next(iter(images.items()))
    rows = []
    for size in (16, 24, 32, 48, 64, 80):
        rect = (8, 8, size, size)
        try:
            output = owner_encode(image, rect, KeyBundle(2026), "edge_directed")
            rows.append({"image": name, "area": size * size, "height": size, "width": size, "fits": True, "bpp_free": output.stats["bpp_free"]})
        except Exception as exc:
            rows.append({"image": name, "area": size * size, "height": size, "width": size, "fits": False, "bpp_free": np.nan, "error": type(exc).__name__})
    save_csv(rows, "E7")
    frame = pd.DataFrame(rows)
    frame[frame["fits"]].plot(x="area", y="bpp_free", marker="o", title="MSA size sweep")
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "E7_msa_sweep.png")
    plt.close()


def e8(images):
    rows = []
    for name, image in images.items():
        low = image[0::2, 0::2]
        from pss.predictor import predict, residual, residual_sequence
        from pss.huffman import compress_residuals
        for method in ("bilinear", "edge_directed", "med", "cv2_bicubic"):
            errors = residual(image, predict(low, method))
            sequence = residual_sequence(errors)
            paper = compress_residuals(sequence, canonical=False)
            canonical = compress_residuals(sequence, canonical=True)
            rows.append({"image": name, "method": method, "residual_mean_abs": float(np.mean(np.abs(sequence))), "paper_bits": paper.bit_length, "canonical_bits": canonical.bit_length, "paper_bpp": paper.bit_length / image.size, "canonical_bpp": canonical.bit_length / image.size})
    save_csv(rows, "E8")
    frame = pd.DataFrame(rows)
    frame.pivot(index="image", columns="method", values="residual_mean_abs").plot(kind="bar", title="Mean absolute residual by predictor")
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "E8_residuals.png")
    plt.close()
    frame.pivot(index="image", columns="method", values="paper_bpp").plot(kind="bar", title="Paper-faithful Huffman rate by predictor")
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "E8_embedding_rate.png")
    plt.close()


def e9(images):
    rows = []
    # With one fewer than r shares and a known evaluation point, the second
    # polynomial coefficient leaves 256 candidate (c, a1) pairs per sample.
    for name in images:
        rows.append({"image": name, "candidate_pairs_with_true_key": 256, "candidate_pairs_total": 65536, "without_key_observation": "not enumerable from a single sample"})
    save_csv(rows, "E9")


def e10(images):
    rows = []
    for name, image in images.items():
        output = owner_encode(image, None, KeyBundle(2026), "edge_directed")
        for index, share in enumerate(output.final_shares):
            capacity = output.stats["free_bytes"] - 8
            payload = bytes((value + index) % 256 for value in range(max(0, capacity)))
            embedded = hider_embed(share, payload, KeyBundle(2026).dh_seed(index))
            extracted = hider_extract(embedded, KeyBundle(2026).dh_seed(index))
            rows.append({"image": name, "share": index + 1, "capacity_bytes": capacity, "payload_bytes": len(payload), "round_trip": extracted == payload, "capacity_bpp": 8 * capacity / image.size})
    save_csv(rows, "E10")


def e11(images):
    rows = []
    for name, image in images.items():
        keys = KeyBundle(2026)
        output = owner_encode(image, None, keys, "edge_directed")
        clean = [verify_share(share, index, keys.auth_seed(index)) for index, share in enumerate(output.final_shares)]
        tampered = output.final_shares.copy()
        tampered[1] = tampered[1].copy()
        tampered[1][10, 10] ^= 0xFF
        tampered_status = [verify_share(share, index, keys.auth_seed(index)) for index, share in enumerate(tampered)]
        rows.append({"image": name, "clean_valid_count": int(sum(clean)), "tampered_valid_count": int(sum(tampered_status)), "tampered_share_rejected": not tampered_status[1], "valid_shares_remaining": int(sum(tampered_status))})
    save_csv(rows, "E11")


def e12_adaptive_msa(images):
    rows = []
    for name, image in images.items():
        manual = (image.shape[0] // 4, image.shape[1] // 4, image.shape[0] // 8, image.shape[1] // 10)
        for mode, rect in (("manual", manual), ("auto_texture", manual), ("auto_edge", manual)):
            output = owner_encode(image, rect, KeyBundle(2026), "med", msa_mode=mode)
            recovered = recover_stage3(output.final_shares, KeyBundle(2026), "med")
            rows.append({
                "image": name,
                "mode": mode,
                "top": output.stats["msa_top"],
                "left": output.stats["msa_left"],
                "height": output.stats["msa_height"],
                "width": output.stats["msa_width"],
                "area": output.stats["msa_height"] * output.stats["msa_width"],
                "bpp_free": output.stats["bpp_free"],
                "stage3_exact": bool(np.array_equal(recovered, image)),
            })
    save_csv(rows, "E12")


def e13_resilient_recovery(images):
    rows = []
    for name, image in images.items():
        keys = KeyBundle(2026)
        output = owner_encode(image, None, keys, "med")
        tampered = list(output.final_shares)
        tampered[1] = tampered[1].copy()
        tampered[1][10, 10] ^= 0xFF
        for label, shares, indexes in (("one_tampered", tampered, [0, 1, 2, 3]), ("one_missing", [output.final_shares[i] for i in (0, 2, 3)], [0, 2, 3])):
            for stage, target in ((1, output.masked[0::2, 0::2]), (2, output.masked), (3, image)):
                try:
                    recovered, valid = recover_available(shares, keys, "med", stage, indexes=indexes)
                    rows.append({"image": name, "case": label, "stage": stage, "valid_count": len(valid), "used_shares": "".join(str(i + 1) for i in valid[:3]), "exact": bool(np.array_equal(recovered, target)), "rejected": False})
                except Exception as exc:
                    rows.append({"image": name, "case": label, "stage": stage, "valid_count": 0, "used_shares": "", "exact": False, "rejected": True, "error": type(exc).__name__})
    save_csv(rows, "E13")


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    images = load_images()
    (ROOT / "results" / "run_metadata.json").write_text(json.dumps({
        "master_seed": 2026,
        "methods": METHODS + ["cv2_bicubic"],
        "image_names": IMAGE_NAMES,
        "image_sources": {name: image_source(name) for name in IMAGE_NAMES},
        "image_shapes": {name: list(image.shape) for name, image in images.items()},
        "quoted_bpp_source": "SPEC.md / He et al. 2026 Table III",
    }, indent=2) + "\n")
    e1(images)
    e2(images)
    e3(images)
    e4(images)
    e5(images)
    e6(images)
    e7(images)
    e8(images)
    e9(images)
    e10(images)
    e11(images)
    e12_adaptive_msa(images)
    e13_resilient_recovery(images)
    print("Wrote E1-E13 experiment files to results/ and figures/")


if __name__ == "__main__":
    main()
