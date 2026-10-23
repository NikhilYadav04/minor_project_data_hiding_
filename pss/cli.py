from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .keys import KeyBundle
from .pipeline import owner_encode, recover_stage1, recover_stage2, recover_stage3


def load_gray512(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)


def demo(args: argparse.Namespace) -> None:
    image = load_gray512(args.image)
    rect = tuple(args.rect) if args.rect else None
    keys = KeyBundle(args.seed)
    output = owner_encode(image, rect, keys, args.method)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "original.png"), image)
    cv2.imwrite(str(output_dir / "masked.png"), output.masked)
    for index, (expanded, final) in enumerate(zip(output.expanded_shares, output.final_shares), 1):
        cv2.imwrite(str(output_dir / f"share_{index}_expanded.png"), expanded)
        cv2.imwrite(str(output_dir / f"share_{index}_final.png"), final)
    lr = recover_stage1(output.final_shares, [0, 1, 2], keys, args.method)
    stage2 = recover_stage2(output.final_shares, keys, args.method)
    stage3 = recover_stage3(output.final_shares, keys, args.method)
    cv2.imwrite(str(output_dir / "lr.png"), lr)
    cv2.imwrite(str(output_dir / "recovered_stage2.png"), stage2)
    cv2.imwrite(str(output_dir / "recovered_stage3.png"), stage3)
    print("Stage 1 shape:", lr.shape)
    print("Stage 2 exact:", np.array_equal(stage2, output.masked))
    print("Stage 3 exact:", np.array_equal(stage3, image))
    print("bpp_paper:", output.stats["bpp_paper"])
    print("bpp_free:", output.stats["bpp_free"])


def main() -> None:
    parser = argparse.ArgumentParser(description="PSS simplified reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--image", required=True)
    demo_parser.add_argument("--rect", nargs=4, type=int)
    demo_parser.add_argument("--seed", type=int, default=123)
    demo_parser.add_argument("--method", choices=["bilinear", "edge_directed", "med"], default="edge_directed")
    demo_parser.add_argument("--out", default="out")
    demo_parser.set_defaults(func=demo)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
