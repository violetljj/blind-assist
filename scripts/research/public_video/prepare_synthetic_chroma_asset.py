#!/usr/bin/env python3
"""Convert a reviewed chroma-key image into a deterministic hard-alpha asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(image: np.ndarray, *, chroma_margin: int, crop_margin: int) -> tuple[np.ndarray, dict[str, int]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("input must be BGR RGB image")
    blue, green, red = [image[..., index].astype(np.int16) for index in range(3)]
    background = (np.minimum(red, blue) - green) >= chroma_margin
    foreground = (~background).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    if count < 2:
        raise ValueError("no foreground component found")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == component
    area = int(mask.sum())
    if area < image.shape[0] * image.shape[1] * 0.05:
        raise ValueError("largest foreground component is implausibly small")
    ys, xs = np.flatnonzero(mask) // mask.shape[1], np.flatnonzero(mask) % mask.shape[1]
    x1, x2 = max(0, int(xs.min()) - crop_margin), min(image.shape[1], int(xs.max()) + crop_margin + 1)
    y1, y2 = max(0, int(ys.min()) - crop_margin), min(image.shape[0], int(ys.max()) + crop_margin + 1)
    crop = image[y1:y2, x1:x2].copy()
    alpha = (mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    crop[alpha == 0] = 0
    rgba = np.dstack([crop, alpha])
    return rgba, {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "foreground_pixel_count": area}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--chroma-margin", type=int, default=100)
    parser.add_argument("--crop-margin", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise ValueError("refusing to overwrite prepared asset or receipt")
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input)
    rgba, bounds = prepare(image, chroma_margin=args.chroma_margin, crop_margin=args.crop_margin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rgba):
        raise RuntimeError("failed to write RGBA asset")
    receipt = {
        "schema": "blindassist_synthetic_chroma_asset_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(args.input), "source_sha256": sha256_file(args.input),
        "output_path": str(args.output.resolve()), "output_sha256": sha256_file(args.output),
        "algorithm": "largest non-magenta connected component with deterministic binary alpha",
        "chroma_margin": args.chroma_margin, "crop_margin": args.crop_margin,
        "output_shape": list(rgba.shape), "alpha_values": sorted(map(int, np.unique(rgba[..., 3]))),
        "bounds_in_source": bounds,
        "authorization": {"train_only_synthetic_asset": True, "real_event_truth": False,
                          "provider_evaluation_credit": False, "production": False},
    }
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.receipt) + ".sha256").write_text(sha256_file(args.receipt) + "\n", encoding="ascii")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
