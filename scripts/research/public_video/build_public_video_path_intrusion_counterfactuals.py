#!/usr/bin/env python3
"""Build pixel-auditable train-only path-intrusion counterfactual pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


SCHEMA = "blindassist_public_video_path_intrusion_counterfactuals_v1"
PLACEMENTS = {
    "jakarta_t025_original.png": [(245, 264, 50), (295, 266, 55), (350, 266, 55), (405, 264, 50)],
    "jakarta_t050_original.png": [(225, 274, 48), (282, 277, 55), (345, 277, 55), (405, 274, 48)],
    "jakarta_t067_original.png": [(250, 278, 48), (305, 281, 55), (365, 281, 55), (425, 278, 48)],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_asset(asset: Image.Image) -> Image.Image:
    rgba = asset.convert("RGBA")
    alpha = rgba.getchannel("A")
    box = alpha.getbbox()
    if box is None:
        raise ValueError("cone asset has no opaque pixels")
    return rgba.crop(box)


def compose(parent: Image.Image, asset: Image.Image, placements: list[tuple[int, int, int]]) -> tuple[Image.Image, Image.Image]:
    base = parent.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    intervention = Image.new("L", base.size, 0)
    for center_x, base_y, height in placements:
        width = max(1, round(height * asset.width / asset.height))
        cone = asset.resize((width, height), Image.Resampling.LANCZOS)
        x = round(center_x - width / 2)
        y = base_y - height
        shadow = Image.new("L", base.size, 0)
        draw = ImageDraw.Draw(shadow)
        draw.ellipse((x - width // 5, base_y - max(1, height // 20), x + width + width // 5, base_y + max(2, height // 12)), fill=70)
        shadow = shadow.filter(ImageFilter.GaussianBlur(max(1.0, height / 28)))
        shadow_rgba = Image.new("RGBA", base.size, (0, 0, 0, 0))
        shadow_rgba.putalpha(shadow)
        layer = Image.alpha_composite(layer, shadow_rgba)
        layer.alpha_composite(cone, (x, y))
        intervention = Image.fromarray(np.maximum(np.asarray(intervention), np.asarray(shadow)), mode="L")
        cone_mask = Image.new("L", base.size, 0)
        cone_mask.paste(cone.getchannel("A"), (x, y))
        intervention = Image.fromarray(np.maximum(np.asarray(intervention), np.asarray(cone_mask)), mode="L")
    return Image.alpha_composite(base, layer).convert("RGB"), intervention


def unchanged_outside_mask(parent: Image.Image, edited: Image.Image, mask: Image.Image) -> tuple[bool, int]:
    before = np.asarray(parent.convert("RGB"), dtype=np.int16)
    after = np.asarray(edited.convert("RGB"), dtype=np.int16)
    changed = np.any(before != after, axis=2)
    permitted = np.asarray(mask) > 0
    outside = int(np.count_nonzero(changed & ~permitted))
    return outside == 0, outside


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if "secondary-corridor-causal" in str(output).replace("\\", "/").lower():
        raise ValueError("independent direction path is forbidden")
    report_path = output / "generation_report.json"
    if report_path.exists() or Path(str(report_path) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {report_path}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if not spec["isolation"]["train_only"] or spec["isolation"]["real_evaluation_credit"]:
        raise ValueError("counterfactual dataset must be train-only with zero real evaluation credit")
    asset = normalized_asset(Image.open(args.asset))
    candidates = output / "candidates"
    masks = output / "masks"
    candidates.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    pairs = []
    for parent_name, placements in PLACEMENTS.items():
        parent_path = output / "parents" / parent_name
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        parent = Image.open(parent_path).convert("RGB")
        if parent.size != (640, 360):
            raise ValueError(f"unexpected parent size: {parent_path} {parent.size}")
        edited, mask = compose(parent, asset, placements)
        invariant, outside = unchanged_outside_mask(parent, edited, mask)
        if not invariant:
            raise ValueError(f"pixels changed outside intervention mask: {outside}")
        stem = parent_path.stem.replace("_original", "")
        edited_path = candidates / f"{stem}_path_intrusion.png"
        mask_path = masks / f"{stem}_intervention_mask.png"
        edited.save(edited_path)
        mask.save(mask_path)
        changed = int(np.count_nonzero(np.any(np.asarray(parent) != np.asarray(edited), axis=2)))
        pairs.append({
            "pair_id": stem,
            "split": "train",
            "parent_source_id": spec["parent_source"]["source_id"],
            "negative": {"path": str(parent_path.resolve()), "sha256": sha256_file(parent_path), "label": "outside_or_nonblocking"},
            "positive": {"path": str(edited_path.resolve()), "sha256": sha256_file(edited_path), "label": "enters_or_blocks_future_path"},
            "intervention_mask": {"path": str(mask_path.resolve()), "sha256": sha256_file(mask_path)},
            "placement_centers_and_heights": placements,
            "changed_pixel_count": changed,
            "outside_mask_changed_pixel_count": outside,
            "pixel_invariant_passed": invariant,
            "human_truth_claimed": False,
            "real_evaluation_credit": False,
        })
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_spec": {"path": str(args.spec.resolve()), "sha256": sha256_file(args.spec)},
        "generated_asset": {"path": str(args.asset.resolve()), "sha256": sha256_file(args.asset)},
        "generation_method": "image-generated isolated cone plus deterministic alpha compositing",
        "pairs": pairs,
        "summary": {"pair_count": len(pairs), "all_pixel_invariants_passed": all(row["pixel_invariant_passed"] for row in pairs)},
        "authorizations": {"representation_diagnostic": True, "training": False, "calibration": False, "blind": False, "android_runtime_change": False, "production_model_replacement": False},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(report_path) + ".sha256").write_text(sha256_file(report_path) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    value = run(parse_args())
    print(json.dumps({"ok": True, **value["summary"]}, ensure_ascii=False))
