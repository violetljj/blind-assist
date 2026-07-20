#!/usr/bin/env python3
"""Build equal-count lateral-vs-central marker counterfactual pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import build_public_video_path_intrusion_counterfactuals as base


SCHEMA = "blindassist_public_video_equal_count_path_relation_pairs_v1"
CENTRAL = base.PLACEMENTS
LATERAL = {
    "jakarta_t025_original.png": [(430, 264, 50), (480, 266, 55), (535, 266, 55), (590, 264, 50)],
    "jakarta_t050_original.png": [(420, 274, 48), (477, 277, 55), (540, 277, 55), (600, 274, 48)],
    "jakarta_t067_original.png": [(430, 278, 48), (485, 281, 55), (545, 281, 55), (600, 278, 48)],
}


def run(args: argparse.Namespace) -> dict:
    if "secondary-corridor-causal" in str(args.output.resolve()).replace("\\", "/").lower():
        raise ValueError("independent direction path is forbidden")
    report_path = args.output / "generation_report.json"
    if report_path.exists() or Path(str(report_path) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {report_path}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if not spec["isolation"]["train_only"] or spec["isolation"]["real_evaluation_credit"]:
        raise ValueError("matched pairs must remain train-only")
    source_root = args.spec.resolve().parent
    asset = base.normalized_asset(Image.open(args.asset))
    images_dir = args.output / "images" / "train"
    masks_dir = args.output / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for parent_name in sorted(CENTRAL):
        parent_path = source_root / "parents" / parent_name
        parent = Image.open(parent_path).convert("RGB")
        variants = {}
        for label, placements in (("clear", LATERAL[parent_name]), ("risk", CENTRAL[parent_name])):
            image, mask = base.compose(parent, asset, placements)
            invariant, outside = base.unchanged_outside_mask(parent, image, mask)
            if not invariant:
                raise ValueError(f"{parent_name} {label} changed {outside} pixels outside mask")
            stem = parent_path.stem.replace("_original", "")
            image_path = images_dir / f"{stem}_{label}.png"
            mask_path = masks_dir / f"{stem}_{label}_mask.png"
            image.save(image_path)
            mask.save(mask_path)
            variants[label] = {
                "image_path": str(image_path.resolve()),
                "image_sha256": base.sha256_file(image_path),
                "mask_path": str(mask_path.resolve()),
                "mask_sha256": base.sha256_file(mask_path),
                "placements": placements,
                "outside_mask_changed_pixel_count": outside,
            }
        pairs.append({
            "pair_id": parent_path.stem.replace("_original", ""),
            "parent_source_id": spec["parent_source_id"],
            "parent": {"path": str(parent_path.resolve()), "sha256": base.sha256_file(parent_path)},
            "clear": variants["clear"],
            "risk": variants["risk"],
            "same_asset_count": 4,
            "same_scale_depth_and_spacing": [row[2] for row in LATERAL[parent_name]] == [row[2] for row in CENTRAL[parent_name]],
            "horizontal_translation_only": all(a[1:] == b[1:] for a, b in zip(LATERAL[parent_name], CENTRAL[parent_name])),
            "human_truth_claimed": False,
            "real_evaluation_credit": False,
        })
    manifest = args.output / "manifest.jsonl"
    records = []
    for pair in pairs:
        for state in ("clear", "risk"):
            records.append({
                "image_path": str(Path(pair[state]["image_path"]).relative_to(args.output.resolve())).replace("\\", "/"),
                "split": "train",
                "class_labels": ["same_markers_outside_future_path" if state == "clear" else "same_markers_enter_future_path"],
                "attributes": {"counterfactual_pair_id": pair["pair_id"], "risk_state": state},
                "source": {"parent_source_id": pair["parent_source_id"], "parent_image_sha256": pair["parent"]["sha256"]},
                "image_sha256": pair[state]["image_sha256"],
                "synthetic": True,
                "human_truth_claimed": False,
                "real_evaluation_credit": False,
            })
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_spec": {"path": str(args.spec.resolve()), "sha256": base.sha256_file(args.spec)},
        "asset": {"path": str(args.asset.resolve()), "sha256": base.sha256_file(args.asset)},
        "manifest": {"path": str(manifest.resolve()), "sha256": base.sha256_file(manifest)},
        "pairs": pairs,
        "summary": {
            "pair_count": len(pairs),
            "image_count": len(records),
            "all_equal_count": all(row["same_asset_count"] == 4 for row in pairs),
            "all_equal_scale_depth_spacing": all(row["same_scale_depth_and_spacing"] for row in pairs),
            "all_horizontal_translation_only": all(row["horizontal_translation_only"] for row in pairs),
            "all_parent_pixel_invariants_passed": all(row[state]["outside_mask_changed_pixel_count"] == 0 for row in pairs for state in ("clear", "risk")),
        },
        "authorizations": {"representation_diagnostic": True, "training": False, "calibration": False, "blind": False, "android_runtime_change": False, "production_model_replacement": False},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(report_path) + ".sha256").write_text(base.sha256_file(report_path) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"ok": True, **result["summary"]}, ensure_ascii=False))
