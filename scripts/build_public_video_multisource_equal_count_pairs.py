#!/usr/bin/env python3
"""Build source-bound equal-count path-relation counterfactual pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

import build_public_video_path_intrusion_counterfactuals as base


SCHEMA = "blindassist_public_video_multisource_equal_count_pairs_v1"


def placements(value: Any, *, name: str) -> list[tuple[int, int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} placements must be a non-empty list")
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3 or not all(isinstance(item, int) for item in row):
            raise ValueError(f"{name} placement must be [center_x, base_y, height]")
        if row[2] <= 0:
            raise ValueError(f"{name} placement height must be positive")
        rows.append(tuple(row))
    return rows


def validate_spec(spec: dict[str, Any]) -> None:
    isolation = spec.get("isolation", {})
    required_false = (
        "real_evaluation_credit",
        "human_truth_claimed",
        "training_authorized",
        "android_runtime_change_authorized",
    )
    if isolation.get("train_only") is not True or any(isolation.get(key) is not False for key in required_false):
        raise ValueError("dataset must be train-only and cannot authorize truth, training, evaluation, or Android changes")
    pairs = spec.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("spec must contain at least one pair")
    pair_ids = [row.get("pair_id") for row in pairs]
    if any(not isinstance(value, str) or not value for value in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair_id values must be unique non-empty strings")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if "secondary-corridor-causal" in str(output).replace("\\", "/").lower():
        raise ValueError("independent direction path is forbidden")
    report_path = output / "generation_report.json"
    if report_path.exists() or Path(str(report_path) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {report_path}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    validate_spec(spec)
    asset = base.normalized_asset(Image.open(args.asset))
    source_root = args.spec.resolve().parent
    images_dir = output / "images" / "train"
    masks_dir = output / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    generated_pairs = []
    manifest_rows = []
    for pair in spec["pairs"]:
        pair_id = pair["pair_id"]
        parent_path = source_root / "parents" / pair["parent_filename"]
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        actual_parent_sha = base.sha256_file(parent_path)
        if actual_parent_sha != pair["parent_sha256"]:
            raise ValueError(f"parent hash differs from frozen spec: {pair_id}")
        parent = Image.open(parent_path).convert("RGB")
        if parent.size != (640, 360):
            raise ValueError(f"unexpected parent size: {parent_path} {parent.size}")
        clear = placements(pair["clear_placements"], name=f"{pair_id} clear")
        risk = placements(pair["risk_placements"], name=f"{pair_id} risk")
        if len(clear) != len(risk):
            raise ValueError(f"asset count differs between pair states: {pair_id}")
        if [row[1:] for row in clear] != [row[1:] for row in risk]:
            raise ValueError(f"pair must differ by horizontal translation only: {pair_id}")
        if [row[0] for row in clear] == [row[0] for row in risk]:
            raise ValueError(f"pair has no horizontal intervention: {pair_id}")

        variants = {}
        for state, state_placements in (("clear", clear), ("risk", risk)):
            edited, mask = base.compose(parent, asset, state_placements)
            invariant, outside = base.unchanged_outside_mask(parent, edited, mask)
            if not invariant:
                raise ValueError(f"{pair_id} {state} changed {outside} pixels outside mask")
            image_path = images_dir / f"{pair_id}_{state}.png"
            mask_path = masks_dir / f"{pair_id}_{state}_mask.png"
            edited.save(image_path)
            mask.save(mask_path)
            variants[state] = {
                "image_path": str(image_path.resolve()),
                "image_sha256": base.sha256_file(image_path),
                "mask_path": str(mask_path.resolve()),
                "mask_sha256": base.sha256_file(mask_path),
                "placements": [list(row) for row in state_placements],
                "outside_mask_changed_pixel_count": outside,
            }
            manifest_rows.append({
                "image_path": str(image_path.relative_to(output)).replace("\\", "/"),
                "split": "train",
                "class_labels": ["same_marker_outside_future_path" if state == "clear" else "same_marker_enters_future_path"],
                "attributes": {"counterfactual_pair_id": pair_id, "risk_state": state},
                "source": {
                    "parent_source_id": pair["parent_source_id"],
                    "parent_image_sha256": actual_parent_sha,
                    "parent_timestamp_ms": pair["parent_timestamp_ms"],
                },
                "image_sha256": variants[state]["image_sha256"],
                "synthetic": True,
                "human_truth_claimed": False,
                "real_evaluation_credit": False,
            })
        generated_pairs.append({
            "pair_id": pair_id,
            "parent_source_id": pair["parent_source_id"],
            "parent": {"path": str(parent_path.resolve()), "sha256": actual_parent_sha, "timestamp_ms": pair["parent_timestamp_ms"]},
            "clear": variants["clear"],
            "risk": variants["risk"],
            "same_asset_count": len(clear),
            "same_scale_depth_and_spacing": True,
            "horizontal_translation_only": True,
            "train_only": True,
            "human_truth_claimed": False,
            "real_evaluation_credit": False,
        })

    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_spec": {"path": str(args.spec.resolve()), "sha256": base.sha256_file(args.spec)},
        "asset": {"path": str(args.asset.resolve()), "sha256": base.sha256_file(args.asset)},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": base.sha256_file(manifest_path)},
        "pairs": generated_pairs,
        "summary": {
            "pair_count": len(generated_pairs),
            "source_count": len({row["parent_source_id"] for row in generated_pairs}),
            "image_count": len(manifest_rows),
            "all_equal_count": all(row["same_asset_count"] > 0 for row in generated_pairs),
            "all_equal_scale_depth_spacing": all(row["same_scale_depth_and_spacing"] for row in generated_pairs),
            "all_horizontal_translation_only": all(row["horizontal_translation_only"] for row in generated_pairs),
            "all_parent_pixel_invariants_passed": all(row[state]["outside_mask_changed_pixel_count"] == 0 for row in generated_pairs for state in ("clear", "risk")),
        },
        "authorizations": {
            "representation_diagnostic": True,
            "five_prototype_bootstrap_short_runs": False,
            "training": False,
            "calibration": False,
            "blind": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
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
