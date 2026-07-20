#!/usr/bin/env python3
"""Build public-video r5 by adding a second static-corridor matched source.

The parent r4 package remains immutable. This builder copies only the
public-video mainline packages and adds a hash-bound SANPO Chcne episode pair:
far/passable static furniture versus near-field static chokepoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_public_video_silver_labels import load_json, sha256_file, validate


SCHEMA = "blindassist_public_silver_static_mechanism_expansion_v1"
PACKAGE_NAME = "sanpo-chcne-static-chokepoint-gpt5-20260717"
PAIR_ID = "sanpo-chcne-static-furniture-approach-0177-0252"
PROMPT_TEXT = """Review temporally ordered public RGB for a static navigation chokepoint.
Compare the same walking corridor while fixed obstacles are distant/passable
versus near enough to materially narrow the route. Object presence alone is
not an alert. Keep lateral seated people separate from the static corridor
mechanism. Labels are provisional model supervision, never human event truth."""


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this builder's scope: {path}")


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT_TEXT.encode("utf-8")).hexdigest()


def build_silver(source_manifest_path: Path) -> dict[str, Any]:
    return {
        "schema": "blindassist_public_video_silver_labels_v2",
        "source": {
            "source_id": "sanpo_real_ChcneDEXiD6dzznmQNtTnlVflFy2nK2s",
            "source_manifest_path": "source_manifest_v2.json",
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "human_event_truth_present": False,
            "privacy_audit_required": True,
        },
        "labeler": {
            "provider": "openai",
            "model": "gpt-5",
            "prompt_id": "blindassist-public-static-chokepoint-silver-v1",
            "prompt_sha256": prompt_sha256(),
            "review_mode": "multiframe_temporal",
        },
        "episodes": [
            {
                "episode_id": "sanpo-chcne-static-furniture-passable-0177-0207",
                "evidence_frame_sha256": [
                    "aaf2de445911b539f74910b49cac5d65bc489cf0b0f3646eccfa87647bbd28d1",
                    "a01863d9c717132ca81fddc1f3b070d60e551c668cdf49b47d1595e194df8732",
                    "b116db00eb822a4e98323972be391f5c8eef9d11ada3f3236ad0a1bf808eb833",
                ],
                "silver_should_alert": "candidate_no_alert",
                "confidence": 0.71,
                "counterfactual_pair_id": PAIR_ID,
                "risk_profile": {
                    "risk_mechanism": "static_corridor_narrowing",
                    "primary_hazard_type": "distant_static_trash_bins_and_sign",
                    "corridor_relation": "distant_and_passable",
                    "lifecycle": "no_alert",
                    "counterfactual_pair_id": PAIR_ID,
                },
                "negative_decision_quality": {
                    "corridor_heading_stability": 0.80,
                    "near_field_visibility": 0.86,
                    "corridor_clearance": 0.75,
                    "near_field_lateral_intrusion_absent": 0.72,
                },
                "uncertainty_reasons": [
                    "Seated people remain on the right edge but do not occupy the intended center passage.",
                    "The precise intended route is inferred from forward motion between the bins and standing sign.",
                ],
            },
            {
                "episode_id": "sanpo-chcne-static-furniture-chokepoint-0222-0252",
                "evidence_frame_sha256": [
                    "6135a76f50e10caf8fdd6f48dc59d32c0cb42b1ecfa85a371664a6b044387692",
                    "4f6e7866e6dd40d980ebd4a434d0a72284773928214e0b383486bc5a5ce0bc46",
                    "8f02fd3da37caa1bfb684d232eebb84dedc15e8045a1224e5436412b63528ac9",
                ],
                "silver_should_alert": "candidate_alert",
                "confidence": 0.70,
                "counterfactual_pair_id": PAIR_ID,
                "risk_profile": {
                    "risk_mechanism": "static_corridor_narrowing",
                    "primary_hazard_type": "static_trash_bins_and_standing_sign",
                    "corridor_relation": "near_field_chokepoint",
                    "lifecycle": "approach_alertable",
                    "counterfactual_pair_id": PAIR_ID,
                },
                "uncertainty_reasons": [
                    "The passage remains physically traversable, so this is a narrowing warning rather than a complete blockage.",
                    "Seated people and benches are lateral context and may still influence generic visual features.",
                ],
            },
        ],
        "training_execution_authorized": True,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
        "training_mode": "provisional_model_supervision",
    }


def build(*, parent_root: Path, timeline_root: Path, source_manifest: Path, output_root: Path) -> dict[str, Any]:
    parent_root = parent_root.resolve()
    timeline_root = timeline_root.resolve()
    source_manifest = source_manifest.resolve()
    output_root = output_root.resolve()
    for path in (parent_root, timeline_root, source_manifest, output_root):
        reject_independent_direction(path)
    if output_root.exists():
        raise ValueError(f"refusing to overwrite output root: {output_root}")
    image_root = (timeline_root / "images").resolve()
    if not parent_root.is_dir() or not source_manifest.is_file() or not image_root.is_dir():
        raise FileNotFoundError("parent root, Chcne source manifest, or image root is missing")

    parent_packages = sorted(path.parent for path in parent_root.glob("*/silver_labels_v2.json"))
    if not parent_packages:
        raise ValueError(f"no public-silver packages found under {parent_root}")
    output_root.mkdir(parents=True)
    try:
        for parent_package in parent_packages:
            shutil.copytree(parent_package, output_root / parent_package.name)

        package_root = output_root / PACKAGE_NAME
        package_root.mkdir()
        source = load_json(source_manifest)
        source["promotion"] = {
            "source_manifest_candidate_sha256": sha256_file(source_manifest),
            "source_manifest_candidate_path": str(source_manifest),
            "image_root": str(image_root),
            "mode": "provisional_model_supervision",
            "important_limit": "Not human event truth, calibration data, blind-evaluation truth, or production authorization.",
        }
        source_path = package_root / "source_manifest_v2.json"
        source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        silver = build_silver(source_path)
        silver_path = package_root / "silver_labels_v2.json"
        silver_path.write_text(json.dumps(silver, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        validation = validate(silver, source_manifest_path=source_path)
        package_receipt = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_name": PACKAGE_NAME,
            "parent_root": str(parent_root),
            "timeline_root": str(timeline_root),
            "source_manifest_v2_sha256": sha256_file(source_path),
            "silver_labels_v2_sha256": sha256_file(silver_path),
            "validation": validation,
            "risk_mechanism": "static_corridor_narrowing",
            "independent_model_directions_used": False,
            "human_event_truth_present": False,
            "calibration_authorized": False,
            "blind_evaluation_authorized": False,
            "production_model_replacement_authorized": False,
        }
        (package_root / "promotion_receipt.json").write_text(
            json.dumps(package_receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        root_receipt = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_root": str(parent_root),
            "output_root": str(output_root),
            "copied_package_count": len(parent_packages),
            "added_package": PACKAGE_NAME,
            "total_package_count": len(parent_packages) + 1,
            "added_episode_count": 2,
            "added_counterfactual_pair_id": PAIR_ID,
            "isolation_contract": {
                "public_video_mainline_only": True,
                "independent_model_direction_data_used": False,
                "independent_model_direction_metrics_used_as_gate": False,
            },
        }
        receipt_path = output_root / "r5_build_receipt.json"
        receipt_path.write_text(json.dumps(root_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        Path(str(receipt_path) + ".sha256").write_text(sha256_file(receipt_path) + "\n", encoding="ascii")
        return root_receipt
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--timeline-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = build(
            parent_root=args.parent_root,
            timeline_root=args.timeline_root,
            source_manifest=args.source_manifest,
            output_root=args.output_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **receipt}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
