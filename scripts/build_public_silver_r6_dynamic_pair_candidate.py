#!/usr/bin/env python3
"""Build public-video r6 with an isolated low-confidence JtMY dynamic pair.

The parent r5 package remains immutable. Earlier RGB from the same official
SANPO train session is copied into a new public-video timeline beside the
existing horse-crossing frames. The original .63 alert confidence is retained,
so the pair can be tested without falsely satisfying the confidence-qualified
mechanism coverage gate.
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


SCHEMA = "blindassist_public_silver_dynamic_pair_candidate_v1"
PACKAGE_NAME = "sanpo-jtmy-horse-carriage-gpt5-20260716"
SOURCE_ID = "sanpo_real_JtMYI6rJ4wiDsEVffAkee0kR5Zmrf8vM"
PAIR_ID = "sanpo-jtmy-horse-carriage-approach-0226-0429"
EARLY_FILES = (
    ("sanpo_JtMYI6rJ4wiDsEVffAkee0kR5Zmrf8vM_camera_chest_left_000226_10fps_000000.png", 226),
    ("sanpo_JtMYI6rJ4wiDsEVffAkee0kR5Zmrf8vM_camera_chest_left_000226_10fps_000024.png", 262),
    ("sanpo_JtMYI6rJ4wiDsEVffAkee0kR5Zmrf8vM_camera_chest_left_000226_10fps_000049.png", 300),
)
LATE_FILES = (
    ("0000_000339.png", 339),
    ("0003_000384.png", 384),
    ("0006_000429.png", 429),
)
PROMPT_TEXT = """Compare two intervals from the same forward walking session.
The negative interval must show the same route before the horse carriage enters
the near crossing area. The positive interval must preserve uncertainty about
the user's intended crossing direction. These labels are provisional model
supervision, not human event truth or calibration evidence."""


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this builder's scope: {path}")


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT_TEXT.encode("utf-8")).hexdigest()


def copy_timeline(*, early_root: Path, late_root: Path, timeline_output_root: Path) -> dict[str, Any]:
    if timeline_output_root.exists():
        raise ValueError(f"refusing to overwrite timeline output: {timeline_output_root}")
    image_root = timeline_output_root / "images"
    image_root.mkdir(parents=True)
    frames: list[dict[str, Any]] = []
    try:
        for source_root, rows in ((early_root, EARLY_FILES), (late_root, LATE_FILES)):
            for source_name, source_frame_index in rows:
                source_path = source_root / source_name
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                destination_name = f"{len(frames):04d}_{source_frame_index:06d}.png"
                destination = image_root / destination_name
                shutil.copy2(source_path, destination)
                frames.append({
                    "frame_index": len(frames),
                    "source_frame_index": source_frame_index,
                    "file_name": destination_name,
                    "sha256": sha256_file(destination),
                })
        manifest = {
            "format": "blindassist_public_rgb_timeline_source_manifest_v2",
            "source_id": SOURCE_ID,
            "source": {
                "dataset": "SANPO-Real v0",
                "official_split": "train",
                "session_id": "JtMYI6rJ4wiDsEVffAkee0kR5Zmrf8vM",
                "camera": "camera_chest",
                "lens": "left",
                "license": "CC-BY-4.0",
            },
            "frame_count": len(frames),
            "frames": frames,
            "privacy_audit_required": True,
            "human_event_truth_present": False,
            "source_masks_or_geometry_used": False,
            "training_execution_authorized": True,
            "production_model_replacement_authorized": False,
            "provisional_training_authorized": True,
            "promotion": {
                "image_root": str(image_root.resolve()),
                "mode": "provisional_model_supervision",
                "important_limit": "Not human event truth, calibration data, blind-evaluation truth, or production authorization.",
            },
        }
        manifest_path = timeline_output_root / "source_manifest_v2.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"manifest": manifest, "manifest_path": manifest_path, "image_root": image_root}
    except Exception:
        shutil.rmtree(timeline_output_root, ignore_errors=True)
        raise


def build_silver(source_manifest_path: Path, frames: list[dict[str, Any]]) -> dict[str, Any]:
    hashes = {row["source_frame_index"]: row["sha256"] for row in frames}
    return {
        "schema": "blindassist_public_video_silver_labels_v2",
        "source": {
            "source_id": SOURCE_ID,
            "source_manifest_path": "source_manifest_v2.json",
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "human_event_truth_present": False,
            "privacy_audit_required": True,
        },
        "labeler": {
            "provider": "openai",
            "model": "gpt-5",
            "prompt_id": "blindassist-public-dynamic-pair-candidate-v1",
            "prompt_sha256": prompt_sha256(),
            "review_mode": "multiframe_temporal",
        },
        "episodes": [
            {
                "episode_id": "sanpo-jtmy-open-approach-before-carriage-0226-0300",
                "evidence_frame_sha256": [hashes[index] for index in (226, 262, 300)],
                "silver_should_alert": "candidate_no_alert",
                "confidence": 0.72,
                "counterfactual_pair_id": PAIR_ID,
                "risk_profile": {
                    "risk_mechanism": "dynamic_agent_approach",
                    "primary_hazard_type": "horse_carriage_not_yet_in_near_crossing",
                    "corridor_relation": "same_approach_route_open",
                    "lifecycle": "no_alert",
                    "counterfactual_pair_id": PAIR_ID,
                },
                "negative_decision_quality": {
                    "corridor_heading_stability": 0.73,
                    "near_field_visibility": 0.82,
                    "corridor_clearance": 0.80,
                    "near_field_lateral_intrusion_absent": 0.76,
                },
                "uncertainty_reasons": [
                    "The route curves toward an intersection, so later crossing intent is not fully observable.",
                    "Pedestrians ahead remain passable and do not occupy the immediate near-field corridor.",
                ],
            },
            {
                "episode_id": "sanpo-jtmy-horse-carriage-near-crossing-0339-0429",
                "evidence_frame_sha256": [hashes[index] for index in (339, 384, 429)],
                "silver_should_alert": "candidate_alert",
                "confidence": 0.63,
                "counterfactual_pair_id": PAIR_ID,
                "risk_profile": {
                    "risk_mechanism": "dynamic_agent_approach",
                    "primary_hazard_type": "near_crossing_large_animal_vehicle",
                    "corridor_relation": "enters_or_blocks_possible_crossing_route",
                    "lifecycle": "approach_alertable",
                    "counterfactual_pair_id": PAIR_ID,
                },
                "uncertainty_reasons": [
                    "The camera is at a street corner rather than a fixed pedestrian corridor, so intended crossing direction remains uncertain.",
                    "The horse and carriage are close, but sparse evidence cannot establish the earliest safe alert time.",
                ],
            },
        ],
        "training_execution_authorized": True,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
        "training_mode": "provisional_model_supervision",
    }


def build(
    *,
    parent_root: Path,
    early_root: Path,
    late_root: Path,
    timeline_output_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    paths = [path.resolve() for path in (parent_root, early_root, late_root, timeline_output_root, output_root)]
    parent_root, early_root, late_root, timeline_output_root, output_root = paths
    for path in paths:
        reject_independent_direction(path)
    if output_root.exists():
        raise ValueError(f"refusing to overwrite output root: {output_root}")
    parent_packages = sorted(path.parent for path in parent_root.glob("*/silver_labels_v2.json"))
    if not any(path.name == PACKAGE_NAME for path in parent_packages):
        raise ValueError(f"required JtMY package is missing under {parent_root}")

    timeline = copy_timeline(
        early_root=early_root,
        late_root=late_root,
        timeline_output_root=timeline_output_root,
    )
    output_root.mkdir(parents=True)
    try:
        for parent_package in parent_packages:
            shutil.copytree(parent_package, output_root / parent_package.name)
        package_root = output_root / PACKAGE_NAME
        source_path = package_root / "source_manifest_v2.json"
        shutil.copy2(timeline["manifest_path"], source_path)
        source = load_json(source_path)
        source["promotion"]["image_root"] = str(timeline["image_root"].resolve())
        source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        silver = build_silver(source_path, source["frames"])
        silver_path = package_root / "silver_labels_v2.json"
        silver_path.write_text(json.dumps(silver, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        validation = validate(silver, source_manifest_path=source_path)
        receipt = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_name": PACKAGE_NAME,
            "change": "expanded_single_alert_to_low_confidence_same_source_dynamic_pair",
            "parent_root": str(parent_root),
            "timeline_output_root": str(timeline_output_root),
            "source_manifest_v2_sha256": sha256_file(source_path),
            "silver_labels_v2_sha256": sha256_file(silver_path),
            "validation": validation,
            "minimum_pair_confidence": 0.63,
            "confidence_qualified_at_0_65": False,
            "independent_model_directions_used": False,
            "human_event_truth_present": False,
            "calibration_authorized": False,
            "blind_evaluation_authorized": False,
            "production_model_replacement_authorized": False,
        }
        (package_root / "promotion_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        root_receipt = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_root": str(parent_root),
            "output_root": str(output_root),
            "package_count": len(parent_packages),
            "changed_packages": [PACKAGE_NAME],
            "added_counterfactual_pair_id": PAIR_ID,
            "pair_is_exploratory_below_coverage_confidence": True,
            "isolation_contract": {
                "public_video_mainline_only": True,
                "independent_model_direction_data_used": False,
                "independent_model_direction_metrics_used_as_gate": False,
            },
        }
        receipt_path = output_root / "r6_build_receipt.json"
        receipt_path.write_text(json.dumps(root_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        Path(str(receipt_path) + ".sha256").write_text(sha256_file(receipt_path) + "\n", encoding="ascii")
        return root_receipt
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        shutil.rmtree(timeline_output_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--early-root", type=Path, required=True)
    parser.add_argument("--late-root", type=Path, required=True)
    parser.add_argument("--timeline-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = build(
            parent_root=args.parent_root,
            early_root=args.early_root,
            late_root=args.late_root,
            timeline_output_root=args.timeline_output_root,
            output_root=args.output_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **receipt}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
