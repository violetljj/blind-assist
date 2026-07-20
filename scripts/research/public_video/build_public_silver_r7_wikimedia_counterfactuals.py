#!/usr/bin/env python3
"""Build r7 from r6 plus isolated Wikimedia static and dynamic pairs.

The source is the reviewed CC-BY-3.0 POPtravel Bangkok walking video mirrored
on Wikimedia Commons. Only RGB frames at pre-registered timestamps are
decoded. No source masks, independent-direction assets, calibration truth, or
blind data are read. The generated GPT/VLM labels remain provisional model
supervision and never authorize production replacement.
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


SCHEMA = "blindassist_public_silver_wikimedia_counterfactual_expansion_v1"
PACKAGE_NAME = "wikimedia-poptravel-bangkok-gpt5-20260717"
SOURCE_ID = "wikimedia_commons_poptravel_bangkok_sukhumvit_2019"
EXPECTED_VIDEO_SHA256 = "8f0efe24eddd939e8396abc60cfa35789003e9a3b9f115b9538182d0060e6a17"
STATIC_PAIR_ID = "wikimedia-poptravel-bangkok-sand-pile-pass-2098-2106"
DYNAMIC_PAIR_ID = "wikimedia-poptravel-bangkok-driveway-crossing-3064-3086"
FRAME_SPECS = (
    ("dynamic_clear_3064_0", 3064.0),
    ("dynamic_clear_3067_0", 3067.0),
    ("dynamic_clear_3069_5", 3069.5),
    ("dynamic_crossing_3083_5", 3083.5),
    ("dynamic_crossing_3084_5", 3084.5),
    ("dynamic_crossing_3085_5", 3085.5),
    ("static_narrow_2098_5", 2098.5),
    ("static_narrow_2100_0", 2100.0),
    ("static_narrow_2101_5", 2101.5),
    ("static_clear_2103_5", 2103.5),
    ("static_clear_2104_5", 2104.5),
    ("static_clear_2105_5", 2105.5),
)
PROMPT_TEXT = """Review two same-source counterfactual pairs from a continuous
first-person walking video. For the static pair, distinguish an approaching
sand pile that occupies the sidewalk corridor from the clear near field after
passing it. For the dynamic pair, distinguish a clear near driveway from a van
crossing immediately in front of the camera. Use multiframe temporal evidence,
abstain on ambiguity, and preserve that outputs are provisional machine silver,
not human event truth, calibration data, blind truth, or production approval."""


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(
            f"independent model direction is outside this builder's scope: {path}"
        )


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT_TEXT.encode("utf-8")).hexdigest()


def extract_timeline(*, video_path: Path, timeline_output_root: Path) -> dict[str, Any]:
    if timeline_output_root.exists():
        raise ValueError(f"refusing to overwrite timeline output: {timeline_output_root}")
    if sha256_file(video_path) != EXPECTED_VIDEO_SHA256:
        raise ValueError("Wikimedia source video SHA256 does not match the reviewed artifact")
    try:
        import cv2
    except ImportError as error:
        raise ValueError("OpenCV is required to decode the reviewed Wikimedia video") from error

    image_root = timeline_output_root / "images"
    image_root.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        shutil.rmtree(timeline_output_root, ignore_errors=True)
        raise ValueError(f"cannot open source video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[dict[str, Any]] = []
    try:
        for frame_index, (name, timestamp_seconds) in enumerate(FRAME_SPECS):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"cannot decode timestamp {timestamp_seconds:.1f}s")
            file_name = f"{frame_index:04d}_{name}.png"
            destination = image_root / file_name
            if not cv2.imwrite(str(destination), image):
                raise ValueError(f"cannot write decoded frame: {destination}")
            frames.append({
                "frame_index": frame_index,
                "source_frame_index": int(round(timestamp_seconds * fps)),
                "source_timestamp_ms": int(round(timestamp_seconds * 1000.0)),
                "file_name": file_name,
                "sha256": sha256_file(destination),
            })
    except Exception:
        capture.release()
        shutil.rmtree(timeline_output_root, ignore_errors=True)
        raise
    capture.release()

    manifest = {
        "format": "blindassist_public_rgb_timeline_source_manifest_v2",
        "source_id": SOURCE_ID,
        "source": {
            "dataset": "Wikimedia Commons public video",
            "file_title": "Walking in BANGKOK - Thailand - Sukhumvit Road - 4K 60fps (UHD).webm",
            "author": "POPtravel",
            "license": "CC-BY-3.0",
            "source_url": "https://commons.wikimedia.org/wiki/File:Walking_in_BANGKOK_-_Thailand_-_Sukhumvit_Road_-_4K_60fps_(UHD).webm",
            "original_source_url": "https://www.youtube.com/watch?v=DxP8fc2XYb4",
        },
        "license_review": {
            "status": "license_confirmed_by_youtube_review_bot",
            "reviewed_at": "2021-03-30",
            "file_page_url": "https://commons.wikimedia.org/wiki/File:Walking_in_BANGKOK_-_Thailand_-_Sukhumvit_Road_-_4K_60fps_(UHD).webm",
            "original_source_url": "https://www.youtube.com/watch?v=DxP8fc2XYb4",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "author": "POPtravel",
            "review_limit": "License confirmation does not create human event labels or privacy clearance.",
        },
        "source_video": {
            "path": str(video_path.resolve()),
            "sha256": EXPECTED_VIDEO_SHA256,
            "decoded_fps": fps,
        },
        "frame_count": len(frames),
        "frames": frames,
        "privacy_audit_required": True,
        "human_event_truth_present": False,
        "source_masks_or_geometry_used": False,
        "provisional_training_authorized": True,
        "training_execution_authorized": True,
        "production_model_replacement_authorized": False,
        "promotion": {
            "image_root": str(image_root.resolve()),
            "mode": "provisional_model_supervision",
            "important_limit": "Not human event truth, calibration data, blind-evaluation truth, or production authorization.",
        },
    }
    manifest_path = timeline_output_root / "source_manifest_v2.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest, "manifest_path": manifest_path, "image_root": image_root}


def build_silver(source_manifest_path: Path, frames: list[dict[str, Any]]) -> dict[str, Any]:
    hashes = {
        row["source_timestamp_ms"]: row["sha256"]
        for row in frames
    }
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
            "prompt_id": "blindassist-public-wikimedia-counterfactuals-v1",
            "prompt_sha256": prompt_sha256(),
            "review_mode": "multiframe_temporal",
        },
        "episodes": [
            {
                "episode_id": "wikimedia-bangkok-driveway-clear-nearfield-3064-3070",
                "evidence_frame_sha256": [
                    hashes[index] for index in (3064000, 3067000, 3069500)
                ],
                "silver_should_alert": "candidate_no_alert",
                "confidence": 0.72,
                "counterfactual_pair_id": DYNAMIC_PAIR_ID,
                "risk_profile": {
                    "risk_mechanism": "dynamic_agent_approach",
                    "primary_hazard_type": "driveway_cross_traffic_still_distant",
                    "corridor_relation": "near_field_sidewalk_clear",
                    "lifecycle": "no_alert",
                    "counterfactual_pair_id": DYNAMIC_PAIR_ID,
                },
                "negative_decision_quality": {
                    "corridor_heading_stability": 0.76,
                    "near_field_visibility": 0.82,
                    "corridor_clearance": 0.78,
                    "near_field_lateral_intrusion_absent": 0.74,
                },
                "uncertainty_reasons": [
                    "Vehicles are visible across the driveway, but none occupies the immediate near field.",
                    "The intended continuation remains the forward sidewalk beside the building.",
                ],
            },
            {
                "episode_id": "wikimedia-bangkok-van-near-crossing-3083-3086",
                "evidence_frame_sha256": [
                    hashes[index] for index in (3083500, 3084500, 3085500)
                ],
                "silver_should_alert": "candidate_alert",
                "confidence": 0.88,
                "counterfactual_pair_id": DYNAMIC_PAIR_ID,
                "risk_profile": {
                    "risk_mechanism": "dynamic_agent_approach",
                    "primary_hazard_type": "van_crossing_driveway_near_camera",
                    "corridor_relation": "near_field_crossing_and_full_visual_occlusion",
                    "lifecycle": "approach_alertable",
                    "counterfactual_pair_id": DYNAMIC_PAIR_ID,
                },
                "uncertainty_reasons": [
                    "The exact vehicle trajectory is inferred from sparse decoded timestamps.",
                    "The van nevertheless fills the near field and crosses the forward route continuously.",
                ],
            },
            {
                "episode_id": "wikimedia-bangkok-sand-pile-near-corridor-2098-2102",
                "evidence_frame_sha256": [
                    hashes[index] for index in (2098500, 2100000, 2101500)
                ],
                "silver_should_alert": "candidate_alert",
                "confidence": 0.86,
                "counterfactual_pair_id": STATIC_PAIR_ID,
                "risk_profile": {
                    "risk_mechanism": "static_corridor_narrowing",
                    "primary_hazard_type": "construction_sand_pile",
                    "corridor_relation": "occupies_central_sidewalk_and_forces_lateral_pass",
                    "lifecycle": "approach_alertable",
                    "counterfactual_pair_id": STATIC_PAIR_ID,
                },
                "uncertainty_reasons": [
                    "A narrow left-side passage remains, so the scene is a warning rather than a total blockage.",
                    "The exact usable width cannot be measured from the low-resolution public transcode.",
                ],
            },
            {
                "episode_id": "wikimedia-bangkok-clear-after-sand-pile-2103-2106",
                "evidence_frame_sha256": [
                    hashes[index] for index in (2103500, 2104500, 2105500)
                ],
                "silver_should_alert": "candidate_no_alert",
                "confidence": 0.78,
                "counterfactual_pair_id": STATIC_PAIR_ID,
                "risk_profile": {
                    "risk_mechanism": "static_corridor_narrowing",
                    "primary_hazard_type": "sand_pile_already_passed",
                    "corridor_relation": "near_field_forward_sidewalk_clear",
                    "lifecycle": "no_alert",
                    "counterfactual_pair_id": STATIC_PAIR_ID,
                },
                "negative_decision_quality": {
                    "corridor_heading_stability": 0.83,
                    "near_field_visibility": 0.88,
                    "corridor_clearance": 0.82,
                    "near_field_lateral_intrusion_absent": 0.80,
                },
                "uncertainty_reasons": [
                    "Street traffic remains lateral and does not occupy the immediate sidewalk corridor.",
                    "The negative interval is after passing the pile rather than a separately captured empty construction site.",
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
    video_path: Path,
    timeline_output_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    paths = [
        path.resolve()
        for path in (parent_root, video_path, timeline_output_root, output_root)
    ]
    parent_root, video_path, timeline_output_root, output_root = paths
    for path in paths:
        reject_independent_direction(path)
    if output_root.exists():
        raise ValueError(f"refusing to overwrite output root: {output_root}")
    parent_packages = sorted(
        path.parent for path in parent_root.glob("*/silver_labels_v2.json")
    )
    if not parent_packages:
        raise ValueError(f"parent package contains no silver manifests: {parent_root}")

    timeline = extract_timeline(
        video_path=video_path,
        timeline_output_root=timeline_output_root,
    )
    output_root.mkdir(parents=True)
    try:
        for parent_package in parent_packages:
            shutil.copytree(parent_package, output_root / parent_package.name)
        package_root = output_root / PACKAGE_NAME
        package_root.mkdir()
        source_path = package_root / "source_manifest_v2.json"
        shutil.copy2(timeline["manifest_path"], source_path)
        source = load_json(source_path)
        source["promotion"]["image_root"] = str(timeline["image_root"].resolve())
        source_path.write_text(
            json.dumps(source, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        silver = build_silver(source_path, source["frames"])
        silver_path = package_root / "silver_labels_v2.json"
        silver_path.write_text(
            json.dumps(silver, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validation = validate(silver, source_manifest_path=source_path)
        receipt = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_name": PACKAGE_NAME,
            "change": "added_reviewed_wikimedia_static_and_dynamic_counterfactual_pairs",
            "parent_root": str(parent_root),
            "timeline_output_root": str(timeline_output_root),
            "source_video_sha256": EXPECTED_VIDEO_SHA256,
            "source_manifest_v2_sha256": sha256_file(source_path),
            "silver_labels_v2_sha256": sha256_file(silver_path),
            "validation": validation,
            "counterfactual_pair_ids": [DYNAMIC_PAIR_ID, STATIC_PAIR_ID],
            "minimum_pair_confidence": 0.72,
            "confidence_qualified_at_0_65": True,
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
            "package_count": len(parent_packages) + 1,
            "added_package": PACKAGE_NAME,
            "added_counterfactual_pair_ids": [DYNAMIC_PAIR_ID, STATIC_PAIR_ID],
            "confidence_qualified_at_0_65": True,
            "isolation_contract": {
                "public_video_mainline_only": True,
                "independent_model_direction_data_used": False,
                "independent_model_direction_metrics_used_as_gate": False,
            },
        }
        receipt_path = output_root / "r7_build_receipt.json"
        receipt_path.write_text(
            json.dumps(root_receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        Path(str(receipt_path) + ".sha256").write_text(
            sha256_file(receipt_path) + "\n",
            encoding="ascii",
        )
        return root_receipt
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        shutil.rmtree(timeline_output_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timeline-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = build(
            parent_root=args.parent_root,
            video_path=args.video,
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
