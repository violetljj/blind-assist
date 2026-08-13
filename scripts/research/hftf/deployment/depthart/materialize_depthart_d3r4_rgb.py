#!/usr/bin/env python3
"""Materialize RGB for the exact D3R4 8 TRAIN + 8 DEVELOPMENT roster."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.census_depthart_task_preserving_d3r3_phase_b_source_coverage import (
    archive_coverage,
    download_file,
)
from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d3r3_phase_b_source_truth import (
    load_json,
    selected_stem_sha256,
    write_json_exclusive,
)
from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r3_phase_b_assets import (
    head,
    row_available,
)


ASSET = "lowres_wide.zip"
BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_truth_lookup(base: dict[str, Any], extension: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(base["processed"]) + list(extension["processed_extension"])
    result = {str(row["video_id"]): row for row in rows}
    require(len(result) == 53, "source-truth identity lookup drift")
    return result


def rgb_plan(
    phase_a_manifest: dict[str, Any], selective_canary: dict[str, Any],
    base_truth: dict[str, Any], extension_truth: dict[str, Any],
) -> list[dict[str, Any]]:
    phase_a = {
        str(row["video_id"]): row for row in phase_a_manifest["processed"] if row["eligible"] is True
    }
    require(len(phase_a) == 53, "Phase-A eligible identity count drift")
    truth = source_truth_lookup(base_truth, extension_truth)
    result: list[dict[str, Any]] = []
    for role in ("TRAIN", "DEVELOPMENT"):
        identities = selective_canary["candidate_role_split"][role]["identities"]
        require(len(identities) == 8, f"{role} identity count drift")
        for identity in identities:
            video_id = str(identity["video_id"])
            checkpoint = phase_a[video_id]
            stems = list(checkpoint["selected_frame_stems"])
            require(len(stems) == 300, "selected stem count drift")
            truth_row = truth[video_id]
            result.append({
                "role": role,
                "role_order": int(identity["role_order"]),
                "phase_a_eligible_order": int(identity["phase_a_eligible_order"]),
                "pool_order": int(identity["pool_order"]),
                "visit_id": str(identity["visit_id"]),
                "video_id": video_id,
                "fold": "Training",
                "asset": ASSET,
                "url": f"{BASE_URL}/{video_id}/{ASSET}",
                "selected_frame_stems": stems,
                "selected_frame_plan_sha256": selected_stem_sha256(stems),
                "depth_confidence_unavailable_stems": list(truth_row["source_unavailable_stems"]),
                "prior_head": {
                    "content_length_bytes": None,
                    "etag": None,
                    "last_modified": None,
                },
            })
    require(len(result) == 16 and len({row["video_id"] for row in result}) == 16, "RGB roster drift")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--base-source-truth-result", type=Path, required=True)
    parser.add_argument("--extension-result", type=Path, required=True)
    parser.add_argument("--selective-canary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    phase_a = load_json(args.phase_a_manifest)
    base = load_json(args.base_source_truth_result)
    extension = load_json(args.extension_result)
    selective = load_json(args.selective_canary)
    require(selective.get("status") == "D3R4_SELECTIVE_HORIZON_SOURCE_SUPPORT_PASS", "selective canary drift")
    require(selective.get("candidate_role_lock") is True, "candidate role roster is not locked")
    plan = rgb_plan(phase_a, selective, base, extension)
    if args.plan_only:
        print(json.dumps({
            "identity_count": len(plan),
            "TRAIN": [row["video_id"] for row in plan if row["role"] == "TRAIN"],
            "DEVELOPMENT": [row["video_id"] for row in plan if row["role"] == "DEVELOPMENT"],
            "selected_frame_count": sum(len(row["selected_frame_stems"]) for row in plan),
        }, sort_keys=True))
        return 0
    require(not args.output_root.exists(), f"fresh RGB root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    heads: list[dict[str, Any]] = []
    for row in plan:
        probed = head(
            row,
            timeout=20.0,
            max_attempts=3,
            user_agent="BlindAssist-DepthART-D3R4-RGB",
        )
        require(row_available(probed), f"RGB HEAD unavailable: {row['video_id']}")
        heads.append(probed)
    write_json_exclusive(args.output_root / "head.json", {
        "schema": "blindassist_depthart_d3r4_rgb_head_v1",
        "identity_count": 16,
        "asset_count": 16,
        "response_body_bytes_read": 0,
        "total_content_length_bytes": sum(int(row["content_length_bytes"]) for row in heads),
        "assets": heads,
    })

    processed: list[dict[str, Any]] = []
    for row in heads:
        video_id = str(row["video_id"])
        temporary = args.output_root / "_temporary_downloads" / video_id
        output = args.output_root / "source" / "Training" / video_id / ASSET
        try:
            receipt = download_file(row, output, temporary / "lowres_wide")
            coverage = archive_coverage(output, row["selected_frame_stems"])
            rgb_missing = list(coverage["selected_missing_stems"])
            unavailable_set = set(rgb_missing) | set(row["depth_confidence_unavailable_stems"])
            effective_missing = [
                stem for stem in row["selected_frame_stems"] if stem in unavailable_set
            ]
            processed.append({
                **{key: row[key] for key in (
                    "role", "role_order", "phase_a_eligible_order", "pool_order",
                    "visit_id", "video_id", "fold", "selected_frame_plan_sha256",
                )},
                "fixed_frame_count": 300,
                "rgb_coverage": coverage,
                "depth_confidence_unavailable_stems": row["depth_confidence_unavailable_stems"],
                "effective_multimodal_unavailable_stems": effective_missing,
                "effective_multimodal_available_frame_count": 300 - len(effective_missing),
                "source_asset": receipt,
            })
            print(json.dumps({
                "completed": len(processed),
                "total": 16,
                "role": row["role"],
                "video_id": video_id,
                "rgb_missing": len(rgb_missing),
                "effective_available": 300 - len(effective_missing),
            }, sort_keys=True), flush=True)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            parent = args.output_root / "_temporary_downloads"
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
    require(len(processed) == 16, "RGB materialization incomplete")
    minimum_available = min(row["effective_multimodal_available_frame_count"] for row in processed)
    passed = minimum_available >= 297
    result = {
        "schema": "blindassist_depthart_d3r4_rgb_materialization_v1",
        "status": "D3R4_RGB_SOURCE_PASS" if passed else "D3R4_RGB_SOURCE_NOT_EVALUABLE",
        "identity_count": 16,
        "TRAIN_identity_count": 8,
        "DEVELOPMENT_identity_count": 8,
        "fixed_frame_count": 4800,
        "minimum_effective_multimodal_available_frames_per_identity": minimum_available,
        "effective_multimodal_unavailable_frame_count": sum(
            300 - row["effective_multimodal_available_frame_count"] for row in processed
        ),
        "processed": processed,
        "model_output_read": False,
        "r2_access": "NONE",
        "next_action": "MATERIALIZE_D3R4_TRAIN_FEATURES_AND_SELECTIVE_TARGETS" if passed else None,
    }
    write_json_exclusive(args.output_root / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "minimum_available": minimum_available,
        "total_unavailable": result["effective_multimodal_unavailable_frame_count"],
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
