#!/usr/bin/env python3
"""Expand D3R3 source-truth support to the remaining 21 continuity parents."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from scripts.research.assistive_geometry.arkitscenes_truth_reader import TruthReaderPolicy
from scripts.research.hftf.deployment.depthart.analyze_depthart_task_preserving_d3r3_cohort_composition import (
    geometry_observable,
    solve_role_split,
)
from scripts.research.hftf.deployment.depthart.census_depthart_task_preserving_d3r3_phase_b_source_coverage import (
    archive_coverage,
    download_file,
)
from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d3r3_phase_b_source_truth import (
    evaluate_identity,
    load_json,
    selected_stem_sha256,
    write_json_exclusive,
)
from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r3_phase_b_assets import (
    head,
    row_available,
)


ASSETS = ("lowres_depth.zip", "confidence.zip")
BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def remaining_eligible_plan(phase_a_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected_keys = {
        (int(row["pool_order"]), str(row["visit_id"]), str(row["video_id"]))
        for row in phase_a_manifest["selected_phase_a"]
    }
    eligible = [
        row for row in phase_a_manifest["processed"]
        if row["eligible"] is True
        and (int(row["pool_order"]), str(row["visit_id"]), str(row["video_id"]))
        not in selected_keys
    ]
    eligible.sort(key=lambda row: int(row["pool_order"]))
    require(len(eligible) == 21, f"remaining eligible count drift: {len(eligible)}")
    result: list[dict[str, Any]] = []
    for extension_order, row in enumerate(eligible, start=1):
        stems = list(row["selected_frame_stems"])
        require(len(stems) == 300 and len(set(stems)) == 300, "extension frame plan drift")
        result.append({
            "selection_order": 32 + extension_order,
            "extension_order": extension_order,
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": "Training",
            "selected_frame_stems": stems,
            "selected_frame_plan_sha256": selected_stem_sha256(stems),
            "phase_a_checkpoint": row,
        })
    return result


def asset_plan(identities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        identity | {
            "asset": asset,
            "url": f"{BASE_URL}/{identity['video_id']}/{asset}",
            "prior_head": {
                "content_length_bytes": None,
                "etag": None,
                "last_modified": None,
            },
        }
        for identity in identities
        for asset in ASSETS
    ]


def coverage_row(
    identity: dict[str, Any], depth: dict[str, Any], confidence: dict[str, Any]
) -> dict[str, Any]:
    selected = identity["selected_frame_stems"]
    depth_missing = list(depth["selected_missing_stems"])
    confidence_missing = list(confidence["selected_missing_stems"])
    missing_set = set(depth_missing) | set(confidence_missing)
    paired_missing = [stem for stem in selected if stem in missing_set]
    paired_present = [stem for stem in selected if stem not in missing_set]
    return {
        "selection_order": identity["selection_order"],
        "pool_order": identity["pool_order"],
        "visit_id": identity["visit_id"],
        "video_id": identity["video_id"],
        "fold": "Training",
        "selected_frame_count": 300,
        "selected_frame_plan_sha256": identity["selected_frame_plan_sha256"],
        "lowres_depth": depth,
        "confidence": confidence,
        "paired_exact_present_count": len(paired_present),
        "paired_exact_missing_count": len(paired_missing),
        "paired_exact_missing_stems": paired_missing,
        "paired_exact_present_stems_sha256": selected_stem_sha256(paired_present),
        "neighbor_substitution_used": False,
        "source_truth_derived": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--base-source-truth-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    phase_a_manifest = load_json(args.phase_a_manifest)
    base_result = load_json(args.base_source_truth_result)
    identities = remaining_eligible_plan(phase_a_manifest)
    plan = asset_plan(identities)
    if args.plan_only:
        print(json.dumps({
            "identity_count": len(identities),
            "asset_count": len(plan),
            "selected_frame_count": sum(len(row["selected_frame_stems"]) for row in identities),
            "pool_orders": [row["pool_order"] for row in identities],
        }, sort_keys=True))
        return 0
    require(not args.output_root.exists(), f"fresh output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)

    head_rows: list[dict[str, Any]] = []
    for row in plan:
        probed = head(
            row,
            timeout=20.0,
            max_attempts=3,
            user_agent="BlindAssist-DepthART-D3R3-rare-support-extension",
        )
        require(row_available(probed), f"HEAD unavailable: {row['video_id']}/{row['asset']}")
        head_rows.append(probed)
    write_json_exclusive(args.output_root / "head.json", {
        "schema": "blindassist_depthart_d3r3_rare_support_extension_head_v1",
        "identity_count": 21,
        "asset_count": 42,
        "response_body_bytes_read": 0,
        "assets": head_rows,
    })

    head_lookup = {(str(row["video_id"]), str(row["asset"])): row for row in head_rows}
    policy = TruthReaderPolicy()
    policy.validate()
    processed: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for identity in identities:
        video_id = str(identity["video_id"])
        source_dir = args.output_root / "source" / "Training" / video_id
        temporary = args.output_root / "_temporary_downloads" / video_id
        receipts: dict[str, dict[str, Any]] = {}
        try:
            for asset in ASSETS:
                receipts[asset] = download_file(
                    head_lookup[(video_id, asset)],
                    source_dir / asset,
                    temporary / asset.replace(".zip", ""),
                )
            depth_coverage = archive_coverage(
                source_dir / "lowres_depth.zip", identity["selected_frame_stems"]
            )
            confidence_coverage = archive_coverage(
                source_dir / "confidence.zip", identity["selected_frame_stems"]
            )
            coverage = coverage_row(identity, depth_coverage, confidence_coverage)
            audit_identity = identity | {"coverage": coverage}
            truth = evaluate_identity(
                audit_identity,
                args.phase_a_manifest.parent,
                args.output_root,
                policy,
            )
            truth["source_assets"] = [receipts[asset] for asset in ASSETS]
            coverage_rows.append(coverage)
            processed.append(truth)
            print(json.dumps({
                "completed": len(processed),
                "total": 21,
                "video_id": video_id,
                "available": truth["source_available_frame_count"],
                "geometry_observable": geometry_observable(truth),
                "far_center_clear": truth["truth_support"]["clear_by_grid"]["center@2.0m"],
            }, sort_keys=True), flush=True)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            parent = args.output_root / "_temporary_downloads"
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()

    require(len(processed) == 21, "extension incomplete")
    combined = list(base_result["processed"]) + processed
    require(len(combined) == 53, "combined pool count drift")
    train, development, optimum = solve_role_split(combined)
    minimum_parent_contributors = min(
        train["minimum_contributing_parent_count"],
        development["minimum_contributing_parent_count"],
    )
    candidate_lock = minimum_parent_contributors >= 2
    result = {
        "schema": "blindassist_depthart_task_preserving_d3r3_rare_support_extension_v1",
        "status": (
            "D3R3_RARE_SUPPORT_EXTENSION_PARENT_DIVERSITY_PASS"
            if candidate_lock
            else "D3R3_RARE_SUPPORT_EXTENSION_PARENT_DIVERSITY_INSUFFICIENT"
        ),
        "hypothesis": "Additional continuity-eligible parents increase rare CLEAR parent diversity enough for an 8+8 parent-disjoint role split.",
        "base_identity_count": 32,
        "extension_identity_count": 21,
        "combined_identity_count": 53,
        "extension_selected_frame_count": 6300,
        "extension_source_available_frame_count": sum(row["source_available_frame_count"] for row in processed),
        "extension_source_unavailable_frame_count": sum(row["source_unavailable_frame_count"] for row in processed),
        "extension_geometry_observable_parent_count": sum(geometry_observable(row) for row in processed),
        "combined_geometry_observable_parent_count": sum(geometry_observable(row) for row in combined),
        "max_min_role_stratum_count": optimum,
        "minimum_parent_contributors_per_role_stratum": minimum_parent_contributors,
        "candidate_role_split": {"TRAIN": train, "DEVELOPMENT": development},
        "candidate_role_lock": candidate_lock,
        "coverage": coverage_rows,
        "processed_extension": processed,
        "rgb_read": False,
        "model_output_read": False,
        "r2_access": "NONE",
        "next_action": (
            "REGISTER_PHASE_C_RGB_FOR_CANDIDATE_16"
            if candidate_lock
            else "RETHINK_RARE_CLEAR_ACQUISITION_OR_TASK_DEFINITION"
        ),
    }
    write_json_exclusive(args.output_root / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "extension_missing": result["extension_source_unavailable_frame_count"],
        "geometry_observable": result["combined_geometry_observable_parent_count"],
        "max_min": optimum,
        "minimum_parent_contributors": minimum_parent_contributors,
        "candidate_role_lock": candidate_lock,
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
