#!/usr/bin/env python3
"""Confirm the frozen D3R5 veto on fresh parent model outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.research.hftf.deployment.depthart.analyze_depthart_d3r5_parent_relative_veto import (
    parent_grid_relative_features,
    predict,
    route_veto,
)
from scripts.research.hftf.deployment.depthart.analyze_depthart_task_preserving_d3r3_cohort_composition import (
    geometry_observable,
)
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
from scripts.research.hftf.deployment.depthart.run_depthart_d3r4_selective_router_canary import (
    CertificateHead,
    _load_depthart,
    materialize_role,
    metrics,
    require,
    sha256_file,
)


ASSET = "lowres_wide.zip"
BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training"
EXPECTED_FRESH_VIDEO_IDS = (
    "47331091",
    "42899325",
    "41126374",
    "42923208",
    "42897523",
    "41069178",
    "44796487",
    "42898799",
)


def select_fresh_rows(
    rows: list[dict[str, Any]], used_video_ids: set[str], count: int = 8
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if geometry_observable(row) and str(row["video_id"]) not in used_video_ids
    ]
    eligible.sort(key=lambda row: (int(row["pool_order"]), str(row["video_id"])))
    require(len(eligible) >= count, "insufficient fresh geometry-observable parents")
    return eligible[:count]


def confirmation_plan(
    phase_a: dict[str, Any],
    base_truth: dict[str, Any],
    extension_truth: dict[str, Any],
    selective: dict[str, Any],
) -> list[dict[str, Any]]:
    phase_rows = {
        str(row["video_id"]): row
        for row in phase_a["processed"]
        if row["eligible"] is True
    }
    require(len(phase_rows) == 53, "Phase-A eligible identity count drift")
    truth_rows = list(base_truth["processed"]) + list(
        extension_truth["processed_extension"]
    )
    require(len(truth_rows) == 53, "source-truth identity count drift")
    used = {
        str(row["video_id"])
        for role in ("TRAIN", "DEVELOPMENT")
        for row in selective["candidate_role_split"][role]["identities"]
    }
    require(len(used) == 16, "D3R4 used identity count drift")
    chosen = select_fresh_rows(truth_rows, used)
    require(
        tuple(str(row["video_id"]) for row in chosen) == EXPECTED_FRESH_VIDEO_IDS,
        "fresh confirmation roster drift",
    )
    result: list[dict[str, Any]] = []
    for confirmation_order, truth_row in enumerate(chosen, start=1):
        video_id = str(truth_row["video_id"])
        phase_row = phase_rows[video_id]
        stems = list(phase_row["selected_frame_stems"])
        require(len(stems) == 300, "selected stem count drift")
        unavailable = list(truth_row["source_unavailable_stems"])
        require(len(unavailable) <= 3, "source coverage below 99 percent")
        result.append({
            "role": "FRESH_CONFIRMATION",
            "role_order": confirmation_order,
            "confirmation_order": confirmation_order,
            "phase_a_eligible_order": int(truth_row["selection_order"]),
            "pool_order": int(truth_row["pool_order"]),
            "visit_id": str(truth_row["visit_id"]),
            "video_id": video_id,
            "fold": "Training",
            "asset": ASSET,
            "url": f"{BASE_URL}/{video_id}/{ASSET}",
            "selected_frame_stems": stems,
            "selected_frame_plan_sha256": selected_stem_sha256(stems),
            "depth_confidence_unavailable_stems": unavailable,
            "prior_head": {
                "content_length_bytes": None,
                "etag": None,
                "last_modified": None,
            },
        })
    require(len({row["video_id"] for row in result}) == 8, "fresh roster duplicate")
    return result


def load_frozen_head(
    checkpoint: dict[str, Any], result: dict[str, Any]
) -> tuple[CertificateHead, np.ndarray, np.ndarray, float]:
    require(
        result.get("status")
        == "D3R5_PARENT_RELATIVE_ZERO_FALSE_BLOCK_VETO_DISCOVERY_SUPPORTED",
        "D3R5 discovery status drift",
    )
    require(
        checkpoint.get("schema")
        == "blindassist_depthart_d3r5_parent_relative_veto_checkpoint_v1",
        "D3R5 checkpoint schema drift",
    )
    threshold = float(checkpoint["threshold"])
    require(threshold == float(result["selected_threshold"]), "threshold drift")
    head_model = CertificateHead().to(dtype=torch.float64)
    head_model.load_state_dict({
        name: torch.as_tensor(value, dtype=torch.float64)
        for name, value in checkpoint["head"].items()
    })
    return (
        head_model,
        np.asarray(checkpoint["mean"], dtype=np.float64),
        np.asarray(checkpoint["std"], dtype=np.float64),
        threshold,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--base-source-truth-result", type=Path, required=True)
    parser.add_argument("--extension-result", type=Path, required=True)
    parser.add_argument("--selective-canary", type=Path, required=True)
    parser.add_argument("--d3r5-root", type=Path, required=True)
    parser.add_argument("--depthart-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-depth-root", type=Path, required=True)
    parser.add_argument("--extension-depth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    phase_a = load_json(args.phase_a_manifest)
    base_truth = load_json(args.base_source_truth_result)
    extension_truth = load_json(args.extension_result)
    selective = load_json(args.selective_canary)
    require(
        selective.get("status") == "D3R4_SELECTIVE_HORIZON_SOURCE_SUPPORT_PASS",
        "selective source canary drift",
    )
    plan = confirmation_plan(phase_a, base_truth, extension_truth, selective)
    if args.plan_only:
        print(json.dumps({
            "identity_count": 8,
            "video_ids": [row["video_id"] for row in plan],
            "pool_orders": [row["pool_order"] for row in plan],
            "fixed_frame_count": 2400,
            "previous_model_output_overlap": 0,
        }, sort_keys=True))
        return 0
    require(not args.output_root.exists(), f"fresh output root exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    probed_rows: list[dict[str, Any]] = []
    for row in plan:
        probed = head(
            row,
            timeout=20.0,
            max_attempts=3,
            user_agent="BlindAssist-DepthART-D3R5-Fresh-Confirmation-RGB",
        )
        require(row_available(probed), f"RGB HEAD unavailable: {row['video_id']}")
        probed_rows.append(probed)
    write_json_exclusive(args.output_root / "rgb-head.json", {
        "schema": "blindassist_depthart_d3r5_fresh_confirmation_rgb_head_v1",
        "identity_count": 8,
        "asset_count": 8,
        "response_body_bytes_read": 0,
        "total_content_length_bytes": sum(
            int(row["content_length_bytes"]) for row in probed_rows
        ),
        "assets": probed_rows,
    })
    processed: list[dict[str, Any]] = []
    model_plan: list[dict[str, Any]] = []
    for row in probed_rows:
        video_id = str(row["video_id"])
        temporary = args.output_root / "_temporary_downloads" / video_id
        output = args.output_root / "source" / "Training" / video_id / ASSET
        try:
            receipt = download_file(row, output, temporary / "lowres_wide")
            coverage = archive_coverage(output, row["selected_frame_stems"])
            rgb_missing = list(coverage["selected_missing_stems"])
            unavailable_set = set(rgb_missing) | set(
                row["depth_confidence_unavailable_stems"]
            )
            effective_missing = [
                stem for stem in row["selected_frame_stems"] if stem in unavailable_set
            ]
            require(len(effective_missing) <= 3, "effective source coverage below 99 percent")
            processed.append({
                **{key: row[key] for key in (
                    "confirmation_order",
                    "phase_a_eligible_order",
                    "pool_order",
                    "visit_id",
                    "video_id",
                    "fold",
                    "selected_frame_plan_sha256",
                )},
                "fixed_frame_count": 300,
                "rgb_coverage": coverage,
                "depth_confidence_unavailable_stems": row[
                    "depth_confidence_unavailable_stems"
                ],
                "effective_multimodal_unavailable_stems": effective_missing,
                "effective_multimodal_available_frame_count": 300
                - len(effective_missing),
                "source_asset": receipt,
            })
            model_plan.append({
                "role": "FRESH_CONFIRMATION",
                "role_order": int(row["confirmation_order"]),
                "pool_order": int(row["pool_order"]),
                "visit_id": str(row["visit_id"]),
                "video_id": video_id,
                "selected_frame_stems": list(row["selected_frame_stems"]),
                "unavailable_stems": effective_missing,
            })
            print(json.dumps({
                "stage": "RGB_MATERIALIZATION",
                "completed": len(processed),
                "total": 8,
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
    require(len(processed) == 8, "fresh RGB materialization incomplete")
    write_json_exclusive(args.output_root / "rgb-result.json", {
        "schema": "blindassist_depthart_d3r5_fresh_confirmation_rgb_result_v1",
        "status": "D3R5_FRESH_CONFIRMATION_RGB_PASS",
        "identity_count": 8,
        "fixed_frame_count": 2400,
        "minimum_effective_multimodal_available_frames_per_identity": min(
            row["effective_multimodal_available_frame_count"] for row in processed
        ),
        "processed": processed,
        "model_output_read": False,
        "r2_access": "NONE",
    })
    require(args.checkpoint.is_file(), "DepthART checkpoint missing")
    require(
        sha256_file(args.checkpoint)
        == "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65",
        "DepthART checkpoint drift",
    )
    model, preprocess = _load_depthart(args.depthart_source, args.checkpoint)
    dataset, dataset_meta = materialize_role(
        "FRESH_CONFIRMATION",
        model_plan,
        phase_a_root=args.phase_a_manifest.parent,
        base_depth_root=args.base_depth_root,
        extension_depth_root=args.extension_depth_root,
        rgb_root=args.output_root,
        model=model,
        preprocess=preprocess,
        batch_size=args.batch_size,
    )
    np.savez_compressed(args.output_root / "confirmation-dataset.npz", **dataset)
    del model
    torch.cuda.empty_cache()
    discovery_result_path = args.d3r5_root / "result.json"
    discovery_checkpoint_path = args.d3r5_root / "veto-checkpoint.json"
    discovery_result = load_json(discovery_result_path)
    discovery_checkpoint = load_json(discovery_checkpoint_path)
    require(
        sha256_file(discovery_checkpoint_path)
        == discovery_result["checkpoint"]["sha256"],
        "D3R5 checkpoint hash drift",
    )
    head_model, mean, std, threshold = load_frozen_head(
        discovery_checkpoint, discovery_result
    )
    relative = parent_grid_relative_features(dataset)
    probabilities = predict(relative, head_model, mean, std)
    states, actions = route_veto(dataset, probabilities, threshold)
    baseline = metrics(dataset, dataset["baseline_state"])
    candidate = metrics(dataset, states)
    base_pooled = baseline["pooled"]
    candidate_pooled = candidate["pooled"]
    false_clear_improvement = (
        base_pooled["false_clear_all_known"]
        - candidate_pooled["false_clear_all_known"]
    )
    false_block_improvement = (
        base_pooled["false_block_given_clear"]
        - candidate_pooled["false_block_given_clear"]
    )
    coverage_decrease = (
        base_pooled["known_coverage_all_cells"]
        - candidate_pooled["known_coverage_all_cells"]
    )
    passed = bool(
        actions["direct_veto_actions"] > 0
        and false_clear_improvement >= 0.01
        and false_block_improvement >= -0.01
        and coverage_decrease <= 0.02
    )
    result = {
        "schema": "blindassist_depthart_d3r5_parent_relative_veto_fresh_confirmation_result_v1",
        "status": (
            "D3R5_PARENT_RELATIVE_ZERO_FALSE_BLOCK_VETO_FRESH_CONFIRMATION_PASS"
            if passed
            else "D3R5_PARENT_RELATIVE_ZERO_FALSE_BLOCK_VETO_FRESH_CONFIRMATION_FAIL"
        ),
        "hypothesis": "The TRAIN-frozen parent-relative high-precision veto transfers to parent-disjoint identities whose DepthART outputs were not previously read.",
        "identity_count": 8,
        "fixed_frame_count": 2400,
        "identity_selection": "first eight pool-ordered geometry-observable parents excluding all D3R4 TRAIN and DEVELOPMENT identities; selected before model output",
        "previous_model_output_overlap": 0,
        "dataset": dataset_meta,
        "selected_threshold": threshold,
        "threshold_or_weight_tuning_on_confirmation": False,
        "actions": actions,
        "baseline": baseline,
        "candidate": candidate,
        "false_clear_all_known_improvement": false_clear_improvement,
        "false_block_given_clear_improvement": false_block_improvement,
        "known_coverage_decrease": coverage_decrease,
        "decision_rule": {
            "false_clear_improvement_min": 0.01,
            "false_block_improvement_min": -0.01,
            "known_coverage_decrease_max": 0.02,
        },
        "input_bindings": {
            "discovery_result": {
                "path": str(discovery_result_path.resolve()),
                "bytes": discovery_result_path.stat().st_size,
                "sha256": sha256_file(discovery_result_path),
            },
            "discovery_checkpoint": {
                "path": str(discovery_checkpoint_path.resolve()),
                "bytes": discovery_checkpoint_path.stat().st_size,
                "sha256": sha256_file(discovery_checkpoint_path),
            },
        },
        "source_unavailable_as_negative": False,
        "far_clear_as_negative": False,
        "r2_access": "NONE",
        "performance_claim": False,
        "production_authority": False,
        "elapsed_seconds_diagnostic_only": time.time() - started,
        "next_action": (
            "PROMOTE_PARENT_RELATIVE_VETO_TO_DEVELOPMENT_CANDIDATE"
            if passed
            else "PRESERVE_NEGATIVE_AND_RETHINK_PARENT_SHIFT"
        ),
    }
    write_json_exclusive(args.output_root / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "threshold": threshold,
        "actions": actions,
        "baseline_false_clear": base_pooled["false_clear_all_known"],
        "candidate_false_clear": candidate_pooled["false_clear_all_known"],
        "baseline_false_block": base_pooled["false_block_given_clear"],
        "candidate_false_block": candidate_pooled["false_block_given_clear"],
        "known_coverage_decrease": coverage_decrease,
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
