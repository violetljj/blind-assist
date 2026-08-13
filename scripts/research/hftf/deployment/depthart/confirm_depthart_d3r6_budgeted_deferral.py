#!/usr/bin/env python3
"""Fresh-parent confirmation of the frozen D3R6 UNKNOWN budget."""

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
)
from scripts.research.hftf.deployment.depthart.analyze_depthart_d3r6_budgeted_deferral import (
    budgeted_deferral,
    load_frozen_score,
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
    _load_depthart,
    materialize_role,
    metrics,
    require,
    sha256_file,
)


ASSET = "lowres_wide.zip"
BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training"
EXPECTED_VIDEO_IDS = (
    "47333061",
    "41126786",
    "42899596",
    "43828361",
    "47429874",
    "41126829",
    "43896117",
    "42899490",
)


def next_fresh_plan(
    phase_a: dict[str, Any],
    base_truth: dict[str, Any],
    extension_truth: dict[str, Any],
    d3r4_selective: dict[str, Any],
    d3r5_confirmation: dict[str, Any],
) -> list[dict[str, Any]]:
    phase_rows = {
        str(row["video_id"]): row
        for row in phase_a["processed"]
        if row["eligible"] is True
    }
    truth_rows = list(base_truth["processed"]) + list(
        extension_truth["processed_extension"]
    )
    used = {
        str(row["video_id"])
        for role in ("TRAIN", "DEVELOPMENT")
        for row in d3r4_selective["candidate_role_split"][role]["identities"]
    }
    used.update(
        str(row["video_id"])
        for row in d3r5_confirmation["dataset"]["identities"]
    )
    require(len(used) == 24, "prior model-output roster drift")
    candidates = [
        row
        for row in truth_rows
        if geometry_observable(row) and str(row["video_id"]) not in used
    ]
    candidates.sort(key=lambda row: (int(row["pool_order"]), str(row["video_id"])))
    chosen = candidates[:8]
    require(
        tuple(str(row["video_id"]) for row in chosen) == EXPECTED_VIDEO_IDS,
        "D3R6 fresh roster drift",
    )
    result: list[dict[str, Any]] = []
    for order, truth_row in enumerate(chosen, start=1):
        video_id = str(truth_row["video_id"])
        phase_row = phase_rows[video_id]
        stems = list(phase_row["selected_frame_stems"])
        unavailable = list(truth_row["source_unavailable_stems"])
        require(len(stems) == 300 and len(unavailable) <= 3, "source plan drift")
        result.append({
            "role": "FRESH_CONFIRMATION",
            "role_order": order,
            "confirmation_order": order,
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
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--base-source-truth-result", type=Path, required=True)
    parser.add_argument("--extension-result", type=Path, required=True)
    parser.add_argument("--selective-canary", type=Path, required=True)
    parser.add_argument("--d3r5-confirmation-root", type=Path, required=True)
    parser.add_argument("--d3r5-discovery-root", type=Path, required=True)
    parser.add_argument("--d3r6-root", type=Path, required=True)
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
    d3r5_confirmation = load_json(args.d3r5_confirmation_root / "result.json")
    require(
        d3r5_confirmation.get("status")
        == "D3R5_PARENT_RELATIVE_ZERO_FALSE_BLOCK_VETO_FRESH_CONFIRMATION_FAIL",
        "D3R5 confirmation binding drift",
    )
    plan = next_fresh_plan(
        phase_a, base_truth, extension_truth, selective, d3r5_confirmation
    )
    if args.plan_only:
        print(json.dumps({
            "identity_count": 8,
            "video_ids": [row["video_id"] for row in plan],
            "pool_orders": [row["pool_order"] for row in plan],
            "fixed_frame_count": 2400,
            "prior_model_output_overlap": 0,
        }, sort_keys=True))
        return 0
    require(not args.output_root.exists(), f"fresh output root exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    heads: list[dict[str, Any]] = []
    for row in plan:
        probed = head(
            row,
            timeout=20.0,
            max_attempts=3,
            user_agent="BlindAssist-DepthART-D3R6-Fresh-Confirmation-RGB",
        )
        require(row_available(probed), f"RGB HEAD unavailable: {row['video_id']}")
        heads.append(probed)
    write_json_exclusive(args.output_root / "rgb-head.json", {
        "schema": "blindassist_depthart_d3r6_fresh_confirmation_rgb_head_v1",
        "identity_count": 8,
        "asset_count": 8,
        "response_body_bytes_read": 0,
        "total_content_length_bytes": sum(
            int(row["content_length_bytes"]) for row in heads
        ),
        "assets": heads,
    })
    model_plan: list[dict[str, Any]] = []
    rgb_rows: list[dict[str, Any]] = []
    for row in heads:
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
            unavailable = [
                stem for stem in row["selected_frame_stems"] if stem in unavailable_set
            ]
            require(len(unavailable) <= 3, "effective source coverage below 99 percent")
            rgb_rows.append({
                "confirmation_order": int(row["confirmation_order"]),
                "pool_order": int(row["pool_order"]),
                "visit_id": str(row["visit_id"]),
                "video_id": video_id,
                "rgb_coverage": coverage,
                "effective_multimodal_unavailable_stems": unavailable,
                "effective_multimodal_available_frame_count": 300 - len(unavailable),
                "source_asset": receipt,
            })
            model_plan.append({
                "role": "FRESH_CONFIRMATION",
                "role_order": int(row["confirmation_order"]),
                "pool_order": int(row["pool_order"]),
                "visit_id": str(row["visit_id"]),
                "video_id": video_id,
                "selected_frame_stems": list(row["selected_frame_stems"]),
                "unavailable_stems": unavailable,
            })
            print(json.dumps({
                "stage": "RGB_MATERIALIZATION",
                "completed": len(rgb_rows),
                "total": 8,
                "video_id": video_id,
                "effective_available": 300 - len(unavailable),
            }, sort_keys=True), flush=True)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            parent = args.output_root / "_temporary_downloads"
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
    write_json_exclusive(args.output_root / "rgb-result.json", {
        "schema": "blindassist_depthart_d3r6_fresh_confirmation_rgb_result_v1",
        "status": "D3R6_FRESH_CONFIRMATION_RGB_PASS",
        "identity_count": 8,
        "fixed_frame_count": 2400,
        "processed": rgb_rows,
        "model_output_read": False,
        "r2_access": "NONE",
    })
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
    dataset_path = args.output_root / "confirmation-dataset.npz"
    np.savez_compressed(dataset_path, **dataset)
    del model
    torch.cuda.empty_cache()
    d3r6_result_path = args.d3r6_root / "result.json"
    d3r6_checkpoint_path = args.d3r6_root / "deferral-checkpoint.json"
    d3r6_result = load_json(d3r6_result_path)
    d3r6_checkpoint = load_json(d3r6_checkpoint_path)
    require(
        d3r6_result.get("status")
        == "D3R6_BUDGETED_UNKNOWN_DEFERRAL_DISCOVERY_SUPPORTED",
        "D3R6 discovery status drift",
    )
    require(
        sha256_file(d3r6_checkpoint_path) == d3r6_result["checkpoint"]["sha256"],
        "D3R6 checkpoint hash drift",
    )
    budget = float(d3r6_checkpoint["budget_fraction_of_all_cells_per_parent"])
    require(budget == float(d3r6_result["selected_budget_fraction"]), "budget drift")
    score_checkpoint_path = args.d3r5_discovery_root / "veto-checkpoint.json"
    score_checkpoint = load_json(score_checkpoint_path)
    require(
        sha256_file(score_checkpoint_path)
        == d3r6_checkpoint["source_score_checkpoint"]["sha256"],
        "risk score checkpoint drift",
    )
    score_head, mean, std = load_frozen_score(score_checkpoint)
    probabilities = predict(
        parent_grid_relative_features(dataset), score_head, mean, std
    )
    states, actions = budgeted_deferral(dataset, probabilities, budget)
    baseline = metrics(dataset, dataset["baseline_state"])
    candidate = metrics(dataset, states)
    base = baseline["pooled"]
    cand = candidate["pooled"]
    false_clear_improvement = (
        base["false_clear_all_known"] - cand["false_clear_all_known"]
    )
    false_block_improvement = (
        base["false_block_given_clear"] - cand["false_block_given_clear"]
    )
    coverage_decrease = (
        base["known_coverage_all_cells"] - cand["known_coverage_all_cells"]
    )
    passed = bool(
        actions["deferred_cell_count"] > 0
        and false_clear_improvement >= 0.01
        and false_block_improvement >= 0.0
        and coverage_decrease <= 0.0200000001
    )
    result = {
        "schema": "blindassist_depthart_d3r6_budgeted_deferral_fresh_confirmation_result_v1",
        "status": (
            "D3R6_BUDGETED_UNKNOWN_DEFERRAL_FRESH_CONFIRMATION_PASS"
            if passed
            else "D3R6_BUDGETED_UNKNOWN_DEFERRAL_FRESH_CONFIRMATION_FAIL"
        ),
        "identity_count": 8,
        "fixed_frame_count": 2400,
        "identity_selection": "next eight pool-ordered geometry-observable parents excluding all prior D3R4 and D3R5 model-output identities; selected before model output",
        "prior_model_output_overlap": 0,
        "dataset": dataset_meta,
        "budget_fraction": budget,
        "budget_or_weight_tuning_on_confirmation": False,
        "actions": actions,
        "baseline": baseline,
        "candidate": candidate,
        "false_clear_all_known_improvement": false_clear_improvement,
        "false_block_given_clear_improvement": false_block_improvement,
        "known_coverage_decrease": coverage_decrease,
        "decision_rule": d3r6_result["decision_rule"],
        "source_unavailable_as_negative": False,
        "unknown_as_negative": False,
        "r2_access": "NONE",
        "performance_claim": False,
        "production_authority": False,
        "input_bindings": {
            "d3r6_result": {
                "path": str(d3r6_result_path.resolve()),
                "bytes": d3r6_result_path.stat().st_size,
                "sha256": sha256_file(d3r6_result_path),
            },
            "d3r6_checkpoint": {
                "path": str(d3r6_checkpoint_path.resolve()),
                "bytes": d3r6_checkpoint_path.stat().st_size,
                "sha256": sha256_file(d3r6_checkpoint_path),
            },
            "confirmation_dataset": {
                "path": str(dataset_path.resolve()),
                "bytes": dataset_path.stat().st_size,
                "sha256": sha256_file(dataset_path),
            },
        },
        "elapsed_seconds_diagnostic_only": time.time() - started,
        "next_action": (
            "LOCK_D3R6_AS_DEVELOPMENT_CANDIDATE_AND_DOCUMENT_CONTRIBUTION"
            if passed
            else "PRESERVE_NEGATIVE_AND_RETHINK_SELECTIVE_ACTION"
        ),
    }
    write_json_exclusive(args.output_root / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "budget_fraction": budget,
        "actions": actions,
        "baseline_false_clear": base["false_clear_all_known"],
        "candidate_false_clear": cand["false_clear_all_known"],
        "baseline_false_block": base["false_block_given_clear"],
        "candidate_false_block": cand["false_block_given_clear"],
        "known_coverage_decrease": coverage_decrease,
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
