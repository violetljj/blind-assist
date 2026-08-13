#!/usr/bin/env python3
"""Replay the frozen AG depth selector against its strict no-regret gate.

This is a Development-only canary.  It reuses the selector's frozen parent
split and threshold, exposes the perfect signed-advantage oracle as an upper
bound, and never opens the external Bonn/TUM evaluation cohorts or ETH3D.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_no_regret_selector_bonn import load_selector
from train_ag_st_bonn_anchored_student import (
    DEFAULT_BONN_ARCHIVE,
    DEFAULT_BONN_CATALOG,
    DEFAULT_BONN_RECEIPT,
    DEFAULT_BONN_ROOT,
    DEFAULT_COHORT_MANIFEST,
    DEFAULT_LABEL_DIRS,
    DEFAULT_STAGE0A_RESULTS,
)
from train_ag_st_masked_student import (
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    build_frame_descriptor_batches,
    extract_depthart_features,
    write_json_exclusive,
)
from train_ag_st_no_regret_selector import (
    collect_selector_observations,
    extract_bonn_anchor_frames,
    extract_tum_anchor_frames,
    summarize_selector_observations,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTOR_ROOT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-no-regret-selector-three-domain-global-group-dro-r0"
)
DEFAULT_SELECTOR_RESULT = DEFAULT_SELECTOR_ROOT / "result.json"
DEFAULT_SELECTOR_CHECKPOINT = DEFAULT_SELECTOR_ROOT / "no-regret-selector.pt"
DEFAULT_TUM_COHORT_MANIFEST = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_TUM_RGBD_THIRD_DOMAIN_COHORT_R0_2026-08-10.json"
)
DEFAULT_BOUNDARY_RESULT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R21_TARGET_MASS_NORMALIZED_ANGULAR_BOUNDARY_RESULT_2026-08-11.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-factorwise-no-regret-oracle-parent-gate-canary-r0/result.json"
)


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"{label} missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{label} root invalid")
    return payload


def evaluate_lane_gate(
    summary: dict[str, Any],
    lane: str,
    *,
    minimum_coverage: float = 0.01,
    minimum_nonzero_parent_fraction: float = 0.5,
    metric_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Apply the same parent-bounded gate to the selector or perfect oracle."""

    require(lane in {"selected", "oracle"}, "no-regret lane invalid")
    require(0.0 <= minimum_coverage <= 1.0, "minimum coverage invalid")
    require(
        0.0 <= minimum_nonzero_parent_fraction <= 1.0,
        "minimum parent fraction invalid",
    )
    parent_rows = summary.get("per_parent")
    require(isinstance(parent_rows, list) and parent_rows, "parent summary empty")
    coverage_key = f"{lane}_coverage_fraction"
    harmful_parents: list[dict[str, Any]] = []
    nonzero_parents: list[str] = []
    for row in parent_rows:
        parent_id = str(row["parent_id"])
        coverage = float(row[coverage_key])
        if coverage > 0.0:
            nonzero_parents.append(parent_id)
        mae_delta = float(row[lane]["mae_m"]) - float(row["base"]["mae_m"])
        bad_delta = float(row[lane]["bad_gt_0_10_m_fraction"]) - float(
            row["base"]["bad_gt_0_10_m_fraction"]
        )
        if mae_delta > metric_tolerance or bad_delta > metric_tolerance:
            harmful_parents.append(
                {
                    "parent_id": parent_id,
                    "domain": str(row["domain"]),
                    "mae_delta_vs_base_m": mae_delta,
                    "bad_gt_0_10_m_delta_vs_base": bad_delta,
                }
            )

    parent_macro = summary["parent_macro"]
    macro_mae_delta = float(parent_macro[lane]["mae_m"]) - float(
        parent_macro["base"]["mae_m"]
    )
    macro_bad_delta = float(parent_macro[lane]["bad_gt_0_10_m_fraction"]) - float(
        parent_macro["base"]["bad_gt_0_10_m_fraction"]
    )
    domain_failures: list[dict[str, Any]] = []
    for domain, values in sorted(summary["by_domain"].items()):
        mae_delta = float(values[lane]["mae_m"]) - float(values["base"]["mae_m"])
        bad_delta = float(values[lane]["bad_gt_0_10_m_fraction"]) - float(
            values["base"]["bad_gt_0_10_m_fraction"]
        )
        if mae_delta > metric_tolerance or bad_delta > metric_tolerance:
            domain_failures.append(
                {
                    "domain": domain,
                    "mae_delta_vs_base_m": mae_delta,
                    "bad_gt_0_10_m_delta_vs_base": bad_delta,
                }
            )

    coverage = float(parent_macro[coverage_key])
    nonzero_parent_fraction = len(nonzero_parents) / len(parent_rows)
    checks = {
        "minimum_macro_coverage": coverage >= minimum_coverage,
        "macro_mae_no_regret": macro_mae_delta <= metric_tolerance,
        "macro_bad_rate_no_regret": macro_bad_delta <= metric_tolerance,
        "all_domains_no_regret": not domain_failures,
        "all_parents_no_regret": not harmful_parents,
        "minimum_nonzero_parent_fraction": (
            nonzero_parent_fraction >= minimum_nonzero_parent_fraction
        ),
    }
    return {
        "lane": lane,
        "pass": all(checks.values()),
        "checks": checks,
        "minimum_coverage": minimum_coverage,
        "minimum_nonzero_parent_fraction": minimum_nonzero_parent_fraction,
        "metric_tolerance": metric_tolerance,
        "parent_macro_coverage_fraction": coverage,
        "parent_macro_mae_delta_vs_base_m": macro_mae_delta,
        "parent_macro_bad_gt_0_10_m_delta_vs_base": macro_bad_delta,
        "nonzero_parent_count": len(nonzero_parents),
        "nonzero_parent_fraction": nonzero_parent_fraction,
        "nonzero_parents": sorted(nonzero_parents),
        "harmful_parent_count": len(harmful_parents),
        "harmful_parents": harmful_parents,
        "domain_failure_count": len(domain_failures),
        "domain_failures": domain_failures,
    }


def derive_terminal(
    oracle_gate: dict[str, Any], selector_gate: dict[str, Any]
) -> tuple[str, str]:
    if not oracle_gate["pass"]:
        return (
            "AG_FACTORWISE_NO_REGRET_ORACLE_SAFE_COVERAGE_NOT_SUPPORTED",
            "STOP_DEPTH_ROUTER_RETAIN_FROZEN_BOUNDARY",
        )
    if selector_gate["pass"]:
        return (
            "AG_FACTORWISE_NO_REGRET_ORACLE_AND_PARENT_GATE_CANARY_PASS",
            "FREEZE_SELECTOR_AND_REQUIRE_FRESH_PARENT_DISJOINT_EVALUATION",
        )
    return (
        "AG_FACTORWISE_NO_REGRET_ORACLE_HEADROOM_SELECTOR_GATE_FAIL",
        "TRAIN_ONE_SIDED_ADVANTAGE_LCB_ROUTER_ON_FIT_ONLY",
    )


def _assert_close(left: float, right: float, label: str, tolerance: float) -> None:
    require(math.isfinite(left) and math.isfinite(right), f"{label} non-finite")
    require(abs(left - right) <= tolerance, f"{label} replay drift")


def validate_frozen_metric_replay(
    stored: dict[str, Any], replayed: dict[str, Any], *, tolerance: float = 1e-5
) -> dict[str, Any]:
    """Confirm base/expert/selector replay before adding new oracle metrics."""

    stored_rows = {str(row["parent_id"]): row for row in stored["per_parent"]}
    replayed_rows = {str(row["parent_id"]): row for row in replayed["per_parent"]}
    require(stored_rows.keys() == replayed_rows.keys(), "selector replay parent drift")
    comparisons = 0
    maximum_abs_difference = 0.0
    for parent_id in sorted(stored_rows):
        before = stored_rows[parent_id]
        after = replayed_rows[parent_id]
        require(before["domain"] == after["domain"], "selector replay domain drift")
        pairs = [
            (
                float(before["selected_coverage_fraction"]),
                float(after["selected_coverage_fraction"]),
                "selected_coverage_fraction",
            )
        ]
        for lane in ("base", "expert", "selected"):
            for metric in ("mae_m", "bad_gt_0_10_m_fraction"):
                pairs.append(
                    (
                        float(before[lane][metric]),
                        float(after[lane][metric]),
                        f"{lane}.{metric}",
                    )
                )
        for left, right, metric in pairs:
            _assert_close(left, right, f"{parent_id}:{metric}", tolerance)
            maximum_abs_difference = max(maximum_abs_difference, abs(left - right))
            comparisons += 1
    return {
        "status": "FROZEN_SELECTOR_METRIC_REPLAY_PASS",
        "parent_count": len(stored_rows),
        "comparison_count": comparisons,
        "tolerance": tolerance,
        "maximum_abs_difference": maximum_abs_difference,
    }


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output = args.output.resolve()
    require(not output.exists(), "factor-wise canary output collision")
    output.parent.mkdir(parents=True, exist_ok=False)
    selector_result_path = args.selector_result.resolve()
    selector_checkpoint_path = args.selector_checkpoint.resolve()
    boundary_result_path = args.boundary_result.resolve()
    selector_result = load_json(selector_result_path, "selector fit result")
    boundary_result = load_json(boundary_result_path, "R21 boundary result")
    require(
        selector_result.get("schema")
        == "blindassist_ag_st_no_regret_selector_fit_result_v1",
        "selector fit result schema drift",
    )
    require(
        selector_result.get("status")
        == "NO_REGRET_SELECTOR_FIT_COMPLETE_EXTERNAL_EVALUATION_UNREAD",
        "selector fit result role drift",
    )
    claim_boundary = selector_result.get("claim_boundary", {})
    require(
        claim_boundary.get("external_bonn_evaluation_read") is False
        and claim_boundary.get("external_tum_evaluation_read") is False,
        "selector fit result opened external evaluation",
    )
    checkpoint_receipt = selector_result.get("checkpoint", {})
    require(selector_checkpoint_path.is_file(), "selector checkpoint missing")
    require(
        sha256_file(selector_checkpoint_path) == checkpoint_receipt.get("sha256"),
        "selector checkpoint hash drift",
    )
    require(torch.cuda.is_available(), "factor-wise canary requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (
        selector_payload,
        selector,
        threshold,
        expert_path,
        _expert_payload,
        expert,
    ) = load_selector(selector_checkpoint_path, device)
    require(
        selector_payload.get("threshold_calibration_decision")
        == "NONTRIVIAL_NO_REGRET_THRESHOLD_FROZEN",
        "selector checkpoint lacks frozen nontrivial threshold",
    )
    require(
        abs(threshold - float(selector_result["threshold_calibration"]["threshold"]))
        <= 1e-12,
        "selector result/checkpoint threshold drift",
    )

    stage0a_results = [path.resolve() for path in args.stage0a_result]
    label_dirs = [path.resolve() for path in args.label_dir]
    require(len(stage0a_results) == len(label_dirs) == 3, "expected three ARKit batches")
    descriptors, source_batches = build_frame_descriptor_batches(
        stage0a_results, label_dirs
    )
    expert_architecture = selector_payload["expert"]["architecture"]
    arkit_frames, arkit_extraction = extract_depthart_features(
        descriptors,
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(selector_payload["seed"]),
        feature_profile=expert_architecture["feature_profile"],
    )
    bonn_frames, bonn_extraction = extract_bonn_anchor_frames(
        args.cohort_manifest.resolve(),
        args.dataset_root.resolve(),
        args.archive.resolve(),
        args.catalog.resolve(),
        args.receipt.resolve(),
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(selector_payload["seed"]),
        cohort_role="fit",
    )
    tum_frames, tum_extraction = extract_tum_anchor_frames(
        args.tum_cohort_manifest.resolve(),
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(selector_payload["seed"]),
        cohort_role="fit",
    )
    parent_domains = {
        **{
            frame.descriptor.parent_id: "ARKITSCENES"
            for frame in arkit_frames
        },
        **{
            frame.descriptor.parent_id: "BONN_RGBD_DYNAMIC"
            for frame in bonn_frames
        },
        **{frame.descriptor.parent_id: "TUM_RGBD" for frame in tum_frames},
    }
    all_frames = [*arkit_frames, *bonn_frames, *tum_frames]
    calibration_parents = {
        str(parent) for parent in selector_payload["split"]["selector_calibration_parents"]
    }
    fit_parents = {
        str(parent) for parent in selector_payload["split"]["selector_fit_parents"]
    }
    require(not (calibration_parents & fit_parents), "selector split overlap")
    require(
        calibration_parents <= set(parent_domains),
        "selector calibration parent source missing",
    )
    calibration_frames = [
        frame
        for frame in all_frames
        if frame.descriptor.parent_id in calibration_parents
    ]
    require(calibration_frames, "selector calibration frames empty")
    observations = collect_selector_observations(
        selector,
        expert,
        calibration_frames,
        parent_domains,
        device,
    )
    summary = summarize_selector_observations(observations, threshold)
    replay = validate_frozen_metric_replay(
        selector_result["threshold_calibration"]["selected_summary"], summary
    )
    oracle_gate = evaluate_lane_gate(summary, "oracle")
    selector_gate = evaluate_lane_gate(summary, "selected")
    status, successor = derive_terminal(oracle_gate, selector_gate)

    require(
        boundary_result.get("schema")
        == "blindassist.assistive_geometry.ag_st_r21_target_mass_normalized_angular_boundary_result.v1",
        "R21 boundary result schema drift",
    )
    result = {
        "schema": "blindassist.ag.factorwise_no_regret_oracle_parent_gate_canary.v1",
        "status": status,
        "mode": "PROJECT_CONSUMED_DEVELOPMENT",
        "question": (
            "Does the frozen factor-wise depth selector have nonzero safe correction "
            "coverage under per-parent MAE and >0.10m error no-regret gates, and is "
            "there perfect signed-advantage oracle headroom?"
        ),
        "frozen_selector": {
            "fit_result_path": str(selector_result_path),
            "fit_result_sha256": sha256_file(selector_result_path),
            "checkpoint_path": str(selector_checkpoint_path),
            "checkpoint_sha256": sha256_file(selector_checkpoint_path),
            "expert_checkpoint_path": str(expert_path),
            "expert_checkpoint_sha256": sha256_file(expert_path),
            "threshold": threshold,
            "threshold_selected_in_this_canary": False,
            "training_performed_in_this_canary": False,
        },
        "source_roles": {
            "arkit_source_batches": source_batches,
            "arkit_feature_extraction": arkit_extraction,
            "bonn_fit_anchors": bonn_extraction,
            "tum_fit_anchors": tum_extraction,
            "selector_fit_parents_read": False,
            "selector_calibration_parents": sorted(calibration_parents),
            "selector_calibration_parent_count": len(calibration_parents),
            "selector_calibration_frame_count": len(calibration_frames),
            "external_bonn_evaluation_read": False,
            "external_tum_evaluation_read": False,
            "eth3d_confirmation_read": False,
        },
        "frozen_metric_replay": replay,
        "comparison": summary,
        "gates": {
            "oracle": oracle_gate,
            "selector": selector_gate,
        },
        "factor_isolation": {
            "depth_selector_evaluated": True,
            "boundary_result_path": str(boundary_result_path),
            "boundary_result_sha256": sha256_file(boundary_result_path),
            "boundary_status": boundary_result["status"],
            "boundary_checkpoint_sha256": boundary_result["training"][
                "checkpoint_sha256"
            ],
            "boundary_external_icl_f1_within_2px": boundary_result[
                "external_confirmation"
            ]["icl_exact"]["f1_within_2px"],
            "boundary_external_bonn_f1_within_4px": boundary_result[
                "external_confirmation"
            ]["bonn_evaluation8"]["f1_within_4px"],
            "boundary_retrained_or_rescored": False,
            "support_modified": False,
            "unknown_policy_modified": False,
            "reducer_called": False,
        },
        "decision": {
            "terminal": status,
            "successor": successor,
            "selector_development_gate_pass": selector_gate["pass"],
            "oracle_safe_coverage_supported": oracle_gate["pass"],
            "selector_promoted_to_confirmation": False,
            "reason": (
                "The canary is limited to already-consumed Development calibration "
                "parents. A PASS freezes the selector for a genuinely fresh, parent-"
                "disjoint evaluation; it is not cross-sensor or task evidence."
            ),
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "total_seconds": time.perf_counter() - started,
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "development_only": True,
            "complete_truth_required": False,
            "fresh_evaluation_claim_authorized": False,
            "cross_sensor_claim_authorized": False,
            "task_superiority_claim_authorized": False,
            "default_app_changed": False,
            "deployment_product_safety_claim_authorized": False,
        },
    }
    write_json_exclusive(output, result)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(output),
                "oracle_gate": oracle_gate,
                "selector_gate": selector_gate,
                "successor": successor,
                "total_seconds": result["execution"]["total_seconds"],
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-result", type=Path, default=DEFAULT_SELECTOR_RESULT)
    parser.add_argument(
        "--selector-checkpoint", type=Path, default=DEFAULT_SELECTOR_CHECKPOINT
    )
    parser.add_argument("--boundary-result", type=Path, default=DEFAULT_BOUNDARY_RESULT)
    parser.add_argument(
        "--stage0a-result",
        type=Path,
        action="append",
        default=list(DEFAULT_STAGE0A_RESULTS),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        action="append",
        default=list(DEFAULT_LABEL_DIRS),
    )
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument("--tum-cohort-manifest", type=Path, default=DEFAULT_TUM_COHORT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_BONN_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_BONN_CATALOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_BONN_RECEIPT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
