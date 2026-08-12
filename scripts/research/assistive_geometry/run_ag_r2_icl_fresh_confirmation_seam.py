#!/usr/bin/env python3
"""Run the frozen SuperTeacher student recipe through AG on unused ICL data.

This is a one-shot confirmation.  ICL labels are opened only after the factor
model, metric-depth student, height estimator, uncertainty calibration, and
reducer profile have been frozen by the consumed r10 recipe.  Nothing in this
script fits, selects, or recalibrates a model from ICL outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from calibrate_ag_r2_f1_attempt20_frame_geometry_uncertainty import (  # noqa: E402
    ATTEMPT18_RESULT,
    EXPECTED_ATTEMPT18_RESULT_SHA256,
)
import run_ag_r2_hybrid_factor_student_to_ag_seam as hybrid  # noqa: E402


DEFAULT_FROZEN_RECIPE_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-hybrid-factor-student-to-ag-seam-r10/result.json"
)
EXPECTED_FROZEN_RECIPE_RESULT_SHA256 = (
    "5077213EA3B5B0CF0186E755BF1D0FDDF8C25DBB65D415E3A05263A6780A9720"
)
DEFAULT_FRESH_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-icl-fresh-confirmation-labels-r0/result.json"
)
EXPECTED_FRESH_LABEL_RESULT_SHA256 = (
    "E3A8F7FF73BD30AD9701D090F5D8959F4C93F45BB70944C85BA01D0AE3CAFBB1"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-icl-fresh-confirmation-seam-r0"
)
EXPECTED_PARENT_ID = "icl_living_room_kt1"
EXPECTED_FRAME_COUNT = 12


def frozen_training_parent_ids(
    factor_result: dict[str, Any],
    metric_result: dict[str, Any],
) -> set[str]:
    parents = set(factor_result["fit_parents"])
    parents.update(factor_result["internal_validation_parents"])
    for role in metric_result["roles"].values():
        parents.update(str(value) for value in role["parents"])
    return parents


def state_and_reason_counts(reduced: dict[str, Any]) -> tuple[Counter[str], Counter[str]]:
    states: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for band in reduced["bands"]:
        for cell in band["cells"]:
            states[str(cell["state"])] += 1
            reasons.update(str(value) for value in cell["reason_codes"])
    return states, reasons


def load_metric_depth_student(
    result: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Path]:
    hybrid.require(result["passed"], "metric depth student prerequisite failed")
    checkpoint = Path(result["checkpoint"]["path"])
    hybrid.require(
        hybrid.sha256_file(checkpoint)
        == hybrid.EXPECTED_METRIC_DEPTH_STUDENT_CHECKPOINT_SHA256,
        "metric depth student checkpoint drift",
    )
    model = hybrid.MetricDepthStudentHead(
        hidden=int(result["architecture"]["hidden_channels"]),
        global_hidden=int(result["architecture"]["global_hidden_channels"]),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    hybrid.require(
        torch.cuda.is_available() and str(args.device).startswith("cuda"),
        "CUDA required",
    )
    hybrid.require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    hybrid.require(
        hybrid.sha256_file(args.frozen_recipe_result)
        == EXPECTED_FROZEN_RECIPE_RESULT_SHA256,
        "frozen r10 recipe result drift",
    )
    hybrid.require(
        hybrid.sha256_file(args.fresh_label_result)
        == EXPECTED_FRESH_LABEL_RESULT_SHA256,
        "ICL fresh label result drift",
    )
    hybrid.require(
        hybrid.sha256_file(args.baseline_result) == hybrid.EXPECTED_BASELINE_RESULT_SHA256,
        "baseline result drift",
    )
    hybrid.require(
        hybrid.sha256_file(args.depthart_checkpoint) == hybrid.EXPECTED_DEPTHART_SHA256,
        "DepthART checkpoint drift",
    )
    hybrid.require(
        hybrid.sha256_file(args.metric_depth_student_result)
        == hybrid.EXPECTED_METRIC_DEPTH_STUDENT_RESULT_SHA256,
        "metric depth student result drift",
    )
    hybrid.require(
        hybrid.sha256_file(ATTEMPT18_RESULT) == EXPECTED_ATTEMPT18_RESULT_SHA256,
        "factor checkpoint result drift",
    )

    frozen = json.loads(args.frozen_recipe_result.read_text(encoding="utf-8"))
    fresh = json.loads(args.fresh_label_result.read_text(encoding="utf-8"))
    metric_result = json.loads(
        args.metric_depth_student_result.read_text(encoding="utf-8")
    )
    factor_result = json.loads(ATTEMPT18_RESULT.read_text(encoding="utf-8"))
    hybrid.require(
        frozen["passed"]
        and frozen["status"]
        == "AG_R2_HYBRID_FACTOR_STUDENT_TO_AG_SEAM_PASS_CONSUMED_DIAGNOSTIC",
        "frozen r10 recipe is not a passed prerequisite",
    )
    hybrid.require(all(frozen["gates"].values()), "frozen r10 recipe gates drift")
    hybrid.require(
        fresh["passed"]
        and fresh["status"] == "AG_R2_ICL_FRESH_CONFIRMATION_LABELS_PASS"
        and fresh["frame_count"] == EXPECTED_FRAME_COUNT,
        "ICL fresh label frontdoor drift",
    )
    hybrid.require(
        fresh["source"]["checkpoint_unseen_by_current_student"] is True,
        "ICL source no longer declared checkpoint-unseen",
    )
    hybrid.require(
        fresh["decision"][
            "current_student_or_reducer_output_opened_during_materialization"
        ]
        is False,
        "ICL labels were materialized from current model or reducer output",
    )

    frozen_parents = frozen_training_parent_ids(factor_result, metric_result)
    hybrid.require(EXPECTED_PARENT_ID not in frozen_parents, "ICL parent leaked into training")
    rows = sorted(
        [{**dict(row), "role": "FRESH_CONFIRMATION"} for row in fresh["frames"]],
        key=lambda row: str(row["sample_id"]),
    )
    hybrid.require(
        len(rows) == EXPECTED_FRAME_COUNT
        and {str(row["parent_id"]) for row in rows} == {EXPECTED_PARENT_ID}
        and len({str(row["sample_id"]) for row in rows}) == EXPECTED_FRAME_COUNT,
        "ICL confirmation roster drift",
    )

    started = time.perf_counter()
    device = torch.device(args.device)
    samples, feature_receipt = hybrid.extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))[
        "baseline_parameters"
    ]
    factor_model, factor_checkpoint = hybrid.load_factor_model(baseline, device)
    raw_by_sample = hybrid.raw_model_outputs(samples, factor_model, device)
    del factor_model
    metric_model, metric_checkpoint = load_metric_depth_student(metric_result, device)
    hybrid.attach_metric_depth_student_outputs(
        samples,
        raw_by_sample,
        metric_model,
        device,
    )
    del metric_model
    torch.cuda.empty_cache()

    # These three objects are copied verbatim from the already-consumed r10
    # recipe.  They are intentionally never selected or recalibrated on ICL.
    height_estimator = dict(frozen["height_estimator"])
    calibration = dict(frozen["uncertainty_calibration"])
    reducer_profile = dict(frozen["reducer_profile"]["profile"])
    identity = dict(frozen["factor_identity"])
    hybrid.require(
        factor_checkpoint["sha256"] == identity["model_checkpoint_sha256"],
        "factor checkpoint does not match frozen identity",
    )
    hybrid.require(
        hybrid.sha256_file(metric_checkpoint)
        == identity["metric_depth_student_checkpoint_sha256"],
        "metric student checkpoint does not match frozen identity",
    )
    hybrid.require(
        calibration["fresh_canary_used"] is False
        and calibration["task_or_reducer_output_used"] is False,
        "frozen calibration firewall drift",
    )

    session_profiles = hybrid.session_height_profiles(samples)
    hybrid.require(
        set(session_profiles) == {EXPECTED_PARENT_ID},
        "one-time source-native session height unavailable",
    )
    hybrid_outputs: dict[str, dict[str, torch.Tensor]] = {}
    geometry_rows: dict[str, dict[str, Any]] = {}
    for sample in samples:
        output, receipt = hybrid.hybrid_output(
            sample,
            raw_by_sample[sample.sample_id],
            height_estimator,
            calibration,
            session_profiles.get(sample.parent_id),
        )
        hybrid_outputs[sample.sample_id] = output
        geometry_rows[sample.sample_id] = receipt
    metrics = hybrid.role_metrics(samples, hybrid_outputs)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    factors = hybrid.serialize_factors(
        samples,
        [hybrid_outputs[sample.sample_id] for sample in samples],
        args.output_dir,
        identity,
    )
    factor_by_sample = {row["sample_id"]: row for row in factors}
    adapter_dir = args.output_dir / "adapter_frames"
    reducer_dir = args.output_dir / "reducer_outputs"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    reducer_dir.mkdir(parents=True, exist_ok=False)
    calibration_payload = hybrid.calibration_receipt(calibration)
    seam_rows: list[dict[str, Any]] = []
    aggregate_states: Counter[str] = Counter()
    aggregate_reasons: Counter[str] = Counter()
    for sample in samples:
        factor = factor_by_sample[sample.sample_id]
        prediction = hybrid.load_prediction(Path(factor["path"]))
        geometry = hybrid.geometry_receipt(sample, prediction)
        adapted = hybrid.adapt_factor_tensor(
            {
                "prediction": prediction,
                "geometry_receipt": geometry,
                "calibration_receipt": calibration_payload,
            }
        )
        hybrid.require(
            adapted["schema"] == hybrid.ADAPTER_OUTPUT_SCHEMA,
            "adapter output schema drift",
        )
        reduced_first = hybrid.reduce_frame(adapted, reducer_profile)
        reduced_second = hybrid.reduce_frame(
            json.loads(json.dumps(adapted)),
            json.loads(json.dumps(reducer_profile)),
        )
        hybrid.require(
            reduced_first["schema"] == hybrid.REDUCER_OUTPUT_SCHEMA,
            "reducer output schema drift",
        )
        deterministic = (
            hybrid.reducer_sha256(reduced_first)
            == hybrid.reducer_sha256(reduced_second)
        )
        state_counts, reason_counts = state_and_reason_counts(reduced_first)
        aggregate_states.update(state_counts)
        aggregate_reasons.update(reason_counts)
        adapter_path = adapter_dir / f"{sample.sample_id}.json"
        reducer_path = reducer_dir / f"{sample.sample_id}.json"
        hybrid.write_json(adapter_path, adapted)
        hybrid.write_json(reducer_path, reduced_first)
        seam_rows.append(
            {
                **geometry_rows[sample.sample_id],
                "factor_tensor": factor,
                "adapter_frame": {
                    "path": str(adapter_path.resolve()),
                    "sha256": hybrid.sha256_file(adapter_path),
                    "canonical_sha256": hybrid.adapter_sha256(adapted),
                },
                "reducer_output": {
                    "path": str(reducer_path.resolve()),
                    "sha256": hybrid.sha256_file(reducer_path),
                    "canonical_sha256": hybrid.reducer_sha256(reduced_first),
                },
                "adapter_depth_valid": bool(adapted["depth_scale"]["valid"]),
                "adapter_support_valid": bool(adapted["support"]["valid"]),
                "adapter_boundary_valid": bool(adapted["boundary"]["valid"]),
                "adapter_boundary_coverage": float(adapted["boundary"]["coverage"]),
                "adapter_obstacle_count": len(adapted["boundary"]["obstacles"]),
                "state_counts": dict(sorted(state_counts.items())),
                "reason_counts": dict(sorted(reason_counts.items())),
                "deterministic_repeat_equal": deterministic,
            }
        )

    structurally_valid = [
        row
        for row in seam_rows
        if row["adapter_depth_valid"]
        and row["adapter_support_valid"]
        and row["adapter_boundary_valid"]
        and row["adapter_boundary_coverage"] > 0.0
    ]
    state_set = set(aggregate_states)
    gates = {
        "ICLSEAM_C01_EXACT_FROZEN_RECIPE_AND_SOURCE_RECEIPTS": bool(
            len(samples) == EXPECTED_FRAME_COUNT
            and factor_checkpoint["sha256"]
            == frozen["factor_checkpoint"]["sha256"]
        ),
        "ICLSEAM_C02_CHECKPOINTS_EXCLUDE_ICL_PARENT": bool(
            EXPECTED_PARENT_ID not in frozen_parents
        ),
        "ICLSEAM_C03_NO_ICL_FIT_SELECTION_OR_RECALIBRATION": bool(
            calibration == frozen["uncertainty_calibration"]
            and height_estimator == frozen["height_estimator"]
            and reducer_profile == frozen["reducer_profile"]["profile"]
        ),
        "ICLSEAM_C04_FACTOR_TENSORS_ROUNDTRIP_12_OF_12": bool(
            len(factors) == EXPECTED_FRAME_COUNT
            and all(Path(row["path"]).is_file() for row in factors)
        ),
        "ICLSEAM_C05_ADAPTER_HAS_VALID_EXTERNAL_FRAMES": bool(
            len(structurally_valid) >= 6
        ),
        "ICLSEAM_C06_REDUCER_DETERMINISTIC_12_OF_12": bool(
            all(row["deterministic_repeat_equal"] for row in seam_rows)
        ),
        "ICLSEAM_C07_NONTRIVIAL_FAIL_CLOSED_GEOMETRY_STATE": bool(
            state_set - {"UNKNOWN"} and "UNKNOWN" in state_set
        ),
        "ICLSEAM_C08_FACTOR_ONLY_AND_LABEL_FIREWALL": bool(
            identity["learned_final_task_head"] is False
            and identity["task_outcome_used"] is False
            and fresh["decision"][
                "current_student_or_reducer_output_opened_during_materialization"
            ]
            is False
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_icl_fresh_confirmation_seam_result_v1",
        "status": "AG_R2_ICL_FRESH_CONFIRMATION_SEAM_PASS"
        if passed
        else "AG_R2_ICL_FRESH_CONFIRMATION_SEAM_FAIL_NO_RETUNING",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_recipe_result": {
            "path": str(args.frozen_recipe_result.resolve()),
            "sha256": EXPECTED_FROZEN_RECIPE_RESULT_SHA256,
        },
        "fresh_label_result": {
            "path": str(args.fresh_label_result.resolve()),
            "sha256": EXPECTED_FRESH_LABEL_RESULT_SHA256,
            "dataset": fresh["source"]["dataset"],
            "synthetic_exact_not_real_world": True,
        },
        "feature_receipt": feature_receipt,
        "factor_checkpoint": factor_checkpoint,
        "metric_depth_student": {
            "result": str(args.metric_depth_student_result.resolve()),
            "result_sha256": hybrid.EXPECTED_METRIC_DEPTH_STUDENT_RESULT_SHA256,
            "checkpoint": str(metric_checkpoint.resolve()),
            "checkpoint_sha256": hybrid.sha256_file(metric_checkpoint),
        },
        "factor_identity": identity,
        "height_estimator": height_estimator,
        "fresh_session_height_profiles": session_profiles,
        "uncertainty_calibration": calibration,
        "metrics": metrics,
        "factor_schema_sha256": hybrid.FACTOR_SCHEMA_SHA256,
        "reducer_profile": frozen["reducer_profile"],
        "aggregate_state_counts": dict(sorted(aggregate_states.items())),
        "aggregate_reason_counts": dict(sorted(aggregate_reasons.items())),
        "valid_adapter_frame_count": len(structurally_valid),
        "gates": gates,
        "frames": seam_rows,
        "decision": {
            "unused_source_factor_student_to_ag_mechanics_complete": passed,
            "ag_research_pipeline_mechanics_landed": passed,
            "icl_outcomes_used_for_fit_selection_or_recalibration": False,
            "real_world_fresh_generalization_claim": False,
            "mobile_or_htp_claim": False,
            "product_or_safety_claim": False,
            "default_app_changed": False,
            "next_action_if_pass": "Freeze the research pipeline result; a real unused RGB-D-plus-pose/IMU source remains the promotion gate.",
            "next_action_if_fail": "Freeze this one-shot negative result and diagnose without retuning on ICL outcomes.",
            "claim_ceiling": "Checkpoint-unseen synthetic-exact external factor-student mechanics through the unchanged deterministic AG seam, combined with consumed real-source evidence; not fresh real-world, mobile, product, or safety proof.",
        },
    }
    hybrid.write_json(args.output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-recipe-result",
        type=Path,
        default=DEFAULT_FROZEN_RECIPE_RESULT,
    )
    parser.add_argument(
        "--fresh-label-result",
        type=Path,
        default=DEFAULT_FRESH_LABEL_RESULT,
    )
    parser.add_argument(
        "--baseline-result", type=Path, default=hybrid.DEFAULT_BASELINE_RESULT
    )
    parser.add_argument(
        "--depthart-source", type=Path, default=hybrid.DEFAULT_DEPTHART_SOURCE
    )
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=hybrid.DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument(
        "--depthart-extension", type=Path, default=hybrid.DEFAULT_DEPTHART_EXTENSION
    )
    parser.add_argument(
        "--metric-depth-student-result",
        type=Path,
        default=hybrid.DEFAULT_METRIC_DEPTH_STUDENT_RESULT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in (
        "frozen_recipe_result",
        "fresh_label_result",
        "baseline_result",
        "depthart_source",
        "depthart_checkpoint",
        "depthart_extension",
        "metric_depth_student_result",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "valid_adapter_frame_count": result["valid_adapter_frame_count"],
                "aggregate_state_counts": result["aggregate_state_counts"],
                "aggregate_reason_counts": result["aggregate_reason_counts"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
