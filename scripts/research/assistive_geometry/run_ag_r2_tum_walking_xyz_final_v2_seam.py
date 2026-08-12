#!/usr/bin/env python3
"""Open walking_xyz once with the frozen multi-source V2 AG recipe.

This runner does not train, select, fit, recalibrate, or compare against a
baseline.  It consumes RGB plus the frozen camera/session geometry contract,
predicts factor tensors, separates global metric-scale uncertainty from local
depth-shape uncertainty, adapts those factors, and invokes the unchanged
deterministic body-swept reducer twice to verify repeatability.

The source is checkpoint-unseen for this recipe but historically consumed by
other repository routes.  Either PASS or FAIL is terminal for this R0 recipe;
the source outcome must never be used to retune the candidate and reopen R0.
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

import run_ag_r2_hybrid_factor_student_to_ag_seam as hybrid  # noqa: E402
import run_ag_r2_multisource_v2_consumed_seam as consumed  # noqa: E402
from factor_tensor_adapter_v2 import (  # noqa: E402
    FACTOR_SCHEMA_SHA256,
    adapt_factor_tensor,
    canonical_sha256 as adapter_sha256,
)
from calibrate_ag_r2_metric_scale_residual_bank import (  # noqa: E402
    load_residual_bank,
    pooled_uncertainty_feature,
    predict_scale_sigma,
)


FINAL_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments"
    / "ag-r2-tum-walking-xyz-final-confirmation-labels-r0/result.json"
)
EXPECTED_FINAL_LABEL_RESULT_SHA256 = (
    "D8083B567CF227AB83423A14B281B8B9A451DF8D6B29F78DF34A5B4459F10812"
)
FROZEN_V2_CONSUMED_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments"
    / "ag-r2-multisource-v2-consumed-seam-r0/result.json"
)
EXPECTED_FROZEN_V2_CONSUMED_RESULT_SHA256 = (
    "106E64706632D19BF327513DE927A8BA68277F800F66DA3379EB2298EE724528"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments"
    / "ag-r2-tum-walking-xyz-final-ag-seam-r1-recovery"
)
PARENT_ID = "rgbd_dataset_freiburg3_walking_xyz"
ROLE = "FINAL_CHECKPOINT_UNSEEN_REAL_SEAM"
FRAME_COUNT = 12
MIN_STRUCTURALLY_VALID_FRAMES = 6


def verify_exact_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    exact = {
        args.final_label_result: args.expected_final_label_result_sha256,
        args.frozen_v2_consumed_result: EXPECTED_FROZEN_V2_CONSUMED_RESULT_SHA256,
        args.frozen_r10_result: consumed.EXPECTED_FROZEN_R10_SHA256,
        args.metric_student_result: consumed.EXPECTED_METRIC_STUDENT_RESULT_SHA256,
        args.scale_bank_result: consumed.EXPECTED_SCALE_BANK_RESULT_SHA256,
    }
    for path, expected in exact.items():
        hybrid.require(path.is_file(), f"input receipt missing: {path}")
        hybrid.require(hybrid.sha256_file(path) == expected, f"receipt drift: {path}")
    hybrid.require(
        hybrid.sha256_file(args.baseline_result)
        == hybrid.EXPECTED_BASELINE_RESULT_SHA256,
        "baseline parameter receipt drift",
    )
    hybrid.require(
        hybrid.sha256_file(args.depthart_checkpoint) == hybrid.EXPECTED_DEPTHART_SHA256,
        "DepthART checkpoint drift",
    )

    labels = json.loads(args.final_label_result.read_text(encoding="utf-8"))
    frozen_v2 = json.loads(
        args.frozen_v2_consumed_result.read_text(encoding="utf-8")
    )
    hybrid.require(labels["passed"] and all(labels["gates"].values()), "labels failed")
    hybrid.require(
        frozen_v2["passed"]
        and frozen_v2["status"]
        == "AG_R2_MULTISOURCE_V2_CONSUMED_SEAM_PASS_READY_FOR_FRESH_REAL",
        "frozen V2 prerequisite failed",
    )
    hybrid.require(labels["frame_count"] == FRAME_COUNT, "label roster size drift")
    source_is_current_recipe_unseen = bool(
        labels["source"].get("checkpoint_unseen_by_current_frozen_recipe", False)
        or (
            labels["source"].get(
                "checkpoint_unseen_by_current_factor_and_metric_students", False
            )
            and bool(labels.get("extra_current_recipe_receipts"))
        )
    )
    hybrid.require(
        source_is_current_recipe_unseen
        and labels["source"]["globally_unopened_claim"] is False,
        "source role boundary drift",
    )
    current_recipe_role = labels["label_contract"].get("current_recipe_role")
    hybrid.require(
        current_recipe_role == "CHECKPOINT_UNSEEN_AFTER_V2_RECIPE_FREEZE"
        or bool(labels.get("extra_current_recipe_receipts")),
        "current recipe role drift",
    )
    for path in (
        args.frozen_r10_result,
        args.metric_student_result,
        args.scale_bank_result,
        args.frozen_v2_consumed_result,
    ):
        hybrid.require(
            args.parent_id not in path.read_text(encoding="utf-8"),
            f"final parent leaked into frozen recipe: {path}",
        )
    return labels, frozen_v2


def code_receipts() -> dict[str, dict[str, str]]:
    paths = {
        "factor_tensor_adapter_v2": Path(adapt_factor_tensor.__code__.co_filename),
        "deterministic_geometry_reducer": Path(hybrid.reduce_frame.__code__.co_filename),
        "final_runner": Path(__file__),
    }
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": hybrid.sha256_file(path),
        }
        for name, path in paths.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    hybrid.require(
        torch.cuda.is_available() and str(args.device).startswith("cuda"),
        "CUDA required",
    )
    hybrid.require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    labels, frozen_v2 = verify_exact_inputs(args)
    anchor_result: dict[str, Any] | None = None
    anchor_config: dict[str, Any] | None = None
    if args.scale_anchor_result is not None:
        hybrid.require(
            args.expected_scale_anchor_result_sha256 is not None,
            "anchored development run requires an exact calibration receipt",
        )
        hybrid.require(
            hybrid.sha256_file(args.scale_anchor_result)
            == args.expected_scale_anchor_result_sha256,
            "scale anchor calibration drift",
        )
        anchor_result = json.loads(
            args.scale_anchor_result.read_text(encoding="utf-8")
        )
        hybrid.require(
            anchor_result["passed"]
            and all(anchor_result["gates"].values())
            and anchor_result["decision"]["walking_xyz_consumed_for_factor_calibration"],
            "scale anchor calibration invalid",
        )
        anchor_config = dict(anchor_result["config"])
    anchor_is_confirmation = bool(anchor_config is not None and args.confirmation_mode)
    metric_result = json.loads(args.metric_student_result.read_text(encoding="utf-8"))
    bank_result = json.loads(args.scale_bank_result.read_text(encoding="utf-8"))
    frozen_r10 = json.loads(args.frozen_r10_result.read_text(encoding="utf-8"))
    hybrid.require(
        metric_result["passed"] and bank_result["passed"] and frozen_r10["passed"],
        "frozen recipe ingredient failed",
    )
    rows = sorted(
        [{**dict(row), "role": ROLE} for row in labels["frames"]],
        key=lambda row: str(row["sample_id"]),
    )
    hybrid.require(
        len(rows) == FRAME_COUNT
        and len({row["sample_id"] for row in rows}) == FRAME_COUNT
        and {row["parent_id"] for row in rows} == {args.parent_id},
        "final roster drift",
    )

    bank_path = Path(bank_result["bank"]["path"])
    hybrid.require(
        hybrid.sha256_file(bank_path) == consumed.EXPECTED_SCALE_BANK_SHA256,
        "scale residual bank drift",
    )
    bank = load_residual_bank(bank_path)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    samples, feature_receipt = hybrid.extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
        load_targets=False,
    )
    hybrid.require(
        feature_receipt["targets_loaded"] is False
        and all(not sample.targets for sample in samples),
        "fresh supervision targets entered inference memory",
    )
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))[
        "baseline_parameters"
    ]
    factor_model, factor_checkpoint = hybrid.load_factor_model(baseline, device)
    raw_by_sample = hybrid.raw_model_outputs(samples, factor_model, device)
    del factor_model
    metric_model, metric_checkpoint = consumed.load_metric_model(metric_result, device)
    hybrid.attach_metric_depth_student_outputs(
        samples,
        raw_by_sample,
        metric_model,
        device,
    )
    del metric_model
    torch.cuda.empty_cache()

    height_estimator = dict(frozen_r10["height_estimator"])
    calibration = dict(frozen_r10["uncertainty_calibration"])
    calibration["depth_shape_relative_sigma"] = consumed.LOCAL_SHAPE_RELATIVE_SIGMA
    calibration["depth_shape_relative_sigma_rms"] = consumed.LOCAL_SHAPE_RELATIVE_SIGMA
    reducer_profile = dict(frozen_r10["reducer_profile"]["profile"])
    session_profiles = hybrid.session_height_profiles(samples)
    hybrid.require(
        set(session_profiles) == {args.parent_id},
        "frozen one-time session-height anchor unavailable",
    )

    outputs: dict[str, dict[str, torch.Tensor]] = {}
    geometry_rows: dict[str, dict[str, Any]] = {}
    scale_sigmas: dict[str, float] = {}
    scale_diagnostics: dict[str, dict[str, float]] = {}
    for sample in samples:
        raw = raw_by_sample[sample.sample_id]
        pooled = pooled_uncertainty_feature(
            sample.feature[None],
            sample.base_depth_feature[None],
            raw["metric_depth_student_log_depth"],
        )[0]
        bank_scale_sigma, diagnostic = predict_scale_sigma(pooled, bank)
        scale_sigma = (
            float(anchor_config["metric_scale_log_sigma"])
            if anchor_config is not None
            else bank_scale_sigma
        )
        scale_sigmas[sample.sample_id] = scale_sigma
        scale_diagnostics[sample.sample_id] = {
            **diagnostic,
            "unanchored_bank_scale_sigma": bank_scale_sigma,
            "session_anchor_scale_sigma_applied": scale_sigma,
        }
        output, receipt = hybrid.hybrid_output(
            sample,
            raw,
            height_estimator,
            calibration,
            session_profiles.get(sample.parent_id),
            metric_scale_anchor=anchor_config,
        )
        outputs[sample.sample_id] = output
        geometry_rows[sample.sample_id] = receipt

    identity = {
        **dict(frozen_r10["factor_identity"]),
        "model_id": "AG_R2_MULTISOURCE_FACTOR_STUDENT_V2_FROZEN_FINAL_R0",
        "metric_depth_student_checkpoint_sha256": (
            consumed.EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256
        ),
        "metric_scale_uncertainty_source": (
            "CONSUMED_SESSION_HEIGHT_ANCHOR_Q90_SCALE_RESIDUAL"
            if anchor_config is not None
            else "PCA_WHITENED_KNN_RESIDUAL_BANK_WITH_OOD_DISTANCE_INFLATION"
        ),
        "local_shape_uncertainty_source": (
            "PARENT_BALANCED_CONSUMED_FACTOR_RESIDUAL_Q68"
        ),
        "factor_schema_sha256": FACTOR_SCHEMA_SHA256,
        "current_input_parent_used_for_factor_calibration": bool(
            anchor_config is not None and not anchor_is_confirmation
        ),
        "final_parent_fit_or_recalibration": bool(
            anchor_config is not None and not anchor_is_confirmation
        ),
    }
    factors = consumed.serialize_v2(
        samples,
        outputs,
        scale_sigmas,
        identity,
        args.output_dir,
    )
    factor_by_sample = {row["sample_id"]: row for row in factors}
    adapter_dir = args.output_dir / "adapter_frames"
    reducer_dir = args.output_dir / "reducer_outputs"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    reducer_dir.mkdir(parents=True, exist_ok=False)
    calibration_payload = consumed.calibration_receipt_v2(calibration)

    aggregate_states: Counter[str] = Counter()
    aggregate_reasons: Counter[str] = Counter()
    seam_rows: list[dict[str, Any]] = []
    for sample in samples:
        factor = factor_by_sample[sample.sample_id]
        prediction = consumed.load_prediction_v2(Path(factor["path"]))
        geometry = hybrid.geometry_receipt(sample, prediction)
        adapted = adapt_factor_tensor(
            {
                "prediction": prediction,
                "geometry_receipt": geometry,
                "calibration_receipt": calibration_payload,
            }
        )
        reduced_first = hybrid.reduce_frame(adapted, reducer_profile)
        reduced_second = hybrid.reduce_frame(
            json.loads(json.dumps(adapted)),
            json.loads(json.dumps(reducer_profile)),
        )
        deterministic = (
            hybrid.reducer_sha256(reduced_first)
            == hybrid.reducer_sha256(reduced_second)
        )
        state_counts, reason_counts = consumed.state_reason_counts(reduced_first)
        aggregate_states.update(state_counts)
        aggregate_reasons.update(reason_counts)
        adapter_path = adapter_dir / f"{sample.sample_id}.json"
        reducer_path = reducer_dir / f"{sample.sample_id}.json"
        hybrid.write_json(adapter_path, adapted)
        hybrid.write_json(reducer_path, reduced_first)
        seam_rows.append(
            {
                **geometry_rows[sample.sample_id],
                "role": sample.role,
                "metric_scale_relative_sigma": scale_sigmas[sample.sample_id],
                "scale_uncertainty_diagnostic": scale_diagnostics[sample.sample_id],
                "factor_tensor": factor,
                "adapter_frame": {
                    "path": str(adapter_path.resolve()),
                    "sha256": hybrid.sha256_file(adapter_path),
                    "canonical_sha256": adapter_sha256(adapted),
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
    nonunknown_count = sum(
        value
        for state, value in aggregate_states.items()
        if state != "UNKNOWN"
    )
    gates = {
        "FINALV2_C01_EXACT_FROZEN_RECIPE_AND_LABEL_RECEIPTS": True,
        "FINALV2_C02_PARENT_EXCLUDED_FROM_MODEL_CHECKPOINT_FITS": True,
        "FINALV2_C03_TIER_A_DEPTH_TIER_B_GEOMETRY_ROLE_BOUND": bool(
            labels["label_contract"]["metric_depth_tier"] == "A_SOURCE_NATIVE"
            and labels["label_contract"]["support_boundary_tier"]
            == "B_GEOMETRY_ANCHORED_TEACHER"
            and labels["source"]["globally_unopened_claim"] is False
        ),
        "FINALV2_C04_TWELVE_FACTOR_TENSORS_ROUNDTRIP": len(factors) == FRAME_COUNT,
        "FINALV2_C05_AT_LEAST_SIX_STRUCTURALLY_VALID_FRAMES": (
            len(structurally_valid) >= MIN_STRUCTURALLY_VALID_FRAMES
        ),
        "FINALV2_C06_REDUCER_DETERMINISTIC_12_OF_12": all(
            row["deterministic_repeat_equal"] for row in seam_rows
        ),
        "FINALV2_C07_AT_LEAST_ONE_OBSERVED_DECISION": nonunknown_count > 0,
        "FINALV2_C08_FAIL_CLOSED_UNKNOWN_PRESERVED": aggregate_states["UNKNOWN"] > 0,
        "FINALV2_C09_FACTOR_ONLY_NO_TASK_OR_REDUCER_OUTCOME_TARGET": bool(
            identity["learned_final_task_head"] is False
            and identity["task_outcome_used"] is False
            and feature_receipt["targets_loaded"] is False
            and (
                anchor_config is None
                or anchor_config["task_or_reducer_output_used"] is False
            )
        ),
        "FINALV2_C10_UNCHANGED_DETERMINISTIC_REDUCER_ENTRYPOINT": bool(
            Path(hybrid.reduce_frame.__code__.co_filename).name
            == "geometry_r2_reducer.py"
        ),
        "FINALV2_C11_ANCHOR_MODE_EXACT_OR_DISABLED": bool(
            anchor_config is None
            or (
                anchor_result is not None
                and anchor_config["task_or_reducer_output_used"] is False
            )
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_tum_walking_xyz_final_v2_seam_result_v1",
        "status": (
            (
                "AG_R2_SESSION_ANCHORED_CONSUMED_SEAM_PASS_READY_FOR_NEW_PARENT"
                if passed
                else "AG_R2_SESSION_ANCHORED_CONSUMED_SEAM_FAIL"
            )
            if anchor_config is not None and not anchor_is_confirmation
            else (
                "AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS"
                if passed
                else "AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_FROZEN_FAIL"
            )
        ),
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "receipts": {
            "final_label_result_sha256": args.expected_final_label_result_sha256,
            "frozen_v2_consumed_result_sha256": (
                EXPECTED_FROZEN_V2_CONSUMED_RESULT_SHA256
            ),
            "frozen_r10_result_sha256": consumed.EXPECTED_FROZEN_R10_SHA256,
            "metric_student_result_sha256": (
                consumed.EXPECTED_METRIC_STUDENT_RESULT_SHA256
            ),
            "metric_student_checkpoint_sha256": (
                consumed.EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256
            ),
            "scale_bank_result_sha256": consumed.EXPECTED_SCALE_BANK_RESULT_SHA256,
            "scale_bank_sha256": consumed.EXPECTED_SCALE_BANK_SHA256,
            "scale_anchor_result_sha256": (
                args.expected_scale_anchor_result_sha256
                if anchor_config is not None
                else None
            ),
        },
        "code_receipts": code_receipts(),
        "feature_receipt": feature_receipt,
        "factor_checkpoint": factor_checkpoint,
        "metric_checkpoint": {
            "path": str(metric_checkpoint.resolve()),
            "sha256": hybrid.sha256_file(metric_checkpoint),
        },
        "factor_identity": identity,
        "session_height_anchor": {
            "profiles": session_profiles,
            "semantics": "ONE_TIME_SOURCE_NATIVE_SESSION_GEOMETRY_FACTOR_NOT_TASK_OUTCOME",
        },
        "uncertainty": {
            "local_shape_relative_sigma": consumed.LOCAL_SHAPE_RELATIVE_SIGMA,
            "metric_scale_relative_sigma_by_frame": scale_sigmas,
        },
        "aggregate_state_counts": dict(sorted(aggregate_states.items())),
        "aggregate_reason_counts": dict(sorted(aggregate_reasons.items())),
        "valid_adapter_frame_count": len(structurally_valid),
        "gates": gates,
        "frames": seam_rows,
        "decision": {
            "execution_version": (
                "R2_CONSUMED_SESSION_SCALE_ANCHOR_DEVELOPMENT"
                if anchor_config is not None and not anchor_is_confirmation
                else (
                    "R3_NEW_PARENT_SESSION_SCALE_ANCHOR_CONFIRMATION"
                    if anchor_is_confirmation
                    else "R1_RECOVERY_AFTER_R0_PRE_METRIC_JSON_SERIALIZATION_FAILURE"
                )
            ),
            "recipe_reopened_or_retuned": bool(
                anchor_config is not None and not anchor_is_confirmation
            ),
            "r0_reopened": False,
            "baseline_comparison_gate": False,
            "mechanical_superteacher_factor_to_ag_landing": (
                passed and (anchor_config is None or anchor_is_confirmation)
            ),
            "consumed_development_ready_for_new_parent": (
                passed and anchor_config is not None and not anchor_is_confirmation
            ),
            "terminal_for_r0_regardless_of_outcome": bool(
                anchor_config is None or anchor_is_confirmation
            ),
            "mobile_or_product_claim": False,
        },
        "claim_ceiling": (
            "Consumed one-parent factor-development seam for a session-height scale anchor; requires a different parent before any confirmation claim."
            if anchor_config is not None and not anchor_is_confirmation
            else "Research-pipeline mechanics on one current-recipe-checkpoint-unseen real TUM parent with a frozen consumed scale-anchor algorithm and source-native session geometry factor; not global freshness, independent cross-sensor generalization, navigation utility, deployment, product, or safety proof."
        ),
        "frozen_v2_prerequisite_decision": frozen_v2["decision"],
    }
    # Validate the complete receipt before opening the destination file.  The
    # R0 operational attempt reached this point with a raw Path and therefore
    # left only a truncated, scientifically unread result.json.
    json.dumps(result, ensure_ascii=False, sort_keys=True)
    hybrid.write_json(args.output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-label-result", type=Path, default=FINAL_LABEL_RESULT)
    parser.add_argument(
        "--expected-final-label-result-sha256",
        default=EXPECTED_FINAL_LABEL_RESULT_SHA256,
    )
    parser.add_argument("--parent-id", default=PARENT_ID)
    parser.add_argument(
        "--frozen-v2-consumed-result",
        type=Path,
        default=FROZEN_V2_CONSUMED_RESULT,
    )
    parser.add_argument(
        "--frozen-r10-result", type=Path, default=consumed.FROZEN_R10_RESULT
    )
    parser.add_argument(
        "--metric-student-result", type=Path, default=consumed.METRIC_STUDENT_RESULT
    )
    parser.add_argument(
        "--scale-bank-result", type=Path, default=consumed.SCALE_BANK_RESULT
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale-anchor-result", type=Path)
    parser.add_argument("--expected-scale-anchor-result-sha256")
    parser.add_argument("--confirmation-mode", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in (
        "final_label_result",
        "frozen_v2_consumed_result",
        "frozen_r10_result",
        "metric_student_result",
        "scale_bank_result",
        "baseline_result",
        "depthart_source",
        "depthart_checkpoint",
        "depthart_extension",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.scale_anchor_result is not None:
        args.scale_anchor_result = args.scale_anchor_result.resolve()
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
