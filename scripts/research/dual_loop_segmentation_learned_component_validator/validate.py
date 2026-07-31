"""Independently recompute the learned validator R0 evidence and terminal."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import (
    ARM_IDS,
    CANDIDATE_ID,
    LEARNED_ARM_ID,
    PROTOCOL_ID,
    RAW_ARM_ID,
    aggregate_arm_report,
    build_component_table,
    build_frame_contexts,
    build_frame_metric_rows,
    build_reference_masks,
    gate_checks,
    load_bound_inputs,
    masks_from_component_decisions,
    near_miss_eligible,
    normalized_gate_margins,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    utility_values,
    validate_static_config,
    validate_table_contract,
    verify_output_scope,
    write_json,
)


VALIDATION_SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_learned_component_validator_r0."
    "validation.v1"
)


def _assert_equal(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise ValueError(f"validation mismatch: {label}")


def _canonical_model_bytes(
    fold: Mapping[str, Any],
    feature_names: Sequence[str],
) -> int:
    model = fold["model"]
    payload = {
        "feature_names": list(feature_names),
        "mean": model["mean"],
        "scale": model["scale"],
        "coefficients": model["coefficients"],
        "intercept": model["intercept"],
        "threshold": fold["selected_threshold"],
    }
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _memory_bound(
    config: Mapping[str, Any],
    maximum_components: int,
) -> int:
    height, width = (int(value) for value in config["analysis_shape"])
    class_count = len(config["candidate_classes"])
    history_max = int(
        config["feature_contract"]["causal_history_max_observations"]
    )
    feature_count = len(config["feature_contract"]["feature_names"])
    return int(
        class_count * history_max * height * width
        + class_count * height * width * np.dtype(np.uint16).itemsize
        + height * width * np.dtype(np.uint8).itemsize
        + maximum_components
        * feature_count
        * np.dtype(np.float64).itemsize
        * 2
        + maximum_components
        * (
            np.dtype(np.float64).itemsize
            + np.dtype(np.bool_).itemsize
        )
        + class_count
        * (maximum_components + 1)
        * np.dtype(np.float64).itemsize
    )


def run_validation(
    *,
    repo_root: Path,
    config_path: Path,
    prepared_root: Path,
    evaluation_root: Path,
    benchmark_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    prepared_root = prepared_root.resolve()
    evaluation_root = evaluation_root.resolve()
    benchmark_root = benchmark_root.resolve()
    output_path = output_path.resolve()
    verify_output_scope(repo_root, output_path)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    checks: list[dict[str, Any]] = []

    def checked(check_id: str, detail: Any) -> None:
        checks.append({"check_id": check_id, "passed": True, "detail": detail})

    config = read_json(config_path)
    validate_static_config(config)
    checked("STATIC_CONFIG", "VALID")

    prepare_receipt_path = prepared_root / "prepare_receipt.json"
    table_path = prepared_root / "component_table.jsonl"
    prepare_receipt = read_json(prepare_receipt_path)
    table_rows = read_jsonl(table_path)
    _assert_equal(prepare_receipt["status"], "COMPLETE", "prepare status")
    _assert_equal(
        sha256_file(table_path),
        prepare_receipt["output_files"]["component_table.jsonl"]["sha256"],
        "prepared table SHA",
    )
    _assert_equal(
        len(table_rows),
        int(
            prepare_receipt["output_files"]["component_table.jsonl"][
                "row_count"
            ]
        ),
        "prepared table rows",
    )
    table_summary = validate_table_contract(config, table_rows)
    checked("PREPARED_BINDING_AND_SCHEMA", table_summary)

    inputs = load_bound_inputs(repo_root, config)
    rebuilt_table = build_component_table(config=config, inputs=inputs)
    _assert_equal(
        sha256_json(rebuilt_table),
        sha256_json(table_rows),
        "deterministic component table",
    )
    feature_names = [
        str(value) for value in config["feature_contract"]["feature_names"]
    ]
    forbidden = {
        str(value)
        for value in config["feature_contract"]["forbidden_model_fields"]
    }
    for row in rebuilt_table:
        _assert_equal(
            set(row["features"]),
            set(feature_names),
            "exact feature allowlist",
        )
        if forbidden & set(row["features"]):
            raise ValueError("forbidden field entered runtime feature namespace")
    checked(
        "DETERMINISTIC_CAUSAL_TABLE_AND_TRUTH_FIREWALL",
        {
            "row_count": len(rebuilt_table),
            "feature_count": len(feature_names),
            "future_or_truth_feature_count": 0,
        },
    )

    evaluation_path = evaluation_root / "result.json"
    folds_path = evaluation_root / "fold_models.jsonl"
    predictions_path = evaluation_root / "held_out_predictions.jsonl"
    frame_metrics_path = evaluation_root / "frame_metrics.jsonl"
    evaluation = read_json(evaluation_path)
    folds = read_jsonl(folds_path)
    predictions = read_jsonl(predictions_path)
    stored_frame_rows = read_jsonl(frame_metrics_path)
    for name, path, rows in (
        ("fold_models.jsonl", folds_path, folds),
        ("held_out_predictions.jsonl", predictions_path, predictions),
        ("frame_metrics.jsonl", frame_metrics_path, stored_frame_rows),
    ):
        expected = evaluation["output_files"][name]
        _assert_equal(sha256_file(path), expected["sha256"], f"{name} SHA")
        _assert_equal(len(rows), int(expected["row_count"]), f"{name} rows")
    checked("EVALUATION_OUTPUT_BINDINGS", "VALID")

    sessions = sorted(
        {str(row["identity"]["session_id"]) for row in table_rows}
    )
    rows_by_session = Counter(
        str(row["identity"]["session_id"]) for row in table_rows
    )
    folds_by_session: dict[str, dict[str, Any]] = {}
    inner_fold_checks = 0
    for fold in folds:
        held_out = str(fold["held_out_session_id"])
        if held_out in folds_by_session:
            raise ValueError("duplicate outer held-out fold")
        folds_by_session[held_out] = fold
        training = set(str(value) for value in fold["outer_training_session_ids"])
        _assert_equal(training, set(sessions) - {held_out}, "outer split")
        _assert_equal(
            int(fold["outer_held_out_row_count"]),
            rows_by_session[held_out],
            "outer held-out row count",
        )
        inner = fold["inner_oof"]
        _assert_equal(
            bool(inner["outer_held_out_session_present"]),
            False,
            "outer held-out absent from inner",
        )
        inner_held: set[str] = set()
        inner_prediction_rows = 0
        for inner_fold in inner["folds"]:
            inner_session = str(inner_fold["inner_held_out_session_id"])
            inner_held.add(inner_session)
            expected_training = training - {inner_session}
            _assert_equal(
                set(
                    str(value)
                    for value in inner_fold["training_session_ids"]
                ),
                expected_training,
                "inner training split",
            )
            _assert_equal(
                int(inner_fold["held_out_row_count"]),
                rows_by_session[inner_session],
                "inner held-out rows",
            )
            inner_prediction_rows += rows_by_session[inner_session]
            inner_fold_checks += 1
        _assert_equal(inner_held, training, "inner held-out coverage")
        _assert_equal(
            int(inner["prediction_row_count"]),
            inner_prediction_rows,
            "inner OOF prediction rows",
        )
        candidates = fold["threshold_selection"]["candidates"]
        independently_selected = max(
            candidates,
            key=lambda value: (
                float(value["minimum_normalized_gate_margin"]),
                float(
                    value["values"]["minimum_session_recall_retention"]
                ),
                float(value["values"]["fp_pixel_reduction"]),
                -float(value["threshold"]),
            ),
        )
        _assert_equal(
            independently_selected,
            fold["threshold_selection"]["selected"],
            "threshold selection",
        )
        _assert_equal(
            float(independently_selected["threshold"]),
            float(fold["selected_threshold"]),
            "fold threshold",
        )
    _assert_equal(set(folds_by_session), set(sessions), "outer fold coverage")
    checked(
        "NESTED_GROUPED_SESSION_FIREWALL",
        {
            "outer_fold_count": len(folds),
            "inner_fold_check_count": inner_fold_checks,
            "random_component_or_frame_split": False,
        },
    )

    table_by_id = {
        str(row["identity"]["component_id"]): row for row in table_rows
    }
    decisions: dict[str, bool] = {}
    probability_checks = 0
    for prediction in predictions:
        component_id = str(prediction["component_id"])
        row = table_by_id.get(component_id)
        if row is None or component_id in decisions:
            raise ValueError("prediction membership is invalid")
        held_out = str(prediction["held_out_session_id"])
        _assert_equal(
            str(row["identity"]["session_id"]),
            held_out,
            "prediction held-out session",
        )
        fold = folds_by_session[held_out]
        model = fold["model"]
        vector = np.asarray(
            [float(row["features"][name]) for name in feature_names],
            dtype=np.float64,
        )
        standardized = (
            vector - np.asarray(model["mean"], dtype=np.float64)
        ) / np.asarray(model["scale"], dtype=np.float64)
        logit = float(
            standardized
            @ np.asarray(model["coefficients"], dtype=np.float64)
            + float(model["intercept"])
        )
        probability = float(
            1.0 / (1.0 + np.exp(-np.clip(logit, -80.0, 80.0)))
        )
        if (
            abs(probability - float(prediction["probability_keep"]))
            > 1e-12
        ):
            raise ValueError("pure NumPy probability drifted")
        decision = probability >= float(fold["selected_threshold"])
        _assert_equal(decision, bool(prediction["keep"]), "keep decision")
        decisions[component_id] = decision
        probability_checks += 1
    _assert_equal(set(decisions), set(table_by_id), "prediction coverage")
    checked(
        "PURE_NUMPY_HELD_OUT_PREDICTION_RECOMPUTE",
        {"component_check_count": probability_checks},
    )

    contexts = build_frame_contexts(
        repo_root=repo_root,
        config=config,
        inputs=inputs,
        table_rows=table_rows,
    )
    reference_masks = build_reference_masks(config=config, contexts=contexts)
    learned_masks = masks_from_component_decisions(
        config=config,
        contexts=contexts,
        keep_by_component_id=decisions,
    )
    recomputed_frame_rows = build_frame_metric_rows(
        contexts=contexts,
        arm_masks={**reference_masks, LEARNED_ARM_ID: learned_masks},
    )
    _assert_equal(
        sha256_json(recomputed_frame_rows),
        sha256_json(stored_frame_rows),
        "frame metric ledger",
    )
    classes = [str(value) for value in config["candidate_classes"]]
    reports = {
        arm_id: aggregate_arm_report(
            recomputed_frame_rows, arm_id, classes
        )
        for arm_id in ARM_IDS
    }
    _assert_equal(reports, evaluation["arms"], "aggregate arm reports")
    recomputed_utility: dict[str, Any] = {}
    for arm_id in ARM_IDS[1:]:
        values = utility_values(
            candidate=reports[arm_id],
            baseline=reports[RAW_ARM_ID],
        )
        recomputed_utility[arm_id] = {
            "values": values,
            "gates": gate_checks(values, config["utility_gates"]),
            "normalized_gate_margins": normalized_gate_margins(
                values, config["utility_gates"]
            ),
        }
    _assert_equal(recomputed_utility, evaluation["utility"], "nine utility gates")
    checked(
        "FRAME_LEDGER_AND_NINE_GATE_RECOMPUTE",
        {
            "frame_count": len(recomputed_frame_rows),
            "learned_passed_count": recomputed_utility[LEARNED_ARM_ID][
                "gates"
            ]["passed_count"],
        },
    )

    benchmark_path = benchmark_root / "report.json"
    runtime_path = benchmark_root / "runtime_rows.jsonl"
    benchmark = read_json(benchmark_path)
    runtime_rows = read_jsonl(runtime_path)
    _assert_equal(
        sha256_file(runtime_path),
        benchmark["output_files"]["runtime_rows.jsonl"]["sha256"],
        "runtime rows SHA",
    )
    _assert_equal(
        len(runtime_rows),
        int(benchmark["output_files"]["runtime_rows.jsonl"]["row_count"]),
        "runtime row count",
    )
    totals = np.asarray(
        [float(row["total_incremental_ms"]) for row in runtime_rows],
        dtype=np.float64,
    )
    recomputed_p95 = float(np.percentile(totals, 95.0))
    if abs(recomputed_p95 - float(benchmark["latency_ms"]["p95"])) > 1e-12:
        raise ValueError("benchmark P95 drifted")
    model_sizes = [
        _canonical_model_bytes(fold, feature_names) for fold in folds
    ]
    _assert_equal(
        max(model_sizes),
        int(benchmark["memory"]["maximum_model_and_scaler_bytes"]),
        "model memory",
    )
    maximum_components = max(len(context.components) for context in contexts)
    _assert_equal(
        _memory_bound(config, maximum_components),
        int(
            benchmark["memory"]["bounded_state_and_feature_buffer"][
                "total"
            ]
        ),
        "bounded state memory",
    )
    engineering = config["engineering_gates"]
    engineering_passed = (
        recomputed_p95
        < float(
            engineering["maximum_host_p95_incremental_latency_ms_exclusive"]
        )
        and max(model_sizes)
        <= int(engineering["maximum_serialized_model_and_scaler_bytes"])
        and _memory_bound(config, maximum_components)
        <= int(
            engineering[
                "maximum_bounded_incremental_state_and_feature_buffer_bytes"
            ]
        )
    )
    _assert_equal(
        engineering_passed,
        bool(benchmark["engineering_gates"]["all_passed"]),
        "engineering gates",
    )
    checked(
        "ENGINEERING_GATE_RECOMPUTE",
        {
            "p95_incremental_ms": recomputed_p95,
            "maximum_model_bytes": max(model_sizes),
            "bounded_state_and_feature_buffer_bytes": _memory_bound(
                config, maximum_components
            ),
            "all_passed": engineering_passed,
        },
    )

    learned = recomputed_utility[LEARNED_ARM_ID]
    near_miss, near_receipt = near_miss_eligible(
        values=learned["values"],
        utility_gate_result=learned["gates"],
        engineering_passed=engineering_passed,
        stable_diagnostic=evaluation[
            "stable_high_confidence_residual_diagnostic"
        ],
        rule=config["near_miss_rule"],
    )
    if bool(learned["gates"]["all_passed"]) and engineering_passed:
        terminal = "SUPPORTED"
    elif near_miss:
        terminal = "NEAR_MISS_SINGLE_TRAINING_SUCCESSOR"
    else:
        terminal = "NOT_SUPPORTED_AND_GATING_STOP"
    _assert_equal(
        terminal, benchmark["scientific_terminal"], "scientific terminal"
    )
    checked(
        "FROZEN_TERMINAL_MAPPING",
        {
            "scientific_terminal": terminal,
            "near_miss_rule": near_receipt,
        },
    )

    validation = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "status": "VALID",
        "scientific_terminal": terminal,
        "check_count": len(checks),
        "checks": checks,
        "bindings": {
            "config_sha256": sha256_file(config_path),
            "prepare_receipt_sha256": sha256_file(prepare_receipt_path),
            "component_table_sha256": sha256_file(table_path),
            "evaluation_result_sha256": sha256_file(evaluation_path),
            "benchmark_report_sha256": sha256_file(benchmark_path),
        },
        "claim_ceiling": config["claim_ceiling"],
        "fresh_holdout_accessed": False,
        "independent_validation_claimed": False,
        "confirmation_activated": False,
        "android_or_alert_authority": "NONE",
    }
    write_json(output_path, validation)
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_validation(
        repo_root=args.repo_root,
        config_path=args.config,
        prepared_root=args.prepared_root,
        evaluation_root=args.evaluation_root,
        benchmark_root=args.benchmark_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "scientific_terminal": result["scientific_terminal"],
                "check_count": result["check_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
