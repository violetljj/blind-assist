"""Benchmark the frozen cross-fit validator's incremental host path."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import (
    CANDIDATE_ID,
    LEARNED_ARM_ID,
    PROTOCOL_ID,
    atomic_output_directory,
    build_frame_contexts,
    load_bound_inputs,
    near_miss_eligible,
    read_json,
    read_jsonl,
    sha256_file,
    validate_static_config,
    validate_table_contract,
    verify_output_scope,
    write_json,
    write_jsonl,
)


BENCHMARK_SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_learned_component_validator_r0."
    "benchmark.v1"
)
RUNTIME_ROW_SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_learned_component_validator_r0."
    "runtime_row.v1"
)


def _rectangle_hit(
    mask: np.ndarray,
    *,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> bool:
    return bool(np.count_nonzero(mask[y_min:y_max, x_min:x_max]))


def _previous_iou_from_labels(
    mask: np.ndarray,
    area: int,
    labels: np.ndarray | None,
    previous_areas: np.ndarray | None,
) -> tuple[float, bool]:
    if labels is None or previous_areas is None or previous_areas.size <= 1:
        return 0.0, True
    touched = labels[mask]
    intersections = np.bincount(
        touched, minlength=int(previous_areas.size)
    ).astype(np.float64, copy=False)
    intersections[0] = 0.0
    unions = float(area) + previous_areas - intersections
    valid = previous_areas > 0
    valid[0] = False
    if not np.any(valid):
        return 0.0, True
    return float(np.max(intersections[valid] / unions[valid])), False


def _same_footprint_age(
    mask: np.ndarray,
    history: Sequence[np.ndarray],
    maximum: int,
) -> int:
    age = 1
    for prior in reversed(history[-(maximum - 1) :]):
        if np.count_nonzero(mask & prior) == 0:
            break
        age += 1
    return age


def _recent_flicker_count(
    mask: np.ndarray,
    history: Sequence[np.ndarray],
    window: int,
) -> int:
    presence = [
        bool(np.count_nonzero(mask & prior)) for prior in history[-window:]
    ]
    presence.append(True)
    return sum(left != right for left, right in zip(presence, presence[1:]))


class RuntimeFeatureExtractor:
    """Causal state machine using only current and materialized past masks."""

    def __init__(self, config: Mapping[str, Any]):
        contract = config["feature_contract"]
        self.feature_names = [str(value) for value in contract["feature_names"]]
        self.candidate_classes = [
            str(value) for value in config["candidate_classes"]
        ]
        self.height, self.width = (
            int(value) for value in config["analysis_shape"]
        )
        self.diagonal = math.hypot(self.width, self.height)
        self.history_max = int(contract["causal_history_max_observations"])
        self.flicker_window = int(contract["recent_flicker_window_observations"])
        self.upper_end = int(
            math.ceil(
                self.height * float(contract["upper_head_y_max_fraction"])
            )
        )
        central_x = contract["central_body_x_fraction"]
        self.central_x0 = int(math.floor(self.width * float(central_x[0])))
        self.central_x1 = int(math.ceil(self.width * float(central_x[1])))
        self.central_y0 = int(
            math.floor(
                self.height
                * float(contract["central_body_y_min_fraction"])
            )
        )
        self.near_gap = float(contract["near_yolo_union_gap_pixels"])
        self.histories: dict[tuple[str, str, str], list[np.ndarray]] = (
            defaultdict(list)
        )
        self.previous_labels: dict[tuple[str, str, str], np.ndarray] = {}
        self.previous_areas: dict[tuple[str, str, str], np.ndarray] = {}

    def extract(self, context: Any) -> tuple[list[str], np.ndarray]:
        component_ids: list[str] = []
        vectors: list[list[float]] = []
        by_class: dict[str, list[Any]] = defaultdict(list)
        for item in context.components:
            by_class[item.predicted_class].append(item)
        for class_name in self.candidate_classes:
            key = (context.session_id, context.sequence_id, class_name)
            history = self.histories[key]
            previous_labels = self.previous_labels.get(key)
            previous_areas = self.previous_areas.get(key)
            for item in by_class[class_name]:
                component = item.component
                mask = component.mask
                x0, y0, x1, y1 = (int(value) for value in component.bbox)
                bbox_width = x1 - x0
                bbox_height = y1 - y0
                ys, xs = np.nonzero(mask)
                centroid_x = float(np.mean(xs))
                centroid_y = float(np.mean(ys))
                confidence_missing = bool(
                    item.table_row["features"]["top1_confidence_missing"]
                )
                confidence = float(
                    item.table_row["features"]["top1_confidence_median"]
                )
                margin_missing = bool(
                    item.table_row["features"]["top1_top2_margin_missing"]
                )
                margin = float(
                    item.table_row["features"]["top1_top2_margin_median"]
                )
                gap_missing = bool(
                    item.table_row["features"][
                        "nearest_yolo_union_bbox_gap_missing"
                    ]
                )
                normalized_gap = float(
                    item.table_row["features"][
                        "nearest_yolo_union_bbox_gap_fraction"
                    ]
                )
                gap_pixels = (
                    self.diagonal
                    if gap_missing
                    else normalized_gap * self.diagonal
                )
                near = int((not gap_missing) and gap_pixels <= self.near_gap)
                previous_iou, previous_missing = _previous_iou_from_labels(
                    mask,
                    component.area,
                    previous_labels,
                    previous_areas,
                )
                features = {
                    "predicted_class_is_obstacle": float(
                        class_name == "obstacle"
                    ),
                    "log1p_area_pixels": float(math.log1p(component.area)),
                    "bbox_width_fraction": float(bbox_width / self.width),
                    "bbox_height_fraction": float(bbox_height / self.height),
                    "log_bbox_aspect_ratio": float(
                        math.log(bbox_width / bbox_height)
                    ),
                    "centroid_x_fraction": float(centroid_x / self.width),
                    "centroid_y_fraction": float(centroid_y / self.height),
                    "intersects_upper_head_band": float(
                        _rectangle_hit(
                            mask,
                            x_min=0,
                            x_max=self.width,
                            y_min=0,
                            y_max=self.upper_end,
                        )
                    ),
                    "intersects_central_body_corridor": float(
                        _rectangle_hit(
                            mask,
                            x_min=self.central_x0,
                            x_max=self.central_x1,
                            y_min=self.central_y0,
                            y_max=self.height,
                        )
                    ),
                    "top1_confidence_median": confidence,
                    "top1_confidence_missing": float(confidence_missing),
                    "top1_top2_margin_median": margin,
                    "top1_top2_margin_missing": float(margin_missing),
                    "causal_previous_component_iou_max": previous_iou,
                    "causal_previous_component_iou_missing": float(
                        previous_missing
                    ),
                    "causal_same_footprint_age_5": float(
                        _same_footprint_age(
                            mask, history, self.history_max
                        )
                    ),
                    "recent_flicker_count_3": float(
                        _recent_flicker_count(
                            mask, history, self.flicker_window
                        )
                    ),
                    "nearest_yolo_union_bbox_gap_fraction": normalized_gap,
                    "nearest_yolo_union_bbox_gap_missing": float(gap_missing),
                    "near_yolo_union_gap_le_3": float(near),
                    "obstacle_x_near_yolo_union": float(
                        near and class_name == "obstacle"
                    ),
                }
                if list(features) != self.feature_names:
                    raise ValueError("runtime feature order drifted")
                component_ids.append(item.component_id)
                vectors.append([features[name] for name in self.feature_names])

            history.append(context.raw_class_masks[class_name].copy())
            if len(history) > self.history_max:
                del history[:-self.history_max]
            labels = np.zeros(
                (self.height, self.width), dtype=np.uint16
            )
            areas = np.zeros(len(by_class[class_name]) + 1, dtype=np.float64)
            for label, item in enumerate(by_class[class_name], start=1):
                if label > np.iinfo(np.uint16).max:
                    raise ValueError("too many components for uint16 label state")
                labels[item.component.mask] = label
                areas[label] = float(item.component.area)
            self.previous_labels[key] = labels
            self.previous_areas[key] = areas

        matrix = np.asarray(vectors, dtype=np.float64)
        expected_shape = (len(component_ids), len(self.feature_names))
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError("runtime feature matrix is invalid")
        return component_ids, matrix


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


def _bounded_memory_bytes(
    *,
    config: Mapping[str, Any],
    contexts: Sequence[Any],
) -> dict[str, int]:
    height, width = (int(value) for value in config["analysis_shape"])
    class_count = len(config["candidate_classes"])
    history_max = int(
        config["feature_contract"]["causal_history_max_observations"]
    )
    maximum_components = max(len(context.components) for context in contexts)
    feature_count = len(config["feature_contract"]["feature_names"])
    history = class_count * history_max * height * width
    previous_labels = class_count * height * width * np.dtype(np.uint16).itemsize
    output_label_mask = height * width * np.dtype(np.uint8).itemsize
    feature_and_standardized = (
        maximum_components
        * feature_count
        * np.dtype(np.float64).itemsize
        * 2
    )
    scores_and_decisions = maximum_components * (
        np.dtype(np.float64).itemsize + np.dtype(np.bool_).itemsize
    )
    previous_area_arrays = (
        class_count * (maximum_components + 1) * np.dtype(np.float64).itemsize
    )
    result = {
        "history_bool_masks": int(history),
        "previous_uint16_label_maps": int(previous_labels),
        "output_uint8_class_label_mask": int(output_label_mask),
        "feature_and_standardized_buffers": int(feature_and_standardized),
        "score_and_decision_buffers": int(scores_and_decisions),
        "previous_component_area_arrays": int(previous_area_arrays),
        "maximum_components_in_one_frame": int(maximum_components),
    }
    result["total"] = sum(
        value
        for key, value in result.items()
        if key != "maximum_components_in_one_frame"
    )
    return result


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_benchmark(
    *,
    repo_root: Path,
    config_path: Path,
    prepared_root: Path,
    evaluation_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    prepared_root = prepared_root.resolve()
    evaluation_root = evaluation_root.resolve()
    output_root = output_root.resolve()
    verify_output_scope(repo_root, output_root)
    config = read_json(config_path)
    validate_static_config(config)
    table_path = prepared_root / "component_table.jsonl"
    evaluation_path = evaluation_root / "result.json"
    fold_path = evaluation_root / "fold_models.jsonl"
    prediction_path = evaluation_root / "held_out_predictions.jsonl"
    table_rows = read_jsonl(table_path)
    validate_table_contract(config, table_rows)
    evaluation = read_json(evaluation_path)
    folds = read_jsonl(fold_path)
    predictions = read_jsonl(prediction_path)
    for name, path in (
        ("fold_models.jsonl", fold_path),
        ("held_out_predictions.jsonl", prediction_path),
    ):
        expected = evaluation["output_files"][name]
        if (
            sha256_file(path) != expected["sha256"]
            or len(read_jsonl(path)) != int(expected["row_count"])
        ):
            raise ValueError(f"evaluation output binding drifted: {name}")
    inputs = load_bound_inputs(repo_root, config)
    contexts = build_frame_contexts(
        repo_root=repo_root,
        config=config,
        inputs=inputs,
        table_rows=table_rows,
    )
    folds_by_session = {
        str(fold["held_out_session_id"]): fold for fold in folds
    }
    if len(folds_by_session) != len(folds):
        raise ValueError("duplicate held-out session fold")
    expected_prediction = {
        str(row["component_id"]): row for row in predictions
    }
    if len(expected_prediction) != len(predictions):
        raise ValueError("duplicate held-out prediction")

    feature_names = [
        str(value) for value in config["feature_contract"]["feature_names"]
    ]
    warmups = int(config["engineering_gates"]["warmup_repetitions"])
    measured = int(config["engineering_gates"]["measured_repetitions"])
    runtime_rows: list[dict[str, Any]] = []
    verification_count = 0
    garbage_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repetition in range(warmups + measured):
            extractor = RuntimeFeatureExtractor(config)
            for context in contexts:
                fold = folds_by_session[context.session_id]
                model = fold["model"]
                start = time.perf_counter_ns()
                component_ids, matrix = extractor.extract(context)
                feature_end = time.perf_counter_ns()
                standardized = (
                    matrix - np.asarray(model["mean"], dtype=np.float64)
                ) / np.asarray(model["scale"], dtype=np.float64)
                logits = (
                    standardized
                    @ np.asarray(model["coefficients"], dtype=np.float64)
                    + float(model["intercept"])
                )
                scores = 1.0 / (
                    1.0 + np.exp(-np.clip(logits, -80.0, 80.0))
                )
                keep = scores >= float(fold["selected_threshold"])
                classifier_end = time.perf_counter_ns()
                output = np.zeros(
                    tuple(int(value) for value in config["analysis_shape"]),
                    dtype=np.uint8,
                )
                item_by_id = {
                    item.component_id: item for item in context.components
                }
                for component_id, decision in zip(component_ids, keep):
                    if bool(decision):
                        item = item_by_id[component_id]
                        output[item.component.mask] = (
                            2 if item.predicted_class == "obstacle" else 1
                        )
                end = time.perf_counter_ns()

                if repetition == 0:
                    rows_by_id = {
                        item.component_id: item.table_row
                        for item in context.components
                    }
                    expected_matrix = np.asarray(
                        [
                            [
                                float(rows_by_id[component_id]["features"][name])
                                for name in feature_names
                            ]
                            for component_id in component_ids
                        ],
                        dtype=np.float64,
                    )
                    if not np.allclose(
                        matrix, expected_matrix, atol=1e-12, rtol=0.0
                    ):
                        raise ValueError(
                            f"runtime feature reproduction drifted: "
                            f"{context.view_row_id}"
                        )
                    for component_id, score, decision in zip(
                        component_ids, scores, keep
                    ):
                        expected = expected_prediction[component_id]
                        if (
                            abs(float(score) - float(expected["probability_keep"]))
                            > 1e-12
                            or bool(decision) != bool(expected["keep"])
                        ):
                            raise ValueError(
                                f"runtime prediction reproduction drifted: "
                                f"{component_id}"
                            )
                        verification_count += 1
                if repetition >= warmups:
                    runtime_rows.append(
                        {
                            "schema_version": RUNTIME_ROW_SCHEMA_VERSION,
                            "protocol_id": PROTOCOL_ID,
                            "candidate_id": CANDIDATE_ID,
                            "repetition": repetition - warmups,
                            "view_row_id": context.view_row_id,
                            "session_id": context.session_id,
                            "component_count": len(component_ids),
                            "kept_component_count": int(np.count_nonzero(keep)),
                            "feature_extraction_ms": (
                                feature_end - start
                            )
                            / 1_000_000.0,
                            "classification_ms": (
                                classifier_end - feature_end
                            )
                            / 1_000_000.0,
                            "postprocess_ms": (
                                end - classifier_end
                            )
                            / 1_000_000.0,
                            "total_incremental_ms": (end - start)
                            / 1_000_000.0,
                        }
                    )
    finally:
        if garbage_was_enabled:
            gc.enable()

    totals = [
        float(row["total_incremental_ms"]) for row in runtime_rows
    ]
    model_sizes = [
        _canonical_model_bytes(fold, feature_names) for fold in folds
    ]
    memory = _bounded_memory_bytes(config=config, contexts=contexts)
    engineering = config["engineering_gates"]
    checks = {
        "host_p95_incremental_latency_ms": {
            "operator": "<",
            "threshold": float(
                engineering["maximum_host_p95_incremental_latency_ms_exclusive"]
            ),
            "value": _percentile(totals, 95.0),
        },
        "serialized_model_and_scaler_bytes": {
            "operator": "<=",
            "threshold": int(
                engineering["maximum_serialized_model_and_scaler_bytes"]
            ),
            "value": max(model_sizes),
        },
        "bounded_incremental_state_and_feature_buffer_bytes": {
            "operator": "<=",
            "threshold": int(
                engineering[
                    "maximum_bounded_incremental_state_and_feature_buffer_bytes"
                ]
            ),
            "value": int(memory["total"]),
        },
    }
    for check in checks.values():
        if check["operator"] == "<":
            check["passed"] = bool(check["value"] < check["threshold"])
        else:
            check["passed"] = bool(check["value"] <= check["threshold"])
    engineering_passed = all(bool(value["passed"]) for value in checks.values())
    utility = evaluation["utility"][LEARNED_ARM_ID]
    utility_passed = bool(utility["gates"]["all_passed"])
    near_miss, near_miss_receipt = near_miss_eligible(
        values=utility["values"],
        utility_gate_result=utility["gates"],
        engineering_passed=engineering_passed,
        stable_diagnostic=evaluation[
            "stable_high_confidence_residual_diagnostic"
        ],
        rule=config["near_miss_rule"],
    )
    if utility_passed and engineering_passed:
        terminal = "SUPPORTED"
    elif near_miss:
        terminal = "NEAR_MISS_SINGLE_TRAINING_SUCCESSOR"
    else:
        terminal = "NOT_SUPPORTED_AND_GATING_STOP"
    if terminal not in config["valid_scientific_terminals"]:
        raise ValueError("scientific terminal mapping drifted")

    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "stage": config["stage"],
        "evidence_instance": config["evidence_instance"],
        "execution_status": "COMPLETE",
        "scientific_terminal": terminal,
        "claim_ceiling": config["claim_ceiling"],
        "bindings": {
            "config_sha256": sha256_file(config_path),
            "component_table_sha256": sha256_file(table_path),
            "evaluation_result_sha256": sha256_file(evaluation_path),
            "fold_models_sha256": sha256_file(fold_path),
            "held_out_predictions_sha256": sha256_file(prediction_path),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "timing_contract": {
            "warmup_repetitions": warmups,
            "measured_repetitions": measured,
            "measured_frame_rows": len(runtime_rows),
            "excludes": engineering["excludes"],
            "includes": engineering["includes"],
            "truth_or_file_io_inside_timed_region": False,
            "fit_inside_timed_region": False,
            "preexisting_raw_components_and_masks": True,
            "garbage_collector_disabled_inside_timed_region": True,
        },
        "latency_ms": {
            "minimum": min(totals),
            "median": _percentile(totals, 50.0),
            "p95": _percentile(totals, 95.0),
            "p99": _percentile(totals, 99.0),
            "maximum": max(totals),
        },
        "memory": {
            "model_and_scaler_bytes_by_fold": model_sizes,
            "maximum_model_and_scaler_bytes": max(model_sizes),
            "bounded_state_and_feature_buffer": memory,
            "representation": (
                "bool causal history masks, uint16 previous-component label "
                "maps, uint8 class-label output mask, float64 feature buffers"
            ),
        },
        "runtime_reproduction": {
            "status": "EXACT_FEATURE_AND_PREDICTION_REPRODUCED",
            "component_check_count": verification_count,
        },
        "engineering_gates": {
            "checks": checks,
            "passed_count": sum(
                bool(value["passed"]) for value in checks.values()
            ),
            "all_passed": engineering_passed,
        },
        "utility_all_nine_passed": utility_passed,
        "near_miss_rule": near_miss_receipt,
        "terminal_action": {
            "active_learned_component_gating": (
                "ELIGIBLE_FOR_SEPARATE_CONFIRMATION_DESIGN"
                if terminal == "SUPPORTED"
                else "STOPPED_ON_CURRENT_REFERENCE"
            ),
            "single_training_successor": (
                config["near_miss_rule"]["successor_authority"]
                if terminal == "NEAR_MISS_SINGLE_TRAINING_SUCCESSOR"
                else "NONE"
            ),
            "android_or_alert_authority": "NONE",
            "fresh_holdout_accessed": False,
            "confirmation_activated": False,
        },
    }

    temporary, finalize = atomic_output_directory(output_root)
    try:
        runtime_path = temporary / "runtime_rows.jsonl"
        write_jsonl(runtime_path, runtime_rows)
        report["output_files"] = {
            "runtime_rows.jsonl": {
                "sha256": sha256_file(runtime_path),
                "row_count": len(runtime_rows),
            }
        }
        write_json(temporary / "report.json", report)
        finalize(True)
    except BaseException:
        finalize(False)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        repo_root=args.repo_root,
        config_path=args.config,
        prepared_root=args.prepared_root,
        evaluation_root=args.evaluation_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": report["execution_status"],
                "scientific_terminal": report["scientific_terminal"],
                "p95_incremental_ms": report["latency_ms"]["p95"],
                "engineering_all_passed": report["engineering_gates"][
                    "all_passed"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
