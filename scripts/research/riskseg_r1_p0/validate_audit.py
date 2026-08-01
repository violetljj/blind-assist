"""Independent feature-canary and event scorer for RISKSEG-R1 P0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


MODEL_ARMS = ("seed-20260801", "seed-20260802", "seed-20260803")
ORACLE_ARM = "truth-mask"
ALL_ARMS = (*MODEL_ARMS, ORACLE_ARM)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(value)
    return rows


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_bound(root: Path, item: dict[str, Any]) -> Path:
    path = (root / item["path"]).resolve()
    if sha256_file(path) != str(item["sha256"]).lower():
        raise ValueError(f"{path}: bound SHA drift")
    return path


def verify_code(root: Path, contract: dict[str, Any]) -> None:
    lock = contract["implementation_lock"]
    files = {
        "core_py_sha256": "scripts/research/riskseg_r1_p0/core.py",
        "run_audit_py_sha256": "scripts/research/riskseg_r1_p0/run_audit.py",
        "validate_audit_py_sha256": "scripts/research/riskseg_r1_p0/validate_audit.py",
        "test_core_py_sha256": "scripts/research/riskseg_r1_p0/test_core.py",
    }
    for key, relative in files.items():
        if sha256_file(root / relative) != str(lock[key]).lower():
            raise ValueError(f"{relative}: implementation lock drift")


def configs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    grid = contract["adapter_grid"]
    result = []
    for boundary_weight in grid["boundary_weights"]:
        for top_fraction in grid["top_fractions"]:
            for profile in sorted(grid["lateral_profiles"]):
                result.append(
                    {
                        "config_id": (
                            f"bw{boundary_weight:g}_top{top_fraction:g}_{profile}"
                        ),
                        "boundary_weight": float(boundary_weight),
                        "top_fraction": float(top_fraction),
                        "lateral_weights": [
                            float(value)
                            for value in grid["lateral_profiles"][profile]
                        ],
                    }
                )
    return result


def threshold_values(contract: dict[str, Any]) -> list[float]:
    item = contract["threshold_grid"]
    return [
        value / 1000.0
        for value in range(
            int(item["start_milli"]),
            int(item["stop_milli"]) + 1,
            int(item["step_milli"]),
        )
    ]


def assignments(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> dict[str, int]:
    fold_count = int(contract["nested_development"]["outer_fold_count"])
    offsets = contract["nested_development"]["bucket_fold_offsets"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in manifest["events"]:
        grouped[event["bucket"]].append(event)
    result: dict[str, int] = {}
    for bucket in sorted(grouped):
        for position, event in enumerate(
            sorted(grouped[bucket], key=lambda value: value["parent_event_id"])
        ):
            result[event["parent_event_id"]] = (
                position + int(offsets[bucket])
            ) % fold_count
    sizes = [
        sum(value == fold for value in result.values())
        for fold in range(fold_count)
    ]
    if sizes != [6, 6, 6, 6, 6]:
        raise ValueError(f"fold cardinality drift: {sizes}")
    return result


def zone_masks(
    height: int, width: int, geometry: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top = int(height * float(geometry["top_ratio"]))
    zones = [np.zeros((height, width), dtype=bool) for _ in range(3)]
    denominator = max(1, height - 1 - top)
    for y in range(top, height):
        progress = (y - top) / denominator
        half = float(geometry["top_half_width_ratio"]) + progress * (
            float(geometry["bottom_half_width_ratio"])
            - float(geometry["top_half_width_ratio"])
        )
        left = max(0, min(width - 1, int(width * (0.5 - half))))
        right = max(left + 1, min(width, int(width * (0.5 + half))))
        span = right - left
        for x in range(left, right):
            zones[min(2, 3 * (x - left) // span)][y, x] = True
    if any(not value.any() for value in zones):
        raise ValueError("empty independent corridor zone")
    return zones[0], zones[1], zones[2]


def top_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, math.ceil(values.size * fraction))
    if count == values.size:
        return float(values.mean())
    start = values.size - count
    return float(np.partition(values, start)[start:].mean())


def independent_pool(
    probabilities: np.ndarray, contract: dict[str, Any]
) -> dict[str, float]:
    if probabilities.ndim == 4:
        probabilities = probabilities[0]
    zones = zone_masks(
        probabilities.shape[0],
        probabilities.shape[1],
        contract["corridor_geometry"],
    )
    known = 1.0 - probabilities[..., 3]
    result: dict[str, float] = {}
    for config in configs(contract):
        evidence = known * (
            probabilities[..., 1]
            + config["boundary_weight"] * probabilities[..., 2]
        )
        pooled = [
            top_mean(evidence[zone], config["top_fraction"]) for zone in zones
        ]
        result[config["config_id"]] = max(
            weight * value
            for weight, value in zip(config["lateral_weights"], pooled)
        )
    return result


def quantization(detail: dict[str, Any], label: str) -> tuple[float, int]:
    parameters = detail["quantization_parameters"]
    scales = np.asarray(parameters["scales"])
    zero_points = np.asarray(parameters["zero_points"])
    if scales.size != 1 or zero_points.size != 1:
        raise ValueError(f"{label}: per-tensor quantization required")
    scale = float(scales[0])
    zero = int(zero_points[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label}: invalid quantization scale")
    return scale, zero


def independent_input(image_path: Path, detail: dict[str, Any]) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    scale, zero = quantization(detail, "input")
    values = (image / 255.0 - MEAN) / STD
    return np.clip(
        np.floor(values / scale + zero + 0.5), -128, 127
    ).astype(np.int8)[None, ...]


def independent_softmax(raw: np.ndarray, scale: float) -> np.ndarray:
    values = raw.astype(np.float32)
    values -= values.max(axis=-1, keepdims=True)
    values *= np.float32(scale)
    np.exp(values, out=values)
    values /= values.sum(axis=-1, keepdims=True)
    return values


def validate_feature_canaries(
    *,
    root: Path,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    manifest_root: Path,
    row_index: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    from ai_edge_litert.interpreter import Interpreter

    checked = 0
    # First frame of every event is a deterministic 30-session canary set.
    for arm in MODEL_ARMS:
        model_path = verify_bound(root, contract["models"][arm])
        interpreter = Interpreter(model_path=str(model_path), num_threads=4)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        output_scale, output_zero = quantization(output_detail, f"{arm} output")
        for event in manifest["events"]:
            frame = event["frames"][0]
            key = (arm, event["parent_event_id"], 0)
            row = row_index[key]
            image_path = manifest_root / frame["image_path"]
            input_value = independent_input(image_path, input_detail)
            interpreter.set_tensor(input_detail["index"], input_value)
            interpreter.invoke()
            raw = interpreter.get_tensor(output_detail["index"])
            raw_sha = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
            if raw_sha != row["raw_int8_output_sha256"]:
                raise ValueError(f"{key}: raw output canary SHA mismatch")
            if (
                float(row["output_quantization_scale"]) != output_scale
                or int(row["output_quantization_zero_point"]) != output_zero
            ):
                raise ValueError(f"{key}: quantization metadata mismatch")
            expected = independent_pool(
                independent_softmax(raw, output_scale), contract
            )
            actual = row["adapter_scores"]
            for config_id, value in expected.items():
                if not math.isclose(
                    float(actual[config_id]), value, rel_tol=1e-6, abs_tol=1e-7
                ):
                    raise ValueError(f"{key}/{config_id}: pooling mismatch")
            checked += 1
        del interpreter
    for event in manifest["events"]:
        frame = event["frames"][0]
        mask_path = manifest_root / frame["oracle_mask_path"]
        mask = np.asarray(Image.open(mask_path))
        expected = independent_pool(np.eye(4, dtype=np.float32)[mask], contract)
        actual = row_index[(ORACLE_ARM, event["parent_event_id"], 0)][
            "adapter_scores"
        ]
        for config_id, value in expected.items():
            if not math.isclose(
                float(actual[config_id]), value, rel_tol=1e-6, abs_tol=1e-7
            ):
                raise ValueError(
                    f"{event['parent_event_id']}/{config_id}: oracle pooling mismatch"
                )
        checked += 1
    return {
        "canary_frame_count": checked,
        "model_canaries": 90,
        "oracle_canaries": 30,
        "selection": "first frame of every parent event for every arm",
    }


def event_score(
    event: dict[str, Any],
    rows: list[dict[str, Any]],
    config_id: str,
    threshold: float,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda value: int(value["frame_index"]))
    active = [
        float(row["adapter_scores"][config_id]) >= threshold for row in rows
    ]
    positive = bool(event["positive"])
    hits: list[int] = []
    passed: list[int] = []
    if positive:
        start, end = map(int, event["alertable_interval_frames"])
        passed_start, passed_end = map(int, event["passed_interval_frames"])
        hits = [index for index in range(start, end + 1) if active[index]]
        passed = [
            index for index in range(passed_start, passed_end + 1) if active[index]
        ]
    return {
        "parent_event_id": event["parent_event_id"],
        "source_session_id": event["source_session_id"],
        "bucket": event["bucket"],
        "positive": positive,
        "event_hit": positive and bool(hits),
        "critical_miss": positive and not hits,
        "false_alert_event": (not positive) and any(active),
        "passed_cleared": positive and not passed,
        "first_alertable_alert_frame": hits[0] if hits else None,
    }


def aggregate(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(scores)
    positives = [value for value in values if value["positive"]]
    negatives = [value for value in values if not value["positive"]]
    hits = sum(bool(value["event_hit"]) for value in positives)
    cleared = sum(bool(value["passed_cleared"]) for value in positives)
    return {
        "positive_event_count": len(positives),
        "hit_event_count": hits,
        "event_recall": hits / len(positives) if positives else None,
        "critical_miss_count": len(positives) - hits,
        "negative_event_count": len(negatives),
        "false_alert_event_count": sum(
            bool(value["false_alert_event"]) for value in negatives
        ),
        "cleared_event_count": cleared,
        "clearance_rate": cleared / len(positives) if positives else None,
        "bucket_hit_counts": {
            bucket: sum(
                bool(value["event_hit"])
                for value in positives
                if value["bucket"] == bucket
            )
            for bucket in (
                "blocking_obstacle_positive",
                "boundary_level_change_positive",
            )
        },
    }


def rank(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    config_id: str,
    threshold: float,
) -> tuple[Any, ...]:
    hit_deficit = max(
        0, baseline["hit_event_count"] - metrics["hit_event_count"]
    )
    false_excess = max(
        0,
        metrics["false_alert_event_count"]
        - baseline["false_alert_event_count"],
    )
    clearance_deficit = max(
        0, baseline["cleared_event_count"] - metrics["cleared_event_count"]
    )
    return (
        hit_deficit + false_excess + clearance_deficit,
        hit_deficit,
        false_excess,
        clearance_deficit,
        -metrics["hit_event_count"],
        metrics["false_alert_event_count"],
        -metrics["cleared_event_count"],
        config_id,
        threshold,
    )


def score_arm(
    arm: str,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    yolo: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    events = {event["parent_event_id"]: event for event in manifest["events"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            grouped[row["parent_event_id"]].append(row)
    fold_by_event = assignments(manifest, contract)
    selections = []
    outer_scores = []
    for outer_fold in range(5):
        inner_ids = sorted(
            event_id
            for event_id, fold in fold_by_event.items()
            if fold != outer_fold
        )
        outer_ids = sorted(
            event_id
            for event_id, fold in fold_by_event.items()
            if fold == outer_fold
        )
        baseline = aggregate(yolo[event_id] for event_id in inner_ids)
        candidates = []
        for config in configs(contract):
            for threshold in threshold_values(contract):
                inner = [
                    event_score(
                        events[event_id],
                        grouped[event_id],
                        config["config_id"],
                        threshold,
                    )
                    for event_id in inner_ids
                ]
                metrics = aggregate(inner)
                candidates.append(
                    (
                        rank(
                            metrics,
                            baseline,
                            config["config_id"],
                            threshold,
                        ),
                        config,
                        threshold,
                        metrics,
                    )
                )
        candidate_rank, selected, threshold, metrics = min(
            candidates, key=lambda value: value[0]
        )
        selections.append(
            {
                "outer_fold": outer_fold,
                "inner_event_count": len(inner_ids),
                "outer_event_count": len(outer_ids),
                "selected_config": selected,
                "selected_threshold": threshold,
                "selection_rank": list(candidate_rank),
                "inner_adapter_metrics": metrics,
                "inner_yolo_metrics": baseline,
            }
        )
        for event_id in outer_ids:
            score = event_score(
                events[event_id],
                grouped[event_id],
                selected["config_id"],
                threshold,
            )
            score.update(
                {
                    "outer_fold": outer_fold,
                    "selected_config_id": selected["config_id"],
                    "selected_threshold": threshold,
                }
            )
            outer_scores.append(score)
    outer_scores.sort(key=lambda value: value["parent_event_id"])
    delays = [
        int(score["first_alertable_alert_frame"])
        - int(yolo[score["parent_event_id"]]["first_alertable_alert_frame"])
        for score in outer_scores
        if score["event_hit"] and yolo[score["parent_event_id"]]["event_hit"]
    ]
    return {
        "arm": arm,
        "fold_selections": selections,
        "oof_event_scores": outer_scores,
        "oof_aggregate": aggregate(outer_scores),
        "timing_against_yolo": {
            "common_hit_count": len(delays),
            "median_delay_frames": (
                float(statistics.median(delays)) if delays else None
            ),
            "late_over_two_frames_rate": (
                sum(value > 2 for value in delays) / len(delays)
                if delays
                else None
            ),
            "delays": delays,
        },
    }


def yolo_scores(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for value in report["event_summaries"]:
        if value["arm"] == "A_CURRENT_YOLO_ONLY":
            result[value["parent_event_id"]] = {
                key: value[key]
                for key in (
                    "parent_event_id",
                    "source_session_id",
                    "bucket",
                    "positive",
                    "event_hit",
                    "critical_miss",
                    "false_alert_event",
                    "passed_cleared",
                    "first_alertable_alert_frame",
                )
            }
    return result


def old_metrics(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for arm in MODEL_ARMS:
        report = read_object(verify_bound(root, contract["old_adapter_reports"][arm]))
        value = report["arm_aggregates"]["B_LEARNED_SEGMENTATION_ONLY"]
        result[arm] = {
            "seed": int(arm.removeprefix("seed-")),
            "hit_event_count": int(value["hit_event_count"]),
            "false_alert_event_count": int(value["false_alert_event_count"]),
            "cleared_event_count": int(value["cleared_event_count"]),
        }
    return result


def independent_decision(
    nested: dict[str, dict[str, Any]],
    old: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    truth = nested[ORACLE_ARM]["oof_aggregate"]
    truth_checks = {
        "hit_event_count": truth["hit_event_count"]
        >= contract["gates"]["truth_mask"]["minimum_hit_event_count"],
        "false_alert_event_count": truth["false_alert_event_count"]
        <= contract["gates"]["truth_mask"]["maximum_false_alert_event_count"],
        "cleared_event_count": truth["cleared_event_count"]
        >= contract["gates"]["truth_mask"]["minimum_cleared_event_count"],
    }
    learned = {}
    for arm in MODEL_ARMS:
        current, baseline = nested[arm]["oof_aggregate"], old[arm]
        guardrails = (
            current["hit_event_count"] >= baseline["hit_event_count"]
            and current["false_alert_event_count"]
            <= baseline["false_alert_event_count"]
            and current["cleared_event_count"] >= baseline["cleared_event_count"]
        )
        strict = (
            current["hit_event_count"] > baseline["hit_event_count"]
            or current["false_alert_event_count"]
            < baseline["false_alert_event_count"]
            or current["cleared_event_count"] > baseline["cleared_event_count"]
        )
        learned[arm] = {
            "guardrails_pass": guardrails,
            "strict_event_improvement": strict,
            "pass": guardrails and strict,
            "old_adapter": baseline,
            "soft_adapter": current,
        }
    count = sum(value["pass"] for value in learned.values())
    authorized = (
        all(truth_checks.values())
        and learned["seed-20260801"]["pass"]
        and count >= 2
    )
    if not all(truth_checks.values()):
        terminal = "TRUTH_MASK_SOFT_ADAPTER_FAIL_CHANGE_ACTIONABILITY_LABELS"
    elif authorized:
        terminal = "RISKSEG_R1_P1_DESIGN_AUTHORIZED"
    else:
        terminal = "RISKSEG_R1_P1_NOT_AUTHORIZED_SOFT_ADAPTER_SIGNAL_INSUFFICIENT"
    return {
        "terminal": terminal,
        "truth_mask_gate": {
            **truth_checks,
            "pass": all(truth_checks.values()),
            "metrics": truth,
        },
        "learned_seed_checks": learned,
        "passing_seed_count": count,
        "decision_seed_pass": learned["seed-20260801"]["pass"],
        "p1_design_authorized": authorized,
        "claim_ceiling": (
            "consumed nested Development mechanism evidence only; "
            "not fresh confirmation, App promotion, or safety evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    report_path = args.report.resolve()
    contract = read_object(contract_path)
    report = read_object(report_path)
    root = contract_path.parents[3]
    verify_code(root, contract)
    if report["contract_sha256"] != sha256_file(contract_path):
        raise ValueError("contract SHA mismatch")
    manifest_path = verify_bound(root, contract["frozen_manifest"])
    verify_bound(root, contract["frozen_manifest_receipt"])
    manifest = read_object(manifest_path)
    fold_map = assignments(manifest, contract)
    if canonical_sha256(fold_map) != report["fold_assignment_sha256"]:
        raise ValueError("fold assignment mismatch")
    trace_path = report_path.parent / report["feature_trace"]["path"]
    if sha256_file(trace_path) != report["feature_trace"]["sha256"]:
        raise ValueError("feature trace SHA mismatch")
    rows = read_jsonl(trace_path)
    if len(rows) != 7680:
        raise ValueError(f"feature trace row count {len(rows)} != 7680")
    row_index = {
        (row["arm"], row["parent_event_id"], int(row["frame_index"])): row
        for row in rows
    }
    if len(row_index) != len(rows):
        raise ValueError("duplicate frame identity")
    expected = {
        (arm, event["parent_event_id"], int(frame["frame_index"]))
        for arm in ALL_ARMS
        for event in manifest["events"]
        for frame in event["frames"]
    }
    if set(row_index) != expected:
        raise ValueError("feature trace membership drift")
    canaries = validate_feature_canaries(
        root=root,
        contract=contract,
        manifest=manifest,
        manifest_root=manifest_path.parent,
        row_index=row_index,
    )
    yolo = yolo_scores(read_object(verify_bound(root, contract["yolo_reference_report"])))
    old = old_metrics(root, contract)
    nested = {
        arm: score_arm(arm, manifest, rows, yolo, contract) for arm in ALL_ARMS
    }
    if nested != report["nested_oof"]:
        raise ValueError("independent nested OOF recomputation mismatch")
    decision = independent_decision(nested, old, contract)
    if decision != report["decision"]:
        raise ValueError("independent decision mismatch")
    evidence_root = (root / "artifacts.local/evidence/riskseg-r1").resolve()
    output = args.output.resolve()
    try:
        output.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(f"validation output must stay below {evidence_root}") from exc
    if output.exists():
        raise ValueError(f"validation output must be new: {output}")
    value = {
        "schema_version": "blindassist.riskseg_r1.p0_validation.v2",
        "status": "PASS",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "trace_sha256": sha256_file(trace_path),
        "row_count": len(rows),
        "feature_canary_validation": canaries,
        "terminal": decision["terminal"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
