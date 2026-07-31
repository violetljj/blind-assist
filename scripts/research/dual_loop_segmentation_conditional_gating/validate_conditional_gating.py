"""Independently validate the conditional-gating aggregation and terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


PROTOCOL_ID = "DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0"
BASELINE_ID = "BASELINE_UNFILTERED"
REFERENCE_IDS = (
    "REFERENCE_CAUSAL_2_OF_3_UNION",
    "REFERENCE_CONFIDENCE_GE_0_65",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _aggregate_pixels(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty pixel rows")
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)
    predicted = sum(int(row["predicted_pixels"]) for row in rows)
    truth = sum(int(row["truth_pixels"]) for row in rows)
    empty = tp + fp + fn == 0
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    iou = _ratio(tp, tp + fp + fn)
    if empty:
        precision = recall = iou = 1.0
    f1 = (
        None
        if precision is None or recall is None
        else 0.0
        if precision + recall == 0
        else float(2.0 * precision * recall / (precision + recall))
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": predicted,
        "truth_pixels": truth,
        "pixel_count": tp + fp + fn + tn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
    }


def _assert_numeric_equal(actual: Any, expected: Any, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label}: {actual!r} != {expected!r}")
        return
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if abs(float(actual) - float(expected)) > 1e-12:
            raise ValueError(f"{label}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise ValueError(f"{label}: {actual!r} != {expected!r}")


def _validate_aggregate(
    rows: Sequence[dict[str, Any]],
    arm_id: str,
    candidate_classes: Sequence[str],
    reported: dict[str, Any],
    label: str,
) -> int:
    checks = 0
    pixel = _aggregate_pixels([row["arms"][arm_id]["pixel"] for row in rows])
    for field in (
        "tp",
        "fp",
        "fn",
        "tn",
        "predicted_pixels",
        "truth_pixels",
        "precision",
        "recall",
        "iou",
        "f1",
    ):
        _assert_numeric_equal(reported["pixel"][field], pixel[field], f"{label}.pixel.{field}")
        checks += 1
    frame_count = len(rows)
    sums = {
        "post_component_count": sum(
            int(row["arms"][arm_id]["post_component_count"]) for row in rows
        ),
        "any_hazard_false_component_count": sum(
            int(row["arms"][arm_id]["any_hazard_false_component_count"])
            for row in rows
        ),
        "class_strict_false_component_count": sum(
            int(row["arms"][arm_id]["class_strict_false_component_count"])
            for row in rows
        ),
    }
    for field, value in sums.items():
        _assert_numeric_equal(reported[field], value, f"{label}.{field}")
        checks += 1
    rates = {
        "post_components_per_frame": sums["post_component_count"] / frame_count,
        "any_hazard_false_components_per_frame": (
            sums["any_hazard_false_component_count"] / frame_count
        ),
        "class_strict_false_components_per_frame": (
            sums["class_strict_false_component_count"] / frame_count
        ),
    }
    for field, value in rates.items():
        _assert_numeric_equal(reported[field], value, f"{label}.{field}")
        checks += 1
    for class_name in candidate_classes:
        class_report = reported["classes"][class_name]
        class_rows = [
            row["arms"][arm_id]["classes"][class_name]["pixel"] for row in rows
        ]
        aggregate = _aggregate_pixels(class_rows)
        for field in ("tp", "fp", "fn", "tn", "precision", "recall", "iou", "f1"):
            _assert_numeric_equal(
                class_report["pixel"][field],
                aggregate[field],
                f"{label}.{class_name}.{field}",
            )
            checks += 1
    return checks


def _validate_comparison(
    reported: dict[str, Any],
    baseline: dict[str, Any],
    label: str,
) -> int:
    expected_fp = _ratio(
        int(baseline["pixel"]["fp"]) - int(reported["pixel"]["fp"]),
        int(baseline["pixel"]["fp"]),
    )
    expected_recall = _ratio(
        int(reported["pixel"]["tp"]),
        int(baseline["pixel"]["tp"]),
    )
    comparison = reported["comparison_to_baseline"]
    _assert_numeric_equal(
        comparison["false_positive_reduction"], expected_fp, f"{label}.fp_reduction"
    )
    _assert_numeric_equal(
        comparison["recall_retention"], expected_recall, f"{label}.recall_retention"
    )
    checks = 2
    for class_name, class_report in reported["classes"].items():
        baseline_class = baseline["classes"][class_name]
        expected = {
            "false_positive_reduction": _ratio(
                int(baseline_class["pixel"]["fp"]) - int(class_report["pixel"]["fp"]),
                int(baseline_class["pixel"]["fp"]),
            ),
            "recall_retention": _ratio(
                int(class_report["pixel"]["tp"]),
                int(baseline_class["pixel"]["tp"]),
            ),
        }
        for field, value in expected.items():
            _assert_numeric_equal(
                class_report["comparison_to_baseline"][field],
                value,
                f"{label}.{class_name}.{field}",
            )
            checks += 1
    return checks


def _definition_hash(config: dict[str, Any]) -> str:
    frozen = {
        "candidate_order": config["candidate_order"],
        "candidate_definitions": config["candidate_definitions"],
        "thresholds": config["thresholds"],
        "implementation_contract": config["implementation_contract"],
        "forbidden_candidate_inputs": config["forbidden_candidate_inputs"],
    }
    return canonical_sha(frozen)


def _validate_component_decisions(
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    candidate_ids = list(config["candidate_order"])
    expected_count = int(config["input_contract"]["expected_component_count"])
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    forbidden = set(config["forbidden_candidate_inputs"])
    checks = 0
    seen: set[tuple[str, str]] = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in candidate_ids:
            raise ValueError(f"unexpected candidate decision: {candidate_id}")
        key = (candidate_id, str(row["component_id"]))
        if key in seen:
            raise ValueError("duplicate candidate/component decision")
        seen.add(key)
        gate_input_fields = set(str(value) for value in row["gate_input_fields"])
        if forbidden & gate_input_fields:
            raise ValueError(
                "candidate callable leaked forbidden fields: "
                f"{sorted(forbidden & gate_input_fields)}"
            )
        raw = int(row["raw_pixels"])
        causal = int(row["causal_supported_pixels"])
        noncausal = int(row["noncausal_pixels"])
        kept = int(row["kept_pixels"])
        rejected = int(row["rejected_pixels"])
        if raw != int(row["raw_area_pixels"]) or raw != causal + noncausal:
            raise ValueError("component evidence does not partition raw pixels")
        if raw != kept + rejected:
            raise ValueError("component decision does not partition raw pixels")
        expected_action = "REJECT" if kept == 0 else "KEEP" if kept == raw else "PARTIAL"
        if row["action"] != expected_action:
            raise ValueError("component action disagrees with pixel counts")
        low = bool(row["low_confidence"])
        small = bool(row["small_fragment"])
        upper = bool(row["intersects_upper_head_band"])
        predicted_class = str(row["predicted_class"])
        if candidate_id == "CLASS_CONDITIONED_MULTI_NEGATIVE":
            expected_rejected = (
                noncausal
                if predicted_class == "obstacle" and low and (small or upper)
                else raw
                if predicted_class == "boundary_step_curb" and low and small
                else 0
            )
        else:  # pragma: no cover - guarded above
            raise AssertionError(candidate_id)
        if rejected != expected_rejected:
            raise ValueError(
                f"candidate predicate mismatch for {candidate_id}/{row['component_id']}"
            )
        by_candidate[candidate_id].append(row)
        checks += 7
    outcomes: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        candidate_rows = by_candidate[candidate_id]
        if len(candidate_rows) != expected_count:
            raise ValueError(f"candidate decision count mismatch: {candidate_id}")
        actions = Counter(str(row["action"]) for row in candidate_rows)
        outcomes[candidate_id] = {
            "raw_component_count": len(candidate_rows),
            "fully_retained": actions["KEEP"],
            "partially_retained": actions["PARTIAL"],
            "removed": actions["REJECT"],
            "split_source_components": sum(
                int(row["post_fragment_count"]) > 1 for row in candidate_rows
            ),
            "post_fragment_count_from_raw_components": sum(
                int(row["post_fragment_count"]) for row in candidate_rows
            ),
        }
        checks += 1
    return checks, outcomes


def _validate_terminal(result: dict[str, Any], config: dict[str, Any]) -> int:
    rules = config["decision_rules"]
    sufficient: list[str] = []
    for candidate_id in config["candidate_order"]:
        report = result["arms"][candidate_id]
        overall = report["overall"]
        sessions = report["by_session_id"]
        comparable = [
            metric["comparison_to_baseline"]["recall_retention"]
            for metric in sessions.values()
            if metric["comparison_to_baseline"]["recall_retention"] is not None
        ]
        values = {
            "false_positive_reduction": overall["comparison_to_baseline"][
                "false_positive_reduction"
            ],
            "overall_recall_retention": overall["comparison_to_baseline"][
                "recall_retention"
            ],
            "minimum_session_recall_retention": min(comparable)
            if comparable
            else None,
            "boundary_step_curb_recall_retention": overall["classes"][
                "boundary_step_curb"
            ]["comparison_to_baseline"]["recall_retention"],
            "obstacle_recall_retention": overall["classes"]["obstacle"][
                "comparison_to_baseline"
            ]["recall_retention"],
        }
        thresholds = {
            "false_positive_reduction": rules["minimum_false_positive_reduction"],
            "overall_recall_retention": rules[
                "minimum_overall_recall_retention"
            ],
            "minimum_session_recall_retention": rules[
                "minimum_session_recall_retention"
            ],
            "boundary_step_curb_recall_retention": rules[
                "minimum_boundary_step_curb_recall_retention"
            ],
            "obstacle_recall_retention": rules[
                "minimum_obstacle_recall_retention"
            ],
        }
        passed = all(
            values[key] is not None and float(values[key]) >= float(threshold)
            for key, threshold in thresholds.items()
        )
        if bool(result["candidate_decisions"][candidate_id]["sufficient"]) != passed:
            raise ValueError(f"candidate sufficient flag mismatch: {candidate_id}")
        if passed:
            sufficient.append(candidate_id)
    expected_terminal = (
        rules["supported_terminal"] if sufficient else rules["unsupported_terminal"]
    )
    if result["decision"]["terminal"] != expected_terminal:
        raise ValueError("terminal does not follow frozen decision rules")
    if result["decision"]["sufficient_candidate_ids"] != sufficient:
        raise ValueError("sufficient candidate list mismatch")
    return len(config["candidate_order"]) + 2


def validate(
    *,
    repo_root: Path,
    config_path: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    result_path = evidence_root / "result.json"
    frame_path = evidence_root / "frame_metrics.jsonl"
    decision_path = evidence_root / "component_decisions.jsonl"
    result = read_json(result_path)
    frames = read_jsonl(frame_path)
    decisions = read_jsonl(decision_path)
    errors: list[str] = []
    checks = 0
    try:
        if result.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("unexpected result protocol_id")
        expected_candidates = ["CLASS_CONDITIONED_MULTI_NEGATIVE"]
        if config["candidate_order"] != expected_candidates:
            raise ValueError("config candidate order drifted")
        if result["candidate_order"] != expected_candidates:
            raise ValueError("result candidate order drifted")
        if result["candidate_definition_sha256"] != _definition_hash(config):
            raise ValueError("candidate definition hash mismatch")
        checks += 4
        if (
            result.get("selection_status")
            != "NO_SELECTION_SINGLE_FROZEN_CANDIDATE_REPORTED"
            or result.get("confirmation") != "NOT_ACTIVATED"
            or result.get("drives_alerts") is not False
        ):
            raise ValueError("authority boundary drifted")
        checks += 3
        if sha256_file(frame_path) != result["output_files"]["frame_metrics.jsonl"][
            "sha256"
        ]:
            raise ValueError("frame metrics output SHA mismatch")
        if sha256_file(decision_path) != result["output_files"][
            "component_decisions.jsonl"
        ]["sha256"]:
            raise ValueError("component decisions output SHA mismatch")
        checks += 2
        if len(frames) != int(config["input_contract"]["expected_frame_count"]):
            raise ValueError("frame row count mismatch")
        if len({row["view_row_id"] for row in frames}) != len(frames):
            raise ValueError("duplicate output frame")
        observed_session_counts = Counter(str(row["session_id"]) for row in frames)
        expected_session_counts = Counter(
            {
                str(key): int(value)
                for key, value in config["input_contract"][
                    "expected_session_frame_counts"
                ].items()
            }
        )
        if observed_session_counts != expected_session_counts:
            raise ValueError("output session membership mismatch")
        checks += 3

        candidate_classes = list(config["candidate_classes"])
        arm_ids = [
            BASELINE_ID,
            *REFERENCE_IDS,
            *config["candidate_order"],
        ]
        for arm_id in arm_ids:
            checks += _validate_aggregate(
                frames,
                arm_id,
                candidate_classes,
                result["arms"][arm_id]["overall"],
                f"{arm_id}.overall",
            )
            for field in ("session_id", "role", "scene_bucket"):
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in frames:
                    grouped[str(row[field])].append(row)
                reported_groups = result["arms"][arm_id][f"by_{field}"]
                if set(grouped) != set(reported_groups):
                    raise ValueError(f"{arm_id} group membership mismatch: {field}")
                for group_id, group_rows in grouped.items():
                    checks += _validate_aggregate(
                        group_rows,
                        arm_id,
                        candidate_classes,
                        reported_groups[group_id],
                        f"{arm_id}.{field}.{group_id}",
                    )
                    baseline_report = result["arms"][BASELINE_ID][f"by_{field}"][
                        group_id
                    ]
                    checks += _validate_comparison(
                        reported_groups[group_id],
                        baseline_report,
                        f"{arm_id}.{field}.{group_id}",
                    )
            checks += _validate_comparison(
                result["arms"][arm_id]["overall"],
                result["arms"][BASELINE_ID]["overall"],
                f"{arm_id}.overall",
            )

        component_checks, outcomes = _validate_component_decisions(decisions, config)
        checks += component_checks
        if canonical_sha(outcomes) != canonical_sha(result["component_outcomes"]):
            raise ValueError("component outcome aggregation mismatch")
        checks += 1

        folds = result["held_out_stress"]["folds"]
        if len(folds) != len(expected_session_counts):
            raise ValueError("held-out fold count mismatch")
        seen_sessions: set[str] = set()
        for fold in folds:
            session_id = str(fold["held_out_session_id"])
            seen_sessions.add(session_id)
            if (
                fold["fit_used"]
                or fold["training_used"]
                or fold["candidate_selection_used"]
                or fold["candidate_definition_sha256"]
                != result["candidate_definition_sha256"]
            ):
                raise ValueError("held-out fold authority drifted")
            for candidate_id in config["candidate_order"]:
                held = fold["arms"][candidate_id]["held_out"]
                direct = result["arms"][candidate_id]["by_session_id"][session_id]
                if canonical_sha(held) != canonical_sha(direct):
                    raise ValueError("held-out metrics differ from direct session metrics")
                if (
                    fold["arms"][candidate_id][
                        "held_out_equals_direct_session_metrics"
                    ]
                    is not True
                ):
                    raise ValueError("held-out identity flag missing")
                checks += 2
            checks += 1
        if seen_sessions != set(expected_session_counts):
            raise ValueError("held-out sessions do not exhaust frozen membership")
        checks += 1
        checks += _validate_terminal(result, config)
    except Exception as exc:  # validation must write a durable fail result
        errors.append(str(exc))

    return {
        "schema_version": (
            "blindassist.dual_loop_segmentation_conditional_gating.validation.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "status": "VALID" if not errors else "INVALID",
        "checks_completed": checks,
        "errors": errors,
        "evidence": {
            "result_path": str(result_path.relative_to(repo_root)).replace("\\", "/"),
            "result_sha256": sha256_file(result_path),
            "frame_metrics_sha256": sha256_file(frame_path),
            "component_decisions_sha256": sha256_file(decision_path),
        },
        "authority": "DEVELOPMENT_AGGREGATION_VALIDATION_ONLY",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = _resolve(repo_root, args.config)
    evidence_root = _resolve(repo_root, args.evidence_root)
    output_path = _resolve(repo_root, args.output)
    validation = validate(
        repo_root=repo_root,
        config_path=config_path,
        evidence_root=evidence_root,
    )
    _write_json(output_path, validation)
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    return 0 if validation["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
