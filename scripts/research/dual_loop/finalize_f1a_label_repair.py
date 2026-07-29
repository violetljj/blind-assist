#!/usr/bin/env python3
"""Finalize adjudicated F-1A labels and evaluate the frozen data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from compare_f1a_label_reviews import ReviewError, duration_by_input, interval


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_disagreement_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    shorthand = re.fullmatch(r"D(\d{3})", value)
    return f"DISAGREEMENT-{shorthand.group(1)}" if shorthand else value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def hull(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        "start": min(float(left["start"]), float(right["start"])),
        "end": max(float(left["end"]), float(right["end"])),
    }


def intersection(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        "start": max(float(left["start"]), float(right["start"])),
        "end": min(float(left["end"]), float(right["end"])),
    }


def consensus_item(
    agreement: dict[str, Any],
    *,
    review_hashes: list[str],
) -> dict[str, Any]:
    left = agreement["review_a"]
    right = agreement["review_b"]
    provenance = {
        "method": "model_consensus",
        "review_sha256s": review_hashes,
        "source_item_ids": [left["item_id"], right["item_id"]],
    }
    if agreement["kind"] == "POSITIVE_EVENT":
        return {
            "item_kind": "positive_event",
            "input_id": left["input_id"],
            "event_type": left["event_type"],
            "onset_interval_seconds": hull(
                left["onset_interval_seconds"], right["onset_interval_seconds"]
            ),
            "alertable_start_interval_seconds": hull(
                left["alertable_start_interval_seconds"],
                right["alertable_start_interval_seconds"],
            ),
            "end_or_clear_interval_seconds": hull(
                left["end_or_clear_interval_seconds"],
                right["end_or_clear_interval_seconds"],
            ),
            "region": left["region"],
            "confidence": min(float(left["confidence"]), float(right["confidence"])),
            "truth_provenance": provenance,
            "notes": "Conservative interval hull of two isolated RGB reviews.",
        }
    if agreement["kind"] == "NEGATIVE_WINDOW":
        bounds = intersection(
            left["window_interval_seconds"], right["window_interval_seconds"]
        )
        if bounds["end"] - bounds["start"] < 2.0:
            raise ReviewError("consensus negative intersection is shorter than 2 seconds")
        return {
            "item_kind": "negative_window",
            "input_id": left["input_id"],
            "negative_type": left["negative_type"],
            "window_interval_seconds": bounds,
            "should_alert": False,
            "region": left.get("region")
            if left.get("region") == right.get("region")
            else None,
            "confidence": min(float(left["confidence"]), float(right["confidence"])),
            "truth_provenance": provenance,
            "notes": "Intersection of two isolated RGB-review negative windows.",
        }
    raise ReviewError(f"unsupported agreement kind: {agreement['kind']}")


def validate_adjudicator(
    adjudication: dict[str, Any],
    *,
    comparison: dict[str, Any],
    comparison_sha256: str,
) -> None:
    if (
        adjudication.get("schema")
        != "blindassist_dual_loop_f1a_label_adjudication_v1"
    ):
        raise ReviewError("adjudication schema mismatch")
    for key in (
        "reviewer_id",
        "provider",
        "model",
        "model_version",
        "review_run_id",
        "workflow_id",
    ):
        if not isinstance(adjudication.get(key), str) or not adjudication[key].strip():
            raise ReviewError(f"adjudication.{key} must be non-empty")
    if adjudication.get("reviewer_role") not in {"gpt_adjudicator", "codex_adjudicator"}:
        raise ReviewError("adjudicator role is invalid")
    if adjudication.get("reviewer_type") != "ai_model":
        raise ReviewError("adjudicator type must be ai_model")
    if adjudication["reviewer_id"] in comparison["reviewer_ids"]:
        raise ReviewError("adjudicator must be a fresh reviewer identity")
    if adjudication.get("input_sha256") != comparison["input_sha256"]:
        raise ReviewError("adjudication input SHA mismatch")
    if adjudication.get("prompt_sha256") != comparison["prompt_sha256"]:
        raise ReviewError("adjudication prompt SHA mismatch")
    if adjudication.get("comparison_sha256") != comparison_sha256:
        raise ReviewError("adjudication comparison SHA mismatch")
    if adjudication.get("input_review_sha256s") != [
        comparison["review_a_sha256"],
        comparison["review_b_sha256"],
    ]:
        raise ReviewError("adjudication review hash binding mismatch")
    if (
        adjudication.get("isolated_context") is not True
        or adjudication.get("candidate_output_visible") is not False
    ):
        raise ReviewError("adjudicator is not isolated and candidate-blind")
    confidence = adjudication.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.65 <= float(confidence) <= 1.0
    ):
        raise ReviewError("adjudication confidence is invalid")
    if adjudication.get("abstained") is not False:
        raise ReviewError("adjudicator abstained at receipt level")


def validate_resolved_item(
    item: dict[str, Any],
    *,
    spec: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    durations = duration_by_input(manifest)
    input_id = item.get("input_id")
    if input_id not in durations:
        raise ReviewError("resolved item input is not frozen")
    confidence = item.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or float(confidence) < 0.65
    ):
        raise ReviewError("resolved item confidence is below 0.65")
    if item.get("item_kind") == "positive_event":
        if item.get("event_type") not in spec["label_contract"]["positive_event_types"]:
            raise ReviewError("resolved positive type is invalid")
        if item.get("region") not in spec["label_contract"]["regions"]:
            raise ReviewError("resolved positive region is invalid")
        bounds: dict[str, tuple[float, float]] = {}
        for key in (
            "onset_interval_seconds",
            "alertable_start_interval_seconds",
            "end_or_clear_interval_seconds",
        ):
            bounds[key] = interval(
                item.get(key), f"resolved_item.{key}", durations[input_id]
            )
        if not (
            bounds["onset_interval_seconds"][0]
            <= bounds["alertable_start_interval_seconds"][0]
            <= bounds["end_or_clear_interval_seconds"][1]
        ):
            raise ReviewError("resolved positive intervals are not causal")
    elif item.get("item_kind") == "negative_window":
        if item.get("negative_type") not in spec["label_contract"]["negative_window_types"]:
            raise ReviewError("resolved negative type is invalid")
        if item.get("should_alert") is not False:
            raise ReviewError("resolved negative should_alert must be false")
        start, end = interval(
            item.get("window_interval_seconds"),
            "resolved_item.window_interval_seconds",
            durations[input_id],
        )
        if (
            end - start
            < float(spec["label_contract"]["minimum_natural_negative_window_seconds"])
        ):
            raise ReviewError("resolved negative window is too short")
    else:
        raise ReviewError("resolved item kind is invalid")


def materialize_items(
    comparison: dict[str, Any],
    *,
    adjudication: dict[str, Any] | None,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    comparison_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    review_hashes = [
        comparison["review_a_sha256"],
        comparison["review_b_sha256"],
    ]
    items = [
        consensus_item(agreement, review_hashes=review_hashes)
        for agreement in comparison["agreements"]
    ]
    quarantines: list[dict[str, Any]] = []
    if comparison["disagreements"]:
        if adjudication is None:
            raise ReviewError("disagreements require adjudication")
        validate_adjudicator(
            adjudication,
            comparison=comparison,
            comparison_sha256=comparison_sha256,
        )
        resolutions = adjudication.get("resolutions")
        if not isinstance(resolutions, list):
            raise ReviewError("adjudication.resolutions must be an array")
        expected_ids = {
            item["disagreement_id"] for item in comparison["disagreements"]
        }
        resolution_ids = [
            canonical_disagreement_id(resolution.get("disagreement_id"))
            for resolution in resolutions
            if isinstance(resolution, dict)
        ]
        if len(resolution_ids) != len(resolutions) or len(set(resolution_ids)) != len(
            resolution_ids
        ):
            raise ReviewError("adjudication resolution identities are duplicated or invalid")
        by_id = {
            canonical_disagreement_id(resolution.get("disagreement_id")): resolution
            for resolution in resolutions
            if isinstance(resolution, dict)
        }
        if set(by_id) != expected_ids:
            raise ReviewError("adjudication resolution identities mismatch")
        for disagreement in comparison["disagreements"]:
            resolution = by_id[disagreement["disagreement_id"]]
            disposition = resolution.get("disposition")
            if disposition == "quarantine":
                quarantines.append(
                    {
                        "disagreement_id": disagreement["disagreement_id"],
                        "reason": resolution.get("rationale", "adjudicator quarantine"),
                        "source_disagreement": disagreement,
                    }
                )
                continue
            if disposition != "accept":
                raise ReviewError("adjudication disposition must be accept or quarantine")
            resolved = resolution.get("resolved_item")
            if not isinstance(resolved, dict):
                raise ReviewError("accepted resolution requires resolved_item")
            validate_resolved_item(resolved, spec=spec, manifest=manifest)
            resolved = dict(resolved)
            resolved["truth_provenance"] = {
                "method": "independent_ai_adjudicator",
                "adjudicator_id": adjudication["reviewer_id"],
                "adjudication_sha256": None,
                "comparison_sha256": comparison_sha256,
                "disagreement_id": disagreement["disagreement_id"],
                "review_sha256s": review_hashes,
            }
            items.append(resolved)
    elif adjudication is not None:
        raise ReviewError("adjudication provided without disagreements")
    return items, quarantines


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return min(left[1], right[1]) - max(left[0], right[0])


def validate_cross_item_consistency(items: list[dict[str, Any]]) -> None:
    intervals: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)
    for index, item in enumerate(items):
        item_id = f"F1A-ITEM-{index + 1:03d}"
        if item["item_kind"] == "positive_event":
            intervals[item["input_id"]].append(
                (
                    float(item["onset_interval_seconds"]["start"]),
                    float(item["end_or_clear_interval_seconds"]["end"]),
                    item_id,
                    "positive_event",
                )
            )
        else:
            intervals[item["input_id"]].append(
                (
                    float(item["window_interval_seconds"]["start"]),
                    float(item["window_interval_seconds"]["end"]),
                    item_id,
                    "negative_window",
                )
            )
    for input_id, input_intervals in intervals.items():
        ordered = sorted(input_intervals)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if right[0] >= left[1]:
                    break
                if overlap(left[:2], right[:2]) > 0:
                    raise ReviewError(
                        f"overlapping natural items in {input_id}: "
                        f"{left[2]} ({left[3]}) vs {right[2]} ({right[3]})"
                    )


def assign_records(
    items: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    input_meta = {item["input_id"]: item for item in manifest["inputs"]}
    records: list[dict[str, Any]] = []
    positive_index = 0
    negative_index = 0
    for item in sorted(
        items,
        key=lambda value: (
            value["input_id"],
            float(
                value.get("onset_interval_seconds", value.get("window_interval_seconds"))[
                    "start"
                ]
            ),
            value["item_kind"],
        ),
    ):
        meta = input_meta[item["input_id"]]
        record = dict(item)
        if item["item_kind"] == "positive_event":
            positive_index += 1
            record["event_id"] = f"F1A-P-{positive_index:03d}"
            record["positive_or_negative"] = "POSITIVE"
            record["should_alert"] = True
        else:
            negative_index += 1
            record["negative_window_id"] = f"F1A-N-{negative_index:03d}"
            record["positive_or_negative"] = "NEGATIVE"
        record.update(
            {
                "source_id": meta["source_id"],
                "session_id": meta["session_id"],
                "sequence_id": meta["session_id"],
                "parent_capture_id": meta["parent_capture_id"],
                "role": meta["role"],
                "clock_or_frame_order_basis": "source timestamp or bound video PTS",
                "outcome_access_state_after_repair": (
                    "CONTENT_INSPECTED_FOR_F1A_LABEL_REPAIR"
                    if meta["role"] == "DECISION"
                    else meta["outcome_access_state_before_repair"]
                ),
            }
        )
        records.append(record)
    return records


def evaluate_gate(
    records: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    positives = [record for record in records if record["item_kind"] == "positive_event"]
    negatives = [record for record in records if record["item_kind"] == "negative_window"]
    positive_sessions = {record["session_id"] for record in positives}
    negative_categories = Counter(record["negative_type"] for record in negatives)
    roles = Counter(item["role"] for item in manifest["inputs"])
    decision_ids = {
        item["input_id"] for item in manifest["inputs"] if item["role"] == "DECISION"
    }
    decision_positive = {
        record["input_id"] for record in positives if record["role"] == "DECISION"
    }
    decision_negative = {
        record["input_id"] for record in negatives if record["role"] == "DECISION"
    }
    gate = spec["ready_gate"]
    checks = {
        "independent_capture_sessions": len(
            {item["parent_capture_id"] for item in manifest["inputs"]}
        )
        >= gate["independent_capture_sessions_min"],
        "positive_events": len(positives) >= gate["positive_events_min"],
        "positive_sessions": len(positive_sessions) >= gate["positive_sessions_min"],
        "negative_windows": len(negatives) >= gate["negative_windows_min"],
        "negative_categories": sum(
            count >= gate["negative_windows_per_category_min"]
            for count in negative_categories.values()
        )
        >= gate["negative_categories_min"],
        "development_role": roles["DEVELOPMENT"] == gate["development_sessions"],
        "decision_roles": roles["DECISION"] == gate["decision_sessions"],
        "each_decision_has_positive": decision_positive == decision_ids,
        "each_decision_has_negative": decision_negative == decision_ids,
    }
    return {
        "checks": checks,
        "counts": {
            "independent_capture_sessions": len(
                {item["parent_capture_id"] for item in manifest["inputs"]}
            ),
            "positive_events": len(positives),
            "positive_sessions": len(positive_sessions),
            "negative_windows": len(negatives),
            "negative_category_counts": dict(sorted(negative_categories.items())),
            "development_sessions": roles["DEVELOPMENT"],
            "decision_sessions": roles["DECISION"],
            "decision_with_positive": sorted(decision_positive),
            "decision_with_negative": sorted(decision_negative),
        },
        "terminal": "READY" if all(checks.values()) else "HOLD_DATA",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ledger.exists() or args.validation.exists():
        raise FileExistsError("formal ledger/validation output already exists")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    adjudication = (
        json.loads(args.adjudication.read_text(encoding="utf-8"))
        if args.adjudication
        else None
    )
    comparison_sha256 = sha256_file(args.comparison)
    items, quarantines = materialize_items(
        comparison,
        adjudication=adjudication,
        spec=spec,
        manifest=manifest,
        comparison_sha256=comparison_sha256,
    )
    validate_cross_item_consistency(items)
    records = assign_records(items, manifest=manifest)
    adjudication_sha256 = (
        sha256_file(args.adjudication) if args.adjudication else None
    )
    if adjudication_sha256 is not None:
        for record in records:
            provenance = record.get("truth_provenance")
            if (
                isinstance(provenance, dict)
                and provenance.get("method") == "independent_ai_adjudicator"
            ):
                provenance["adjudication_sha256"] = adjudication_sha256
    gate = evaluate_gate(records, spec=spec, manifest=manifest)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    validation = {
        "schema": "blindassist_dual_loop_f1a_label_repair_validation_v1",
        "protocol_id": spec["protocol_id"],
        "spec_sha256": sha256_file(args.spec),
        "review_bundle_subject_sha256": manifest["bundle_subject_sha256"],
        "comparison_sha256": comparison_sha256,
        "adjudication_sha256": adjudication_sha256,
        "ledger_sha256": sha256_file(args.ledger),
        "quarantine_count": len(quarantines),
        "quarantines": quarantines,
        "data_protocol_status": "VALID",
        "candidate_output_visibility": False,
        **gate,
    }
    write_json(args.validation, validation)
    print(
        json.dumps(
            {
                "status": "PASS",
                "terminal": validation["terminal"],
                "counts": validation["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
