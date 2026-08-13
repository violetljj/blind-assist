#!/usr/bin/env python3
"""Replay one source-only clear-observability axis on consumed R11 evidence.

This is deliberately not a selector search.  It compares the frozen R9 rule
with exactly one candidate that changes only ``far_fraction_index`` from 0 to
2.  Labels are used only for Development evaluation; UNKNOWN is never a
negative and neither factor predictions nor the consumed R11 selection change.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r9_clear_runtime import clear_enrichment_fit


SCHEMA = "blindassist.taro.development.clear_observability_single_axis_replay.v1"
BASELINE_RULE = {
    "state_policy": "UNKNOWN_ONLY",
    "minimum_far_valid_anchor_count": 6,
    "maximum_far_valid_anchor_count": 1_000_000,
    "far_fraction_index": 0,
    "maximum_far_fraction": 0.0,
    "minimum_observed_support_points": 0,
    "require_query_receipt": True,
    "require_positive_obstacle_veto_false": True,
    "require_all_occupied_hits_false": True,
    "rule_id": "02CE016D6B0011F0",
}
CANDIDATE_RULE = {**BASELINE_RULE, "far_fraction_index": 2}
CANDIDATE_RULE["rule_id"] = adapter.canonical_sha256(
    {key: value for key, value in CANDIDATE_RULE.items() if key != "rule_id"}
)[:16]

STOP_RULE = {
    "minimum_candidate_clear_frame_recall": 0.90,
    "minimum_clear_frame_recall_gain": 0.30,
    "maximum_eligible_frame_multiplier": 2.0,
    "require_all_known_clear_parents_retained": True,
    "maximum_candidate_count": 1,
}


class ReplayError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_sealed(record: Mapping[str, Any], role: str) -> None:
    stored = record.get("content_sha256")
    payload = {key: value for key, value in record.items() if key != "content_sha256"}
    _require(isinstance(stored, str) and stored == adapter.canonical_sha256(payload), f"{role} seal mismatch")


def _validate_single_axis() -> None:
    differing = {
        key
        for key in set(BASELINE_RULE) | set(CANDIDATE_RULE)
        if BASELINE_RULE.get(key) != CANDIDATE_RULE.get(key)
    }
    _require(differing == {"far_fraction_index", "rule_id"}, "candidate is not a single-axis change")
    _require(BASELINE_RULE["far_fraction_index"] == 0 and CANDIDATE_RULE["far_fraction_index"] == 2, "unexpected axis")


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 12)


def _rule_metrics(
    rows: Sequence[tuple[tuple[str, str], str, Mapping[str, Any], str]],
    rule: Mapping[str, Any],
    clear_query_total: int,
    clear_frames: set[str],
    clear_parents: set[tuple[str, str]],
) -> dict[str, Any]:
    eligible_rows = [row for row in rows if clear_enrichment_fit.eligible(row[2], rule)]
    state_counts = Counter(row[3] for row in eligible_rows)
    eligible_frames = {row[1] for row in eligible_rows}
    eligible_parents = {row[0] for row in eligible_rows}
    eligible_clear_frames = {row[1] for row in eligible_rows if row[3] == "CLEAR_OBSERVED"}
    eligible_clear_parents = {row[0] for row in eligible_rows if row[3] == "CLEAR_OBSERVED"}
    return {
        "rule": dict(rule),
        "eligible_query_count": len(eligible_rows),
        "eligible_frame_count": len(eligible_frames),
        "eligible_parent_count": len(eligible_parents),
        "eligible_state_counts": {
            state: int(state_counts[state])
            for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
        },
        "clear_query_recall": _ratio(state_counts["CLEAR_OBSERVED"], clear_query_total),
        "clear_frame_recall": _ratio(len(eligible_clear_frames), len(clear_frames)),
        "eligible_frame_clear_precision": _ratio(len(eligible_clear_frames), len(eligible_frames)),
        "clear_parent_recall": _ratio(len(eligible_clear_parents), len(clear_parents)),
        "eligible_clear_frame_count": len(eligible_clear_frames),
        "eligible_clear_parent_count": len(eligible_clear_parents),
        "eligible_clear_parent_identities": [list(identity) for identity in sorted(eligible_clear_parents)],
    }


def evaluate(
    source_frames: Sequence[Mapping[str, Any]],
    label_frames: Sequence[Mapping[str, Any]],
    all_source_frames: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed baseline and one candidate on aligned frame records."""

    _validate_single_axis()
    _require(len(source_frames) == len(label_frames) > 0, "source/label frame count mismatch")
    rows: list[tuple[tuple[str, str], str, Mapping[str, Any], str]] = []
    clear_frames: set[str] = set()
    clear_parents: set[tuple[str, str]] = set()
    label_counts: Counter[str] = Counter()

    for source, label in zip(source_frames, label_frames, strict=True):
        _require(source["physical_frame_id"] == label["physical_frame_id"], "physical frame mismatch")
        identity = (str(source["parent_id"]), str(source["video_id"]))
        _require(identity == (str(label["parent_id"]), str(label["video_id"])), "parent identity mismatch")
        features = source["query_features"]
        labels = label["query_labels"]
        _require(len(features) == len(labels) == 9, "query cardinality mismatch")
        for feature, query_label in zip(features, labels, strict=True):
            state = str(query_label["state"])
            _require(feature["query_id"] == query_label["query_id"], "query alignment mismatch")
            _require(state in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}, "unexpected label state")
            label_counts[state] += 1
            rows.append((identity, str(source["physical_frame_id"]), feature, state))
            if state == "CLEAR_OBSERVED":
                clear_frames.add(str(source["physical_frame_id"]))
                clear_parents.add(identity)

    baseline = _rule_metrics(rows, BASELINE_RULE, label_counts["CLEAR_OBSERVED"], clear_frames, clear_parents)
    candidate = _rule_metrics(rows, CANDIDATE_RULE, label_counts["CLEAR_OBSERVED"], clear_frames, clear_parents)

    baseline_recall = float(baseline["clear_frame_recall"] or 0.0)
    candidate_recall = float(candidate["clear_frame_recall"] or 0.0)
    eligible_limit = int(STOP_RULE["maximum_eligible_frame_multiplier"] * baseline["eligible_frame_count"])
    checks = {
        "candidate_clear_frame_recall": candidate_recall >= STOP_RULE["minimum_candidate_clear_frame_recall"],
        "clear_frame_recall_gain": candidate_recall - baseline_recall >= STOP_RULE["minimum_clear_frame_recall_gain"],
        "eligible_frame_budget": candidate["eligible_frame_count"] <= eligible_limit,
        "all_known_clear_parents_retained": candidate["eligible_clear_parent_count"] == len(clear_parents),
        "single_candidate_only": STOP_RULE["maximum_candidate_count"] == 1,
    }

    landscape: dict[str, Any] | None = None
    if all_source_frames is not None:
        landscape = {}
        for name, rule in (("baseline", BASELINE_RULE), ("candidate", CANDIDATE_RULE)):
            frame_count = 0
            query_count = 0
            parents: set[tuple[str, str]] = set()
            for source in all_source_frames:
                eligible = [feature for feature in source["query_features"] if clear_enrichment_fit.eligible(feature, rule)]
                if eligible:
                    frame_count += 1
                    query_count += len(eligible)
                    parents.add((str(source["parent_id"]), str(source["video_id"])))
            landscape[name] = {
                "eligible_query_count": query_count,
                "eligible_frame_count": frame_count,
                "nonzero_parent_count": len(parents),
            }

    result = {
        "schema": SCHEMA,
        "analysis_role": "PROJECT_CONSUMED_DEVELOPMENT",
        "hypothesis": "The 4.0 m far-fraction slice is a higher-recall source-only clear-observability proxy than the frozen 2.5 m slice.",
        "baseline": baseline,
        "candidate": candidate,
        "label_state_counts": {
            state: int(label_counts[state])
            for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
        },
        "clear_physical_frame_count": len(clear_frames),
        "clear_parent_count": len(clear_parents),
        "clear_frame_recall_gain": round(candidate_recall - baseline_recall, 12),
        "eligible_frame_limit": eligible_limit,
        "stop_rule": dict(STOP_RULE),
        "stop_checks": checks,
        "development_candidate_passed": all(checks.values()),
        "all_48_source_only_landscape": landscape,
        "candidate_count": 1,
        "threshold_search": False,
        "factor_prediction_changed": False,
        "unknown_is_negative": False,
        "faro_payload_replay_reads": 0,
        "unselected_faro_reads": 0,
        "claim_ceiling": "Post-hoc Development ranking-proxy evidence only; no R11 rewrite, fresh confirmation, CLEAR output, deployment, product, default-App, or safety claim.",
    }
    result["content_sha256"] = adapter.canonical_sha256(result)
    return result


def _iter_lineage_paths(root: Path) -> Iterable[Path]:
    yield from sorted((root / "phase-a-lineage").glob("*/*/*.json.gz"), key=lambda path: path.as_posix())


def load_and_evaluate(phase_a_root: Path, phase_b_root: Path) -> dict[str, Any]:
    completion = _read_json(phase_b_root / "label-completion.json")
    _assert_sealed(completion, "label completion")
    label_paths = sorted((phase_b_root / "labels").glob("*/*/*.json.gz"), key=lambda path: path.as_posix())
    _require(len(label_paths) == completion["frame_count"] == 674, "label file count mismatch")

    labels: dict[str, dict[str, Any]] = {}
    for path in label_paths:
        record = _read_json_gz(path)
        _assert_sealed(record, "label")
        _require(record.get("unknown_is_negative") is False, "label UNKNOWN policy drift")
        physical_frame_id = str(record["physical_frame_id"])
        _require(physical_frame_id not in labels, "duplicate label frame")
        labels[physical_frame_id] = record

    selected_sources: dict[str, dict[str, Any]] = {}
    all_sources: list[dict[str, Any]] = []
    for path in _iter_lineage_paths(phase_a_root):
        lineage = _read_json_gz(path)
        _assert_sealed(lineage, "phase-a lineage")
        _require(lineage.get("faro_payload_read") is False, "Phase A FARO firewall drift")
        source = lineage["r7_source_frame_record"]
        _assert_sealed(source, "R7 source frame")
        all_sources.append(source)
        physical_frame_id = str(source["physical_frame_id"])
        if physical_frame_id in labels:
            label = labels[physical_frame_id]
            _require(label["source_frame_record_sha256"] == source["content_sha256"], "label/source lineage mismatch")
            selected_sources[physical_frame_id] = source

    _require(len(all_sources) == 1043, "all-48 source frame count mismatch")
    _require(set(selected_sources) == set(labels), "selected source coverage mismatch")
    ordered_ids = sorted(labels)
    result = evaluate(
        [selected_sources[frame_id] for frame_id in ordered_ids],
        [labels[frame_id] for frame_id in ordered_ids],
        all_source_frames=all_sources,
    )
    result["source_frame_count"] = len(all_sources)
    result["label_frame_count"] = len(labels)
    result["query_count"] = len(labels) * 9
    result["label_completion_content_sha256"] = completion["content_sha256"]
    result["content_sha256"] = adapter.canonical_sha256(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )
    return result


def _write_exclusive(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.canonical_json_bytes(record) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = load_and_evaluate(arguments.phase_a_root, arguments.phase_b_root)
    if arguments.output is not None:
        _write_exclusive(arguments.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if result["development_candidate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
