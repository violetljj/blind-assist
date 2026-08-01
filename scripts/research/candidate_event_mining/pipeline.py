"""Core contracts and deterministic transforms for candidate event mining.

The module consumes a model-agnostic, truth-free frame trace.  Model-specific
adapters (YOLO, segmentation, depth and HFTF) are deliberately outside this
file and only write normalized ``signals`` in the frame contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence


FRAME_SCHEMA = "blindassist_candidate_event_mining_frame_v1"
CANDIDATE_REPORT_SCHEMA = "blindassist_candidate_event_mining_candidate_report_v1"
BUNDLE_SCHEMA = "blindassist_candidate_event_mining_review_bundle_v1"
LUNA_REVIEW_SCHEMA = "blindassist_candidate_event_mining_luna_review_v1"
POOL_SCHEMA = "blindassist_candidate_event_mining_candidate_pool_v1"
PROJECT_INDEX_SCHEMA = "blindassist_candidate_event_mining_project_index_v1"

TARGETS: tuple[str, ...] = (
    "front_obstacle_approach",
    "crossing",
    "static_obstacle_approach",
    "step_or_drop",
    "parallel_curb",
    "doorframe_table_corner_tree_branch",
    "normal_passage_negative",
    "head_turn_or_jitter_negative",
    "dynamic_crowd",
    "yolo_miss_segmentation_or_depth_response",
    "segmentation_high_frequency_alert",
    "hftf_future_field_change",
)

DIRECT_SIGNAL_TARGETS: dict[str, tuple[str, ...]] = {
    "front_obstacle_approach": (
        "motion.front_approach",
        "depth.front_approach",
        "yolo.front_obstacle",
        "segmentation.front_risk",
    ),
    "crossing": ("motion.crossing", "object.crossing"),
    "static_obstacle_approach": (
        "motion.static_obstacle_approach",
        "depth.static_obstacle_approach",
    ),
    "step_or_drop": (
        "geometry.step_drop",
        "depth.step_drop",
        "segmentation.boundary_level_change",
    ),
    "parallel_curb": (
        "geometry.parallel_curb",
        "segmentation.parallel_curb",
    ),
    "doorframe_table_corner_tree_branch": (
        "object.doorframe",
        "object.table_corner",
        "object.tree_branch",
    ),
    "normal_passage_negative": ("context.normal_passage",),
    "head_turn_or_jitter_negative": (
        "motion.head_turn",
        "motion.jitter",
    ),
    "dynamic_crowd": ("motion.dynamic_crowd", "object.dynamic_crowd"),
    "hftf_future_field_change": (
        "hftf.future_field_change",
        "hftf.future_risk_field_delta",
    ),
}


class ContractError(ValueError):
    """Raised when a candidate-mining input or contract is not admissible."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ContractError(f"JSON object required at {path}:{line_number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def refuse_overwrite(path: Path) -> None:
    if path.exists():
        raise ContractError(f"refusing to overwrite existing output: {path}")


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    text = _require_string(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise ContractError(f"{field} must be a lowercase SHA-256")
    return text


def _finite_score(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ContractError(f"{field} must be finite and in [0, 1]")
    return result


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") != "blindassist_candidate_event_mining_contract_v1":
        raise ContractError("unexpected candidate event mining contract schema")
    if value.get("status") != "discovery":
        raise ContractError("candidate event mining must remain discovery-only")
    if value.get("research_lane") != "THESIS_DEVELOPMENT":
        raise ContractError("candidate event mining must use THESIS_DEVELOPMENT")
    if tuple(value.get("targets", ())) != TARGETS:
        raise ContractError("candidate taxonomy differs from the frozen contract")
    source_root = value.get("source_root")
    if source_root != r"F:\ba-data":
        raise ContractError("source_root must remain F:\\ba-data")

    windowing = value.get("windowing")
    if not isinstance(windowing, dict):
        raise ContractError("windowing is required")
    for key in ("pre_context_ms", "post_context_ms", "max_gap_ms", "max_gap_frames", "max_review_frames"):
        number = windowing.get(key)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ContractError(f"windowing.{key} must be a non-negative integer")
    if windowing.get("min_active_frames", 0) < 1:
        raise ContractError("windowing.min_active_frames must be positive")

    dedup = value.get("deduplication")
    if not isinstance(dedup, dict):
        raise ContractError("deduplication is required")
    if dedup.get("same_session_only") is not True:
        raise ContractError("deduplication must not merge independent sessions")
    for key in ("max_merge_gap_ms",):
        number = dedup.get(key)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ContractError(f"deduplication.{key} must be a non-negative integer")
    for key in ("min_evidence_jaccard", "max_score_delta"):
        _finite_score(dedup.get(key), f"deduplication.{key}")
    if dedup.get("max_duration_ratio", 0) < 1:
        raise ContractError("deduplication.max_duration_ratio must be >= 1")

    review = value.get("review")
    if not isinstance(review, dict):
        raise ContractError("review is required")
    if review.get("reviewer_role") != "luna_reader":
        raise ContractError("Luna review role must remain luna_reader")
    if review.get("candidate_output_visible") is not False:
        raise ContractError("candidate output must be hidden from Luna")
    if review.get("required_reviews") != 1:
        raise ContractError("discovery contract expects one Luna pass per candidate")
    _finite_score(review.get("minimum_confidence"), "review.minimum_confidence")
    if review.get("abstain_disposition") != "quarantine":
        raise ContractError("abstentions must quarantine the affected candidate")

    rules = value.get("rules")
    if not isinstance(rules, dict):
        raise ContractError("rules are required")
    for target in TARGETS:
        rule = rules.get(target)
        if not isinstance(rule, dict):
            raise ContractError(f"missing rule: {target}")
        _finite_score(rule.get("min_score"), f"rules.{target}.min_score")
        minimum = rule.get("min_active_frames")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ContractError(f"rules.{target}.min_active_frames must be positive")
    derived = value.get("derived_signals")
    if not isinstance(derived, dict):
        raise ContractError("derived_signals are required")
    for key in ("yolo_miss_evidence_min", "yolo_miss_min", "segmentation_alert_active_threshold", "segmentation_frequency_min"):
        _finite_score(derived.get(key), f"derived_signals.{key}")
    if any(bool(item) for item in value.get("authorization", {}).values()):
        raise ContractError("candidate mining contract cannot authorize downstream use")
    return dict(value)


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    value = read_json(path)
    validate_contract(value)
    return value, {"path": str(path.resolve()), "sha256": sha256_file(path)}


def validate_project_index(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") != PROJECT_INDEX_SCHEMA:
        raise ContractError("unexpected project index schema")
    if value.get("data_root") != r"F:\ba-data":
        raise ContractError("project index data_root must remain F:\\ba-data")
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise ContractError("project index sources must be a list")
    seen: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ContractError(f"project index source {index} must be an object")
        source_id = _require_string(source.get("source_id"), f"sources[{index}].source_id")
        if source_id in seen:
            raise ContractError(f"duplicate source_id: {source_id}")
        seen.add(source_id)
        _require_string(source.get("session_id"), f"sources[{index}].session_id")
        media_path = _require_string(source.get("media_path"), f"sources[{index}].media_path")
        _require_string(source.get("source_url"), f"sources[{index}].source_url")
        _require_string(source.get("retrieved_at_utc"), f"sources[{index}].retrieved_at_utc")
        _require_sha256(source.get("content_sha256"), f"sources[{index}].content_sha256")
        try:
            PureWindowsPath(media_path).relative_to(PureWindowsPath(value["data_root"]))
        except ValueError as error:
            raise ContractError(f"media_path escapes data_root for {source_id}") from error
        if source.get("retrieval_status") not in {"declared", "downloaded", "verified", "not_available"}:
            raise ContractError(f"invalid retrieval_status for {source_id}")
    return dict(value)


def normalize_frame(value: Mapping[str, Any], line_number: int = 0) -> dict[str, Any]:
    where = f" at line {line_number}" if line_number else ""
    if value.get("schema") != FRAME_SCHEMA:
        raise ContractError(f"unexpected frame schema{where}")
    source_id = _require_string(value.get("source_id"), f"source_id{where}")
    session_id = _require_string(value.get("session_id"), f"session_id{where}")
    frame_ref = _require_string(value.get("frame_ref"), f"frame_ref{where}")
    frame_index = value.get("frame_index")
    timestamp_ms = value.get("timestamp_ms")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ContractError(f"frame_index must be a non-negative integer{where}")
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, (int, float)) or not math.isfinite(float(timestamp_ms)) or timestamp_ms < 0:
        raise ContractError(f"timestamp_ms must be a finite non-negative number{where}")
    signals = value.get("signals")
    if not isinstance(signals, dict) or not signals:
        raise ContractError(f"signals must be a non-empty object{where}")
    normalized_signals: dict[str, float] = {}
    for key, score in signals.items():
        signal_key = _require_string(key, f"signals key{where}")
        normalized_signals[signal_key] = _finite_score(score, f"signals.{signal_key}{where}")
    result: dict[str, Any] = {
        "schema": FRAME_SCHEMA,
        "source_id": source_id,
        "session_id": session_id,
        "frame_index": frame_index,
        "timestamp_ms": int(round(float(timestamp_ms))),
        "frame_ref": frame_ref,
        "signals": normalized_signals,
    }
    if value.get("frame_sha256") is not None:
        result["frame_sha256"] = _require_sha256(value.get("frame_sha256"), f"frame_sha256{where}")
    if value.get("source_manifest_sha256") is not None:
        result["source_manifest_sha256"] = _require_sha256(
            value.get("source_manifest_sha256"), f"source_manifest_sha256{where}"
        )
    return result


def normalize_frames(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [normalize_frame(value, index + 1) for index, value in enumerate(values)]
    rows.sort(key=lambda row: (row["source_id"], row["session_id"], row["frame_index"]))
    seen: set[tuple[str, str, int]] = set()
    previous_timestamp: dict[tuple[str, str], int] = {}
    for row in rows:
        identity = (row["source_id"], row["session_id"], row["frame_index"])
        if identity in seen:
            raise ContractError(f"duplicate frame identity: {identity}")
        seen.add(identity)
        group = identity[:2]
        previous = previous_timestamp.get(group)
        if previous is not None and row["timestamp_ms"] <= previous:
            raise ContractError(f"timestamps must increase within session: {group}")
        previous_timestamp[group] = row["timestamp_ms"]
    return rows


def _signal_max(signals: Mapping[str, float], keys: Sequence[str]) -> tuple[float, list[str]]:
    available = [(key, float(signals[key])) for key in keys if key in signals]
    if not available:
        return 0.0, []
    maximum = max(score for _key, score in available)
    return maximum, sorted(key for key, score in available if score == maximum)


def _score_sequence(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    derived = contract["derived_signals"]
    active_threshold = float(derived["segmentation_alert_active_threshold"])
    frequency_window = max(2, int(contract["windowing"].get("frequency_window_frames", 5)))
    segmentation_alert = [
        float(row["signals"].get("segmentation.alert", 0.0)) >= active_threshold
        for row in rows
    ]
    frequency_scores: list[float] = []
    for index in range(len(rows)):
        start = max(0, index - frequency_window + 1)
        sample = segmentation_alert[start : index + 1]
        transitions = sum(left != right for left, right in zip(sample, sample[1:]))
        frequency_scores.append(transitions / max(1, len(sample) - 1))

    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        signals = row["signals"]
        for target, keys in DIRECT_SIGNAL_TARGETS.items():
            score, used = _signal_max(signals, keys)
            scores[target] = score
            evidence[target] = used

        evidence_score, evidence_keys = _signal_max(
            signals,
            (
                "segmentation.risk",
                "segmentation.front_risk",
                "segmentation.boundary_level_change",
                "depth.approach",
                "depth.step_drop",
            ),
        )
        miss_score = max(
            float(signals.get("yolo.miss", 0.0)),
            1.0 - float(signals.get("yolo.coverage", 1.0)),
        )
        if evidence_score >= float(derived["yolo_miss_evidence_min"]) and miss_score >= float(derived["yolo_miss_min"]):
            scores["yolo_miss_segmentation_or_depth_response"] = min(evidence_score, miss_score)
            evidence["yolo_miss_segmentation_or_depth_response"] = sorted(
                set(evidence_keys + ["yolo.miss"] if "yolo.miss" in signals else evidence_keys)
            )
        else:
            scores["yolo_miss_segmentation_or_depth_response"] = 0.0
            evidence["yolo_miss_segmentation_or_depth_response"] = []

        direct_frequency, direct_frequency_keys = _signal_max(
            signals, ("segmentation.high_frequency_alert",)
        )
        frequency_score = max(direct_frequency, frequency_scores[index])
        if frequency_score < float(derived["segmentation_frequency_min"]):
            frequency_score = 0.0
        scores["segmentation_high_frequency_alert"] = frequency_score
        evidence["segmentation_high_frequency_alert"] = sorted(
            set(direct_frequency_keys + (["segmentation.alert"] if frequency_scores[index] > direct_frequency else []))
        )
        scored.append({"row": row, "scores": scores, "evidence": evidence})
    return scored


def _sample_frame_indexes(indexes: Sequence[int], maximum: int, peak_index: int) -> list[int]:
    if len(indexes) <= maximum:
        return list(indexes)
    selected = {indexes[0], indexes[-1], peak_index}
    remaining = max(0, maximum - len(selected))
    if remaining:
        stride = (len(indexes) - 1) / (remaining + 1)
        for offset in range(1, remaining + 1):
            selected.add(indexes[round(offset * stride)])
    return sorted(selected)[:maximum]


def _candidate_id(run_id: str, fields: Mapping[str, Any]) -> str:
    identity = {
        "run_id": run_id,
        "source_id": fields["source_id"],
        "session_id": fields["session_id"],
        "trigger_type": fields["trigger_type"],
        "start_timestamp_ms": fields["start_timestamp_ms"],
        "end_timestamp_ms": fields["end_timestamp_ms"],
        "peak_frame_index": fields["peak_frame_index"],
    }
    return "cemw-" + canonical_json_sha256(identity)[:20]


def extract_candidate_windows(
    frames: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    run_id: str,
) -> list[dict[str, Any]]:
    if not run_id.strip():
        raise ContractError("run_id must be non-empty")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in frames:
        groups[(row["source_id"], row["session_id"])].append(row)
    windowing = contract["windowing"]
    candidates: list[dict[str, Any]] = []
    for (source_id, session_id), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["frame_index"])
        scored = _score_sequence(rows, contract)
        for target in TARGETS:
            threshold = float(contract["rules"][target]["min_score"])
            min_active = int(contract["rules"][target]["min_active_frames"])
            active = [index for index, item in enumerate(scored) if item["scores"][target] >= threshold]
            if not active:
                continue
            segments: list[list[int]] = []
            start = previous = active[0]
            for index in active[1:]:
                timestamp_gap = rows[index]["timestamp_ms"] - rows[previous]["timestamp_ms"]
                missing_frames = index - previous - 1
                if timestamp_gap > int(windowing["max_gap_ms"]) or missing_frames > int(windowing["max_gap_frames"]):
                    segments.append(list(range(start, previous + 1)))
                    start = index
                previous = index
            segments.append(list(range(start, previous + 1)))

            for segment in segments:
                active_indexes = [index for index in segment if index in set(active)]
                if len(active_indexes) < min_active:
                    continue
                segment_scores = [scored[index]["scores"][target] for index in active_indexes]
                peak_local = max(range(len(active_indexes)), key=lambda pos: segment_scores[pos])
                peak_index = active_indexes[peak_local]
                start_timestamp = rows[segment[0]]["timestamp_ms"]
                end_timestamp = rows[segment[-1]]["timestamp_ms"]
                context_start_time = start_timestamp - int(windowing["pre_context_ms"])
                context_end_time = end_timestamp + int(windowing["post_context_ms"])
                context_indexes = [
                    index
                    for index, row in enumerate(rows)
                    if context_start_time <= row["timestamp_ms"] <= context_end_time
                ]
                review_indexes = _sample_frame_indexes(
                    context_indexes,
                    int(windowing["max_review_frames"]),
                    peak_index,
                )
                evidence_keys = sorted(
                    {
                        key
                        for index in active_indexes
                        for key in scored[index]["evidence"][target]
                    }
                )
                evidence_channels = sorted({key.split(".", 1)[0] for key in evidence_keys})
                base: dict[str, Any] = {
                    "schema": CANDIDATE_REPORT_SCHEMA,
                    "source_id": source_id,
                    "session_id": session_id,
                    "trigger_type": target,
                    "start_frame_index": rows[segment[0]]["frame_index"],
                    "end_frame_index": rows[segment[-1]]["frame_index"],
                    "start_timestamp_ms": start_timestamp,
                    "end_timestamp_ms": end_timestamp,
                    "context_start_timestamp_ms": rows[context_indexes[0]]["timestamp_ms"],
                    "context_end_timestamp_ms": rows[context_indexes[-1]]["timestamp_ms"],
                    "peak_frame_index": rows[peak_index]["frame_index"],
                    "trigger_score_peak": round(max(segment_scores), 6),
                    "trigger_score_mean_active": round(sum(segment_scores) / len(segment_scores), 6),
                    "active_frame_count": len(active_indexes),
                    "context_frame_count": len(context_indexes),
                    "evidence_keys": evidence_keys,
                    "evidence_channels": evidence_channels,
                    "frame_refs": [
                        {
                            "frame_index": rows[index]["frame_index"],
                            "timestamp_ms": rows[index]["timestamp_ms"],
                            "frame_ref": rows[index]["frame_ref"],
                            **({"frame_sha256": rows[index]["frame_sha256"]} if rows[index].get("frame_sha256") else {}),
                        }
                        for index in review_indexes
                    ],
                    "candidate_status": "unreviewed_trigger",
                    "truth_status": "not_evaluated",
                    "authority": "DISCOVERY_CANDIDATE_ONLY",
                }
                base["candidate_id"] = _candidate_id(run_id, base)
                candidates.append(base)
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["source_id"],
            candidate["session_id"],
            candidate["start_timestamp_ms"],
            candidate["trigger_type"],
        ),
    )


def _merge_frame_refs(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], maximum: int) -> list[dict[str, Any]]:
    by_index = {int(item["frame_index"]): dict(item) for item in [*left, *right]}
    indexes = sorted(by_index)
    selected = _sample_frame_indexes(indexes, maximum, indexes[len(indexes) // 2])
    return [by_index[index] for index in selected]


def deduplicate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    run_id: str,
) -> list[dict[str, Any]]:
    gap = int(contract["deduplication"]["max_merge_gap_ms"])
    maximum = int(contract["windowing"]["max_review_frames"])
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate["source_id"], candidate["session_id"], candidate["trigger_type"])].append(candidate)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda item: item["start_timestamp_ms"])
        current: dict[str, Any] | None = None
        for item in items:
            if current is None:
                current = dict(item)
                current["merged_candidate_ids"] = [item["candidate_id"]]
                continue
            if item["start_timestamp_ms"] <= current["end_timestamp_ms"] + gap:
                current["end_frame_index"] = max(current["end_frame_index"], item["end_frame_index"])
                current["end_timestamp_ms"] = max(current["end_timestamp_ms"], item["end_timestamp_ms"])
                current["context_start_timestamp_ms"] = min(current["context_start_timestamp_ms"], item["context_start_timestamp_ms"])
                current["context_end_timestamp_ms"] = max(current["context_end_timestamp_ms"], item["context_end_timestamp_ms"])
                current["peak_frame_index"] = (
                    item["peak_frame_index"]
                    if item["trigger_score_peak"] > current["trigger_score_peak"]
                    else current["peak_frame_index"]
                )
                current["trigger_score_peak"] = round(max(current["trigger_score_peak"], item["trigger_score_peak"]), 6)
                current["trigger_score_mean_active"] = round(
                    max(current["trigger_score_mean_active"], item["trigger_score_mean_active"]), 6
                )
                current["active_frame_count"] += item["active_frame_count"]
                current["context_frame_count"] = max(current["context_frame_count"], item["context_frame_count"])
                current["evidence_keys"] = sorted(set(current["evidence_keys"]) | set(item["evidence_keys"]))
                current["evidence_channels"] = sorted(set(current["evidence_channels"]) | set(item["evidence_channels"]))
                current["frame_refs"] = _merge_frame_refs(current["frame_refs"], item["frame_refs"], maximum)
                current["merged_candidate_ids"].append(item["candidate_id"])
            else:
                current["deduplicated_count"] = len(current["merged_candidate_ids"])
                current["candidate_id"] = _candidate_id(run_id, current)
                output.append(current)
                current = dict(item)
                current["merged_candidate_ids"] = [item["candidate_id"]]
        if current is not None:
            current["deduplicated_count"] = len(current["merged_candidate_ids"])
            current["candidate_id"] = _candidate_id(run_id, current)
            output.append(current)
    return sorted(output, key=lambda item: (item["source_id"], item["session_id"], item["start_timestamp_ms"], item["trigger_type"]))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def cluster_candidates(
    candidates: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        channels = row.get("evidence_channels", [])
        dominant = sorted(channels)[0] if channels else "unknown"
        buckets[(row["trigger_type"], dominant)].append(index)
    dedup = contract["deduplication"]
    for indexes in buckets.values():
        for position, left_index in enumerate(indexes):
            left = rows[left_index]
            left_channels = set(left.get("evidence_channels", []))
            left_duration = max(1, left["end_timestamp_ms"] - left["start_timestamp_ms"])
            for right_index in indexes[position + 1 :]:
                right = rows[right_index]
                right_channels = set(right.get("evidence_channels", []))
                right_duration = max(1, right["end_timestamp_ms"] - right["start_timestamp_ms"])
                duration_ratio = max(left_duration, right_duration) / min(left_duration, right_duration)
                if (
                    _jaccard(left_channels, right_channels) >= float(dedup["min_evidence_jaccard"])
                    and duration_ratio <= float(dedup["max_duration_ratio"])
                    and abs(left["trigger_score_peak"] - right["trigger_score_peak"]) <= float(dedup["max_score_delta"])
                ):
                    union(left_index, right_index)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[find(index)].append(index)
    for group in members.values():
        member_ids = sorted(rows[index]["candidate_id"] for index in group)
        cluster_id = "cemc-" + canonical_json_sha256(member_ids)[:20]
        for index in group:
            rows[index]["cluster_id"] = cluster_id
            rows[index]["cluster_size"] = len(group)
    return sorted(rows, key=lambda item: (item["source_id"], item["session_id"], item["start_timestamp_ms"], item["trigger_type"]))


def build_candidate_report(
    frames: Sequence[Mapping[str, Any]],
    contract_meta: Mapping[str, str],
    project_index_meta: Mapping[str, str],
    input_trace_meta: Mapping[str, str],
    run_id: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = normalize_frames(frames)
    extracted = extract_candidate_windows(normalized, contract, run_id)
    deduplicated = deduplicate_candidates(extracted, contract, run_id)
    clustered = cluster_candidates(deduplicated, contract)
    return {
        "schema": CANDIDATE_REPORT_SCHEMA,
        "module": "candidate_event_mining",
        "run_id": run_id,
        "contract": dict(contract_meta),
        "project_index": dict(project_index_meta),
        "input_trace": dict(input_trace_meta),
        "candidate_output_visibility": "internal_pipeline_only",
        "candidate_truth_authority": "none",
        "candidates": clustered,
        "summary": {
            "input_frame_count": len(normalized),
            "raw_candidate_count": len(extracted),
            "deduplicated_candidate_count": len(deduplicated),
            "cluster_count": len({row["cluster_id"] for row in clustered}),
            "candidate_type_counts": {
                target: sum(row["trigger_type"] == target for row in clustered)
                for target in TARGETS
            },
        },
        "authority": {
            "research_lane": "THESIS_DEVELOPMENT",
            "discovery_only": True,
            "training": False,
            "confirmation": False,
            "production": False,
            "safety": False,
            "default_app": False,
        },
    }


def review_input(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "source_id": candidate["source_id"],
        "session_id": candidate["session_id"],
        "window": {
            "start_timestamp_ms": candidate["context_start_timestamp_ms"],
            "end_timestamp_ms": candidate["context_end_timestamp_ms"],
            "frame_refs": list(candidate["frame_refs"]),
        },
        "review_taxonomy": list(TARGETS),
        "candidate_type_hidden": True,
        "model_signal_fields_hidden": True,
        "truth_fields_hidden": True,
    }


def validate_luna_review(
    review: Mapping[str, Any],
    candidate_id: str,
    bundle_manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if review.get("schema") != LUNA_REVIEW_SCHEMA:
        raise ContractError(f"invalid Luna review schema: {candidate_id}")
    if review.get("candidate_id") != candidate_id:
        raise ContractError(f"Luna review candidate mismatch: {candidate_id}")
    if review.get("reviewer_type") != "ai_model" or review.get("reviewer_role") != "luna_reader":
        raise ContractError(f"Luna reviewer identity invalid: {candidate_id}")
    for key in ("reviewer_id", "provider", "model_version", "review_run_id", "workflow_id"):
        _require_string(review.get(key), f"{candidate_id}.{key}")
    if review.get("independent_review") is not True or review.get("isolated_context") is not True:
        raise ContractError(f"Luna review is not isolated: {candidate_id}")
    if review.get("other_review_outputs_viewed") is not False or review.get("candidate_output_visible") is not False:
        raise ContractError(f"Luna review visibility firewall failed: {candidate_id}")
    if review.get("input_sha256") != bundle_manifest.get("review_inputs_sha256"):
        raise ContractError(f"Luna input hash mismatch: {candidate_id}")
    if review.get("prompt_sha256") != bundle_manifest.get("review_prompt_sha256"):
        raise ContractError(f"Luna prompt hash mismatch: {candidate_id}")
    confidence = _finite_score(review.get("confidence"), f"{candidate_id}.confidence")
    observed_types = review.get("observed_types")
    if not isinstance(observed_types, list) or not observed_types or any(item not in TARGETS for item in observed_types):
        raise ContractError(f"Luna observed_types invalid: {candidate_id}")
    disposition = review.get("disposition")
    if disposition not in {"keep", "reject", "quarantine"}:
        raise ContractError(f"Luna disposition invalid: {candidate_id}")
    abstained = review.get("abstained")
    if not isinstance(abstained, bool):
        raise ContractError(f"Luna abstained must be boolean: {candidate_id}")
    if abstained and disposition != "quarantine":
        raise ContractError(f"abstained Luna review must quarantine: {candidate_id}")
    if confidence < float(contract["review"]["minimum_confidence"]) and disposition == "keep":
        raise ContractError(f"low-confidence Luna review cannot keep: {candidate_id}")
    return dict(review)


def make_review_bundle(
    candidate_report: Mapping[str, Any],
    candidate_report_path: Path,
    contract_meta: Mapping[str, str],
    contract: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    refuse_overwrite(output_dir)
    output_dir.mkdir(parents=True)
    inputs = [review_input(candidate) for candidate in candidate_report.get("candidates", [])]
    input_path = output_dir / "review_inputs.jsonl"
    write_jsonl(input_path, inputs)
    prompt = (
        "You are Luna, an isolated discovery reviewer. Inspect each supplied multi-frame window "
        "without using detector scores, segmentation scores, depth values, HFTF fields, candidate "
        "labels, or another review. For each candidate, report only visually observable categories "
        "from review_taxonomy, confidence, and disposition. Use quarantine when the window is not "
        "evaluable or the event boundary is ambiguous. A kept item is a discovery candidate, not "
        "truth, safety evidence, or production evidence.\n"
    )
    prompt_path = output_dir / "review_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_version": "r0",
        "candidate_report_path": str(candidate_report_path.resolve()),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "contract": dict(contract_meta),
        "review_inputs_path": str(input_path.resolve()),
        "review_inputs_sha256": sha256_file(input_path),
        "review_prompt_path": str(prompt_path.resolve()),
        "review_prompt_sha256": sha256_file(prompt_path),
        "candidate_ids": [candidate["candidate_id"] for candidate in candidate_report.get("candidates", [])],
        "candidate_output_visible": False,
        "reviewer_role": "luna_reader",
        "required_reviews": int(contract["review"]["required_reviews"]),
        "authority": "DISCOVERY_CANDIDATE_ONLY",
    }
    write_json(output_dir / "review_bundle_manifest.json", manifest)
    write_json(output_dir / "review_schema_hint.json", {"schema": LUNA_REVIEW_SCHEMA, "taxonomy": list(TARGETS)})
    return manifest


def finalize_candidate_pool(
    candidate_report: Mapping[str, Any],
    candidate_report_path: Path,
    bundle_manifest: Mapping[str, Any],
    bundle_manifest_path: Path,
    reviews: Sequence[Mapping[str, Any]],
    reviews_path: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = {candidate["candidate_id"]: candidate for candidate in candidate_report.get("candidates", [])}
    expected = set(bundle_manifest.get("candidate_ids", []))
    if set(candidates) != expected:
        raise ContractError("candidate report and review bundle candidate IDs differ")
    by_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        candidate_id = review.get("candidate_id")
        if candidate_id in by_id:
            raise ContractError(f"duplicate Luna review: {candidate_id}")
        if candidate_id not in candidates:
            raise ContractError(f"unknown Luna review candidate: {candidate_id}")
        by_id[candidate_id] = validate_luna_review(review, candidate_id, bundle_manifest, contract)
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        raise ContractError(f"missing Luna reviews: {missing}")

    pool: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for candidate_id in sorted(expected):
        candidate = dict(candidates[candidate_id])
        review = by_id[candidate_id]
        accepted = (
            review["disposition"] == "keep"
            and review["abstained"] is False
            and review["confidence"] >= float(contract["review"]["minimum_confidence"])
        )
        candidate["luna_reviewed_types"] = sorted(review["observed_types"])
        candidate["luna_review_confidence"] = review["confidence"]
        candidate["luna_review_sha256"] = canonical_json_sha256(review)
        candidate["candidate_status"] = "luna_reviewed_keep" if accepted else "luna_reviewed_quarantine"
        if accepted:
            candidate["pool_authority"] = "DISCOVERY_CANDIDATE_ONLY"
            pool.append(candidate)
        else:
            quarantine.append({
                "candidate_id": candidate_id,
                "disposition": review["disposition"],
                "abstained": review["abstained"],
                "confidence": review["confidence"],
                "observed_types": sorted(review["observed_types"]),
            })
    result = {
        "schema": POOL_SCHEMA,
        "pool_version": "r0",
        "candidate_report_path": str(candidate_report_path.resolve()),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "review_bundle_manifest_path": str(bundle_manifest_path.resolve()),
        "review_bundle_manifest_sha256": sha256_file(bundle_manifest_path),
        "reviews_path": str(reviews_path.resolve()),
        "reviews_sha256": sha256_file(reviews_path),
        "candidate_output_visibility": False,
        "pool": pool,
        "quarantine": quarantine,
        "summary": {
            "candidate_count": len(candidates),
            "pool_count": len(pool),
            "quarantine_count": len(quarantine),
        },
        "authority": {
            "research_lane": "THESIS_DEVELOPMENT",
            "discovery_only": True,
            "event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
            "safety": False,
            "default_app": False,
        },
    }
    review_queue = candidate_report.get("review_queue")
    if isinstance(review_queue, dict):
        result["review_queue"] = dict(review_queue)
        result["summary"]["unreviewed_candidate_count"] = int(
            review_queue.get("unreviewed_candidate_count", 0)
        )
    return result
