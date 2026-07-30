from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from contract import EVENT_COUNT, PRIMARY_ARM, REFERENCE_ARM, ContractError, sha256_file


class BindingError(ContractError):
    pass


FORBIDDEN_PATH_TOKENS = (
    "f-1b",
    "confirmation",
)


@dataclass(frozen=True)
class FrozenBundle:
    repo_root: Path
    dependency: dict[str, Any]
    manifest: dict[str, Any]
    replay_rows: tuple[dict[str, Any], ...]
    truth_rows: tuple[dict[str, Any], ...]
    natural_rows: tuple[dict[str, Any], ...]
    r2_rows: tuple[dict[str, Any], ...]
    evaluation: dict[str, Any]
    binding_summary: dict[str, Any]


def _safe_path(repo_root: Path, relative: str) -> Path:
    normalized = relative.replace("\\", "/").lower()
    if any(token in normalized for token in FORBIDDEN_PATH_TOKENS):
        raise BindingError(f"forbidden input path: {relative}")
    if "production-temporal-geometry-factorial-ab-r0" in normalized and "trace" in normalized:
        raise BindingError("production A/B trace access is forbidden")
    root = repo_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BindingError(f"path escapes repository: {relative}") from exc
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BindingError(f"{path} must contain an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise BindingError(f"blank JSONL row: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise BindingError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _index_unique(
    rows: Iterable[dict[str, Any]], fields: tuple[str, ...], label: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if any(value is None for value in key):
            raise BindingError(f"{label} key contains null: {key}")
        if key in index:
            raise BindingError(f"duplicate {label} key: {key}")
        index[key] = row
    return index


def _verify_file(
    repo_root: Path,
    specification: dict[str, Any],
    *,
    expected_rows: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = _safe_path(repo_root, specification["path"])
    actual_sha = sha256_file(path)
    expected_sha = specification.get("sha256")
    if expected_sha is not None and actual_sha != expected_sha:
        raise BindingError(f"SHA-256 drift: {specification['path']}")
    summary: dict[str, Any] = {
        "path": specification["path"],
        "sha256": actual_sha,
    }
    if expected_rows is not None:
        summary["rows"] = expected_rows
    return path, summary


def validate_predecessor_gate(repo_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    gate = protocol["predecessor_gate"]
    result_path = _safe_path(repo_root, gate["result"]["path"])
    validation_path = _safe_path(repo_root, gate["independent_validation"]["path"])
    seal_path = _safe_path(repo_root, gate["seal"]["path"])
    for path, expected in (
        (result_path, gate["result"]["sha256"]),
        (validation_path, gate["independent_validation"]["sha256"]),
        (seal_path, gate["seal"]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise BindingError(f"predecessor binding drift: {path}")
    validation = _load_json(validation_path)
    seal = _load_json(seal_path)
    expected_validation = gate["independent_validation"]
    checks = {
        "status": expected_validation["required_status"],
        "truth_opened": expected_validation["required_truth_opened"],
        "frame_count": expected_validation["required_frame_count"],
        "trace_row_count": expected_validation["required_trace_row_count"],
        "branch_pair_mismatch_count": expected_validation[
            "required_branch_pair_mismatch_count"
        ],
        "failure_count": expected_validation["required_failure_count"],
    }
    for field, expected in checks.items():
        if validation.get(field) != expected:
            raise BindingError(f"predecessor validation field drift: {field}")
    if seal.get("status") != gate["seal"]["required_status"]:
        raise BindingError("predecessor seal status drift")
    return {
        "result_sha256": gate["result"]["sha256"],
        "validation_sha256": gate["independent_validation"]["sha256"],
        "seal_sha256": gate["seal"]["sha256"],
    }


def _evaluation_rows(
    evaluation: dict[str, Any], arm_id: str
) -> list[dict[str, Any]]:
    try:
        events = evaluation["arm_summaries"][arm_id]["events"]
    except (KeyError, TypeError) as exc:
        raise BindingError(f"missing evaluation events for {arm_id}") from exc
    if not isinstance(events, list):
        raise BindingError(f"evaluation events for {arm_id} must be a list")
    output: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise BindingError("evaluation event must be an object")
        output.append({"arm_id": arm_id, **event})
    return output


def _validate_joins(
    replay_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    natural_rows: list[dict[str, Any]],
    r2_rows: list[dict[str, Any]],
    evaluation: dict[str, Any],
    dependency: dict[str, Any],
) -> None:
    replay = _index_unique(
        replay_rows, ("capture_id", "target_id", "source_frame_id"), "replay"
    )
    truth = _index_unique(
        truth_rows, ("capture_id", "target_id", "source_frame_id"), "truth"
    )
    for key, replay_row in replay.items():
        truth_row = truth.get(key)
        if truth_row is None:
            raise BindingError(f"replay row lacks truth join: {key}")
        if truth_row.get("bag_image_timestamp_ns") != replay_row.get("captured_at_ns"):
            raise BindingError(f"truth timestamp mismatch: {key}")

    r2 = _index_unique(
        r2_rows,
        ("arm_id", "capture_id", "target_id", "source_frame_id"),
        "R2 producer",
    )
    replay_keys = set(replay)
    for arm_id in (REFERENCE_ARM, PRIMARY_ARM):
        arm_keys = {
            (capture, target, source)
            for arm, capture, target, source in r2
            if arm == arm_id
        }
        if arm_keys != replay_keys:
            raise BindingError(f"R2 {arm_id} keyset drift")

    primary = {
        row["event_id"]: row
        for row in natural_rows
        if row.get("primary_event_eligible") is True
    }
    if len(primary) != EVENT_COUNT:
        raise BindingError("primary natural-event count drift")
    dependency_ids = [row["event_id"] for row in dependency["event_bindings"]]
    if dependency_ids != sorted(primary):
        raise BindingError("dependency event binding order/keyset drift")
    for binding in dependency["event_bindings"]:
        event = primary[binding["event_id"]]
        midpoint = (event["start_timestamp_ns"] + event["end_timestamp_ns"]) // 2
        for field, expected in (
            ("target_id", event["target_id"]),
            ("anchor_region", event["anchor_region"]),
            ("truth_state", event["truth_state"]),
            ("start_timestamp_ns", event["start_timestamp_ns"]),
            ("end_timestamp_ns", event["end_timestamp_ns"]),
            ("midpoint_timestamp_ns", midpoint),
        ):
            if binding.get(field) != expected:
                raise BindingError(
                    f"dependency binding {field} drift: {binding['event_id']}"
                )

    truth_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in truth_rows:
        if row.get("event_id") in primary:
            truth_by_event.setdefault(row["event_id"], []).append(row)
    for event_id, event in primary.items():
        members = sorted(
            truth_by_event.get(event_id, []), key=lambda row: row["source_frame_index"]
        )
        expected_indices = list(
            range(
                event["start_source_frame_index"],
                event["end_source_frame_index"] + 1,
            )
        )
        if [row["source_frame_index"] for row in members] != expected_indices:
            raise BindingError(f"natural-event membership drift: {event_id}")
        if len(members) != event["eligible_frame_count"]:
            raise BindingError(f"natural-event denominator drift: {event_id}")
        first, last = members[0], members[-1]
        if (
            first["bag_image_timestamp_ns"] != event["start_timestamp_ns"]
            or last["bag_image_timestamp_ns"] != event["end_timestamp_ns"]
        ):
            raise BindingError(f"natural-event endpoint drift: {event_id}")
        for member in members:
            for field, expected in (
                ("capture_id", event["capture_id"]),
                ("target_id", event["target_id"]),
                ("truth_state", event["truth_state"]),
                ("event_id", event_id),
                ("event_anchor_region", event["anchor_region"]),
            ):
                if member.get(field) != expected:
                    raise BindingError(f"natural-event member {field} drift: {event_id}")

    event_keyset = set(primary)
    for arm_id in (REFERENCE_ARM, PRIMARY_ARM):
        events = _evaluation_rows(evaluation, arm_id)
        event_index = _index_unique(events, ("arm_id", "event_id"), "evaluation")
        if {key[1] for key in event_index} != event_keyset:
            raise BindingError(f"evaluation {arm_id} keyset drift")
        for (_, event_id), row in event_index.items():
            natural = primary[event_id]
            for field in ("target_id", "anchor_region", "truth_state"):
                if row.get(field) != natural.get(field):
                    raise BindingError(f"evaluation {field} drift: {arm_id}/{event_id}")
            denominator = natural["eligible_frame_count"]
            if row.get("denominator_rows") != denominator:
                raise BindingError(f"evaluation denominator drift: {arm_id}/{event_id}")
            expected_coverage = row.get("non_abstained_rows") / denominator
            if row.get("coverage") != expected_coverage:
                raise BindingError(f"evaluation coverage drift: {arm_id}/{event_id}")


def load_prestart_bundle(repo_root: Path, protocol: dict[str, Any]) -> FrozenBundle:
    root = repo_root.resolve()
    specifications = protocol["frozen_inputs"]
    summaries: dict[str, Any] = {}

    dep_path, summaries["dependency_receipt"] = _verify_file(
        root, specifications["dependency_receipt"]
    )
    dependency = _load_json(dep_path)
    dep_spec = specifications["dependency_receipt"]
    if (
        dependency.get("schema_version") != dep_spec["schema_version"]
        or dependency.get("status") != dep_spec["status"]
        or len(dependency.get("event_bindings", [])) != dep_spec["primary_event_count"]
        or dependency.get("cross_target_overlap_pair_count")
        != dep_spec["cross_target_overlap_pair_count"]
        or dependency.get("same_target_overlap_pair_count")
        != dep_spec["same_target_overlap_pair_count"]
        or dependency.get("exact_overlap_component_count")
        != dep_spec["exact_overlap_component_count"]
        or dependency.get("old_f1b_decision_opened") is not False
        or dependency.get("production_ab_trace_opened") is not False
        or dependency.get("confirmation_opened") is not False
        or dependency.get("candidate_output_opened") is not False
    ):
        raise BindingError("dependency receipt semantic drift")

    manifest_path = _safe_path(root, specifications["input_manifest"]["path"])
    manifest = _load_json(manifest_path)
    if (
        sha256_file(manifest_path)
        != specifications["input_manifest"]["content_identity_sha256"]
    ):
        raise BindingError("input manifest content identity drift")
    summaries["input_manifest"] = {
        "path": specifications["input_manifest"]["path"],
        "content_identity_sha256": sha256_file(manifest_path),
    }

    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in ("replay_input", "truth", "natural_events", "r2_producer_output"):
        spec = specifications[name]
        path, summary = _verify_file(root, spec)
        rows = _load_jsonl(path)
        if len(rows) != spec["rows"]:
            raise BindingError(f"{name} row-count drift")
        summary["rows"] = len(rows)
        summaries[name] = summary
        loaded[name] = rows
    if (
        sum(row.get("primary_event_eligible") is True for row in loaded["natural_events"])
        != specifications["natural_events"]["primary_rows"]
    ):
        raise BindingError("natural-events primary-row count drift")

    r2_receipt: dict[str, Any] | None = None
    for name in ("r2_producer_receipt", "r2_evaluation"):
        spec = specifications[name]
        path, summary = _verify_file(root, spec)
        summaries[name] = summary
        if name == "r2_producer_receipt":
            r2_receipt = _load_json(path)
        else:
            evaluation = _load_json(path)
    if r2_receipt is None or (
        r2_receipt.get("status") != "PRODUCER_COMPLETE"
        or r2_receipt.get("mode") != "formal"
        or r2_receipt.get("truth_joined") is not False
        or r2_receipt.get("input_rows") != specifications["replay_input"]["rows"]
        or r2_receipt.get("output_rows")
        != specifications["r2_producer_output"]["rows"]
        or r2_receipt.get("replay_input_sha256")
        != specifications["replay_input"]["sha256"]
        or r2_receipt.get("output_sha256")
        != specifications["r2_producer_output"]["sha256"]
        or r2_receipt.get("arm_ids") != [REFERENCE_ARM, PRIMARY_ARM]
    ):
        raise BindingError("R2 producer receipt semantic drift")
    for name in ("revel_dynamic_bag", "revel_calibration", "vicon_audit"):
        spec = specifications[name]
        path, summary = _verify_file(root, spec)
        if "bytes" in spec:
            actual_bytes = path.stat().st_size
            if actual_bytes != spec["bytes"]:
                raise BindingError(f"{name} byte-size drift")
            summary["bytes"] = actual_bytes
        summaries[name] = summary
    validate_predecessor_gate(root, protocol)
    _validate_joins(
        loaded["replay_input"],
        loaded["truth"],
        loaded["natural_events"],
        loaded["r2_producer_output"],
        evaluation,
        dependency,
    )
    return FrozenBundle(
        repo_root=root,
        dependency=dependency,
        manifest=manifest,
        replay_rows=tuple(loaded["replay_input"]),
        truth_rows=tuple(loaded["truth"]),
        natural_rows=tuple(loaded["natural_events"]),
        r2_rows=tuple(loaded["r2_producer_output"]),
        evaluation=evaluation,
        binding_summary=summaries,
    )
