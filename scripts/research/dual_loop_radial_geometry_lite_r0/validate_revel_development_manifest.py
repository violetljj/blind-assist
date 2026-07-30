#!/usr/bin/env python3
"""Validate the frozen REveL producer allowlist and evaluator truth split."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


FORBIDDEN_PRODUCER_FIELDS = {
    "oracle_target_label",
    "truth_available",
    "truth_unavailable_reason",
    "truth_signed_approach_mps",
    "truth_state",
    "truth_deadband_mps",
    "truth_offline_noncausal",
    "event_id",
    "event_anchor_region",
    "primary_event_eligible",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every JSONL row must be an object")
    return rows


def _resolve_recorded_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def validate(output_root: Path, repository_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "blindassist_dual_loop_revel_development_input_freeze_v1":
        errors.append("MANIFEST_FORMAT")
    if manifest.get("protocol_id") != "DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0":
        errors.append("PROTOCOL_ID")
    if manifest.get("independence_group") != "REVEL_DYNAMIC_SINGLE_CAPTURE":
        errors.append("INDEPENDENCE_GROUP")
    if manifest.get("access_and_claims", {}).get("old_f1b_decision_output_access") != "FORBIDDEN":
        errors.append("OLD_F1B_FIREWALL")

    recorded_replay = manifest.get("producer_allowlist", {}).get("replay_input", {})
    recorded_truth = manifest.get("evaluator_truth", {}).get("truth", {})
    recorded_events = manifest.get("evaluator_truth", {}).get("natural_events", {})
    replay_path = _resolve_recorded_path(str(recorded_replay.get("path", "")), repository_root)
    truth_path = _resolve_recorded_path(str(recorded_truth.get("path", "")), repository_root)
    events_path = _resolve_recorded_path(str(recorded_events.get("path", "")), repository_root)
    for label, path, record in (
        ("REPLAY", replay_path, recorded_replay),
        ("TRUTH", truth_path, recorded_truth),
        ("EVENTS", events_path, recorded_events),
    ):
        if not path.is_file():
            errors.append(f"{label}_MISSING")
        elif sha256_file(path) != record.get("sha256"):
            errors.append(f"{label}_SHA256")

    if errors:
        return {"status": "INVALID", "errors": sorted(set(errors))}
    replay_rows = read_jsonl(replay_path)
    truth_rows = read_jsonl(truth_path)
    events = read_jsonl(events_path)
    image_root = _resolve_recorded_path(
        str(manifest.get("producer_allowlist", {}).get("image_root", "")),
        repository_root,
    ).resolve()
    if not image_root.is_dir():
        errors.append("REPLAY_IMAGE_ROOT")
    if len(replay_rows) != recorded_replay.get("rows"):
        errors.append("REPLAY_ROW_COUNT")
    if len(truth_rows) != recorded_truth.get("rows"):
        errors.append("TRUTH_ROW_COUNT")
    if len(events) != recorded_events.get("rows"):
        errors.append("EVENT_ROW_COUNT")

    replay_keys: set[tuple[str, str]] = set()
    previous_by_target: dict[str, tuple[int, int, str]] = {}
    for row in replay_rows:
        if FORBIDDEN_PRODUCER_FIELDS.intersection(row):
            errors.append("PRODUCER_TRUTH_LEAK")
        key = (str(row.get("source_frame_id")), str(row.get("target_id")))
        if key in replay_keys:
            errors.append("REPLAY_DUPLICATE")
        replay_keys.add(key)
        if row.get("target_id") not in {"track-000", "track-001"}:
            errors.append("REPLAY_TARGET")
        if row.get("region") not in {"LEFT", "CENTER", "RIGHT"}:
            errors.append("REPLAY_REGION")
        if not isinstance(row.get("track_epoch"), str):
            errors.append("REPLAY_EPOCH")
        image_relative = Path(str(row.get("image_relative_path", "")))
        if image_relative.is_absolute() or ".." in image_relative.parts:
            errors.append("REPLAY_IMAGE_PATH")
        else:
            image_path = (image_root / image_relative).resolve()
            try:
                image_path.relative_to(image_root)
            except ValueError:
                errors.append("REPLAY_IMAGE_SCOPE")
            else:
                if not image_path.is_file():
                    errors.append("REPLAY_IMAGE_MISSING")
        target_id = str(row.get("target_id"))
        current = (
            int(row.get("source_frame_index", -1)),
            int(row.get("captured_at_ns", -1)),
            str(row.get("track_epoch")),
        )
        previous = previous_by_target.get(target_id)
        if previous and (current[0] <= previous[0] or current[1] <= previous[1]):
            errors.append("REPLAY_TIME_ORDER")
        if bool(row.get("history_reset")) != (previous is None or current[2] != previous[2]):
            errors.append("REPLAY_RESET")
        previous_by_target[target_id] = current

    truth_keys: set[tuple[str, str]] = set()
    event_ids = {str(event.get("event_id")) for event in events}
    for row in truth_rows:
        key = (str(row.get("source_frame_id")), str(row.get("target_id")))
        if key in truth_keys:
            errors.append("TRUTH_DUPLICATE")
        truth_keys.add(key)
        if row.get("target_id") not in {"track-000", "track-001"}:
            errors.append("TRUTH_TARGET")
        if bool(row.get("truth_available")) and row.get("truth_state") not in {
            "approaching",
            "quasi_static",
            "receding",
        }:
            errors.append("TRUTH_STATE")
        event_id = row.get("event_id")
        if event_id is not None and str(event_id) not in event_ids:
            errors.append("TRUTH_EVENT_REFERENCE")
    if not replay_keys.issubset(truth_keys):
        errors.append("REPLAY_TRUTH_JOIN")

    primary = [event for event in events if event.get("primary_event_eligible") is True]
    coverage = Counter(
        f"{event['target_id']}|{event['anchor_region']}|{event['truth_state']}"
        for event in primary
    )
    denominators = manifest.get("fixed_denominators", {})
    frozen_coverage = denominators.get("event_coverage", {})
    if len(replay_rows) != denominators.get("unique_roi_replay_opportunities"):
        errors.append("FROZEN_REPLAY_DENOMINATOR")
    if len(truth_rows) != denominators.get("target_frame_rows"):
        errors.append("FROZEN_TRUTH_DENOMINATOR")
    if len(events) != frozen_coverage.get("raw_event_count"):
        errors.append("FROZEN_RAW_EVENT_DENOMINATOR")
    if len(primary) != frozen_coverage.get("primary_event_count"):
        errors.append("FROZEN_PRIMARY_EVENT_DENOMINATOR")
    if dict(coverage) != frozen_coverage.get("by_target_anchor_region_truth_state"):
        errors.append("FROZEN_EVENT_COVERAGE")
    expected_cells = {
        f"{target}|{region}|{state}"
        for target in ("track-000", "track-001")
        for region in ("LEFT", "CENTER", "RIGHT")
        for state in ("approaching", "quasi_static", "receding")
    }
    if set(coverage) != expected_cells or min(coverage.values(), default=0) < 1:
        errors.append("REQUIRED_EVENT_CELL_EMPTY")

    return {
        "status": "VALID" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "replay_rows": len(replay_rows),
        "truth_rows": len(truth_rows),
        "raw_events": len(events),
        "primary_events": len(primary),
        "minimum_primary_cell_events": min(coverage.values(), default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate(args.output_root, args.repository_root.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
