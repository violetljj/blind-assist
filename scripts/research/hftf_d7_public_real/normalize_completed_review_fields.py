#!/usr/bin/env python3
"""Normalize legacy bucket/decision keys in a completed review artifact."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from pipeline import ContractError, load_jsonl


DECISIONS = {"SUPPORT", "REJECT", "NOT_EVALUABLE", "ESCALATE"}
REVIEW_ROLES = {
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
    "GEOMETRY_EVIDENCE_REVIEWER",
    "COUNTEREXAMPLE_REVIEWER",
}
POSITIVE_BUCKETS = {
    "BLOCKING_BODY_POSITIVE",
    "BOUNDARY_LEVEL_CHANGE_POSITIVE",
    "HEAD_HAZARD_POSITIVE",
    "DYNAMIC_INTRUSION_POSITIVE",
}
INTERVAL_KEYS = {"pre_interval", "alertable_interval", "passed_clearance_interval", "continuous_negative_interval"}


def _interval_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        return int(value.get("start_timestamp_ns")) < int(value.get("end_timestamp_ns"))
    except (TypeError, ValueError):
        return False


def _phase_valid(bucket: str, phases: object) -> bool:
    if not isinstance(phases, dict) or not set(phases).issubset(INTERVAL_KEYS):
        return False
    if bucket in POSITIVE_BUCKETS:
        return all(_interval_valid(phases.get(key)) for key in (
            "pre_interval", "alertable_interval", "passed_clearance_interval"
        ))
    return _interval_valid(phases.get("continuous_negative_interval"))


def _negative_interval_from_positive_phase_span(phases: object) -> dict[str, int] | None:
    """Re-key a contiguous negative SUPPORT phase without changing its span."""

    if not isinstance(phases, dict):
        return None
    ordered = [phases.get(key) for key in (
        "pre_interval", "alertable_interval", "passed_clearance_interval"
    )]
    if not all(_interval_valid(value) for value in ordered):
        return None
    first, second, third = ordered
    if int(first["end_timestamp_ns"]) != int(second["start_timestamp_ns"]):
        return None
    if int(second["end_timestamp_ns"]) != int(third["start_timestamp_ns"]):
        return None
    return {
        "start_timestamp_ns": int(first["start_timestamp_ns"]),
        "end_timestamp_ns": int(third["end_timestamp_ns"]),
    }


def _negative_interval_from_manifest(
    row: dict[str, object], manifest_row: dict[str, object]
) -> dict[str, object] | None:
    """Bind a reviewer-declared full-window negative interval to source time.

    Some older role outputs recorded the negative interval in frame/time-from-
    source-start fields while the frozen review contract requires nanoseconds.
    The immutable role manifest already binds the candidate window to source
    timestamps; reuse that binding without inventing a shorter interval.
    """

    raw = row.get("continuous_negative_interval")
    if not isinstance(raw, dict):
        phases = row.get("phase_intervals")
        if isinstance(phases, dict):
            raw = phases.get("continuous_negative_interval")
    if not isinstance(raw, dict):
        return None
    start = raw.get("start_timestamp_ns")
    end = raw.get("end_timestamp_ns")
    if start is None:
        start = manifest_row.get("window_start_timestamp_ns")
    if end is None:
        end = manifest_row.get("window_end_timestamp_ns")
    try:
        start_ns = int(start)
        end_ns = int(end)
    except (TypeError, ValueError):
        return None
    if start_ns >= end_ns:
        return None
    bound = dict(raw)
    bound["start_timestamp_ns"] = start_ns
    bound["end_timestamp_ns"] = end_ns
    bound["interval_binding"] = "IMMUTABLE_REVIEW_MANIFEST_WINDOW"
    return {"continuous_negative_interval": bound}


def normalize(
    path: Path,
    expected_count: int,
    default_decision: str | None,
    role: str | None = None,
    downgrade_incomplete_support: bool = False,
    normalize_negative_support_phase: bool = False,
    manifest_path: Path | None = None,
    assume_model_blind_from_manifest: bool = False,
    canonicalize_completed_review: bool = False,
    bind_support_intervals_from_manifest: bool = False,
    rebind_identity_from_manifest: bool = False,
) -> dict[str, object]:
    if not path.is_file():
        raise ContractError(f"review output is missing: {path}")
    rows: list[dict[str, object]] = []
    input_ids_by_candidate: dict[str, str] = {}
    model_blind_by_candidate: dict[str, object] = {}
    review_index_by_candidate: dict[str, object] = {}
    manifest_by_candidate: dict[str, dict[str, object]] = {}
    if manifest_path is not None:
        manifest_rows = load_jsonl(manifest_path)
        for manifest_row in manifest_rows:
            candidate_id = str(manifest_row.get("candidate_id") or "")
            review_input_id = str(manifest_row.get("review_input_id") or "")
            if not candidate_id or not review_input_id or candidate_id in input_ids_by_candidate:
                raise ContractError(f"invalid or duplicate review manifest identity: {manifest_path}")
            input_ids_by_candidate[candidate_id] = review_input_id
            model_blind_by_candidate[candidate_id] = manifest_row.get("model_output_visible")
            review_index_by_candidate[candidate_id] = manifest_row.get("review_index")
            manifest_by_candidate[candidate_id] = manifest_row
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ContractError(f"review row is not an object: {path}:{line_number}")
            if "bucket" in row:
                if "event_bucket" in row:
                    raise ContractError(f"both bucket keys are present: {path}:{line_number}")
                row["event_bucket"] = row.pop("bucket")
            if "decision" not in row:
                terminal_alias = row.get("terminal")
                if terminal_alias in DECISIONS:
                    # Some isolated reviewers emit the contract terminal name
                    # instead of the canonical decision key.  Preserve the
                    # original terminal field and bind only recognized
                    # values; never infer SUPPORT/REJECT from free text.
                    row["decision"] = terminal_alias
                elif default_decision is None:
                    raise ContractError(f"missing decision: {path}:{line_number}")
                else:
                    row["decision"] = default_decision
            if row.get("decision") == "NOT_EVALUABLE" and row.get("event_bucket") is None:
                # Geometry/counterexample roles may express the terminal but
                # omit a bucket because no event bucket was evaluable.  The
                # canonical terminal is explicit and does not create a label.
                row["event_bucket"] = "NOT_EVALUABLE"
            if not row.get("review_input_id") and manifest_path is not None:
                candidate_id = str(row.get("candidate_id") or "")
                review_input_id = input_ids_by_candidate.get(candidate_id)
                if review_input_id is None:
                    raise ContractError(f"candidate missing from review manifest: {candidate_id}")
                row["review_input_id"] = review_input_id
                row["normalization_note"] = "REVIEW_INPUT_ID_BOUND_FROM_IMMUTABLE_MANIFEST"
            if row.get("model_output_visible") is None and assume_model_blind_from_manifest:
                candidate_id = str(row.get("candidate_id") or "")
                if model_blind_by_candidate.get(candidate_id) is not False:
                    raise ContractError(
                        f"cannot assume model blind without manifest false: {path}:{line_number}"
                    )
                row["model_output_visible"] = False
                row["normalization_note"] = "MODEL_BLIND_BOUND_FROM_IMMUTABLE_MANIFEST"
            if canonicalize_completed_review:
                if row.get("review_completed") not in (None, True):
                    raise ContractError(f"review is explicitly incomplete: {path}:{line_number}")
                candidate_id = str(row.get("candidate_id") or "")
                manifest_row = manifest_by_candidate.get(candidate_id)
                if manifest_row is None and manifest_path is not None:
                    raise ContractError(f"candidate missing from review manifest: {candidate_id}")
                if row.get("review_role") is None and row.get("reviewer_role") is not None:
                    row["review_role"] = row.pop("reviewer_role")
                if role is not None and row.get("review_role") is None:
                    row["review_role"] = role
                if rebind_identity_from_manifest and manifest_row is not None:
                    row["review_input_id"] = input_ids_by_candidate[candidate_id]
                    if manifest_row.get("batch_id") is not None:
                        row["batch_id"] = manifest_row["batch_id"]
                    row["normalization_note"] = (
                        "REVIEW_IDENTITY_REBOUND_FROM_IMMUTABLE_MANIFEST"
                    )
                row["schema"] = "hftf_d7_public_real_completed_review_v1"
                row["record_kind"] = "COMPLETED_REVIEW"
                row["review_completed"] = True
                if row.get("decision") == "NOT_EVALUABLE" and row.get("event_bucket") == "NOT_EVALUABLE":
                    row["phase_intervals"] = None
                elif row.get("decision") != "SUPPORT" and not isinstance(row.get("phase_intervals"), (dict, type(None))):
                    # REJECT/ESCALATE records do not carry a phase contract. Drop
                    # legacy list-shaped payloads instead of letting a
                    # non-canonical value reach the ingest boundary.
                    row["phase_intervals"] = None
                    row["normalization_note"] = "NON_SUPPORT_PHASE_INTERVALS_CANONICALIZED_TO_NULL"
                if "review_index" not in row and candidate_id in review_index_by_candidate:
                    row["review_index"] = review_index_by_candidate[candidate_id]
            if bind_support_intervals_from_manifest and row.get("decision") == "SUPPORT":
                bucket = str(row.get("event_bucket") or "")
                candidate_id = str(row.get("candidate_id") or "")
                manifest_row = manifest_by_candidate.get(candidate_id)
                if bucket.endswith("NEGATIVE") and manifest_row is not None:
                    bound = _negative_interval_from_manifest(row, manifest_row)
                    if bound is not None:
                        row["phase_intervals"] = bound
                        row["normalization_note"] = (
                            "NEGATIVE_SUPPORT_INTERVAL_BOUND_FROM_IMMUTABLE_REVIEW_MANIFEST"
                        )
            if downgrade_incomplete_support and row.get("decision") == "SUPPORT":
                bucket = str(row.get("event_bucket") or "")
                if normalize_negative_support_phase and bucket not in POSITIVE_BUCKETS:
                    continuous = _negative_interval_from_positive_phase_span(row.get("phase_intervals"))
                    if continuous is not None:
                        row["phase_intervals"] = {"continuous_negative_interval": continuous}
                        row["normalization_note"] = "NEGATIVE_SUPPORT_PHASE_REKEYED_CONTIGUOUS_SPAN"
                if not _phase_valid(bucket, row.get("phase_intervals")):
                    row["decision"] = "NOT_EVALUABLE"
                    row["event_bucket"] = "NOT_EVALUABLE"
                    row["phase_intervals"] = None
                    row["normalization_note"] = "SUPPORT_DOWNGRADED_MISSING_COMPLETE_PHASE_CONTRACT"
            if row.get("decision") not in DECISIONS:
                raise ContractError(f"invalid decision: {path}:{line_number}")
            if row.get("event_bucket") not in {
                "BLOCKING_BODY_POSITIVE",
                "BOUNDARY_LEVEL_CHANGE_POSITIVE",
                "HEAD_HAZARD_POSITIVE",
                "DYNAMIC_INTRUSION_POSITIVE",
                "PARALLEL_STRUCTURE_NEGATIVE",
                "SIDE_OBJECT_NONBLOCKING_NEGATIVE",
                "NORMAL_WALKABLE_NEGATIVE",
                "EGOMOTION_VISUAL_HARD_NEGATIVE",
                "HEAD_NONACTIONABLE_NEGATIVE",
                "NOT_EVALUABLE",
            }:
                raise ContractError(f"invalid event_bucket: {path}:{line_number}")
            if row.get("decision") == "NOT_EVALUABLE" and row.get("event_bucket") != "NOT_EVALUABLE":
                raise ContractError(f"NOT_EVALUABLE bucket mismatch: {path}:{line_number}")
            if row.get("model_output_visible") is not False:
                raise ContractError(f"model visibility is not false: {path}:{line_number}")
            if role is not None:
                if role not in REVIEW_ROLES:
                    raise ContractError(f"unknown review role: {role}")
                expected_geometry_only = role == "GEOMETRY_EVIDENCE_REVIEWER"
                geometry_only = row.get("source_native_geometry_only")
                if geometry_only is None:
                    row["source_native_geometry_only"] = expected_geometry_only
                elif geometry_only is not expected_geometry_only:
                    raise ContractError(f"source-native role flag conflicts with role: {path}:{line_number}")
                expected_counterexample = role == "COUNTEREXAMPLE_REVIEWER"
                counterexample_done = row.get("counterexample_search_completed")
                if counterexample_done is None:
                    row["counterexample_search_completed"] = expected_counterexample
                elif counterexample_done is not expected_counterexample:
                    raise ContractError(f"counterexample role flag conflicts with role: {path}:{line_number}")
            rows.append(row)
    if len(rows) != expected_count:
        raise ContractError(f"expected {expected_count} rows, got {len(rows)}: {path}")
    for key in ("candidate_id", "review_input_id"):
        values = [str(row.get(key) or "") for row in rows]
        if not all(values) or len(set(values)) != len(values):
            raise ContractError(f"{key} is not unique and complete: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return {"path": str(path), "rows": len(rows), "status": "NORMALIZED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--default-decision")
    parser.add_argument("--role", choices=sorted(REVIEW_ROLES))
    parser.add_argument("--downgrade-incomplete-support", action="store_true")
    parser.add_argument("--normalize-negative-support-phase", action="store_true")
    parser.add_argument("--manifest-path")
    parser.add_argument(
        "--assume-model-blind-from-manifest",
        action="store_true",
        help="fill missing model_output_visible only when the immutable role manifest is explicitly false",
    )
    parser.add_argument(
        "--canonicalize-completed-review",
        action="store_true",
        help="rebind a legacy completed row to the frozen completed-review schema",
    )
    parser.add_argument(
        "--bind-support-intervals-from-manifest",
        action="store_true",
        help="bind legacy negative support intervals to the immutable review-window timestamps",
    )
    parser.add_argument(
        "--rebind-identity-from-manifest",
        action="store_true",
        help="explicitly replace legacy review-role/input/batch identity with the immutable manifest identity",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(normalize(
        Path(args.path),
        args.expected_count,
        args.default_decision,
        args.role,
        args.downgrade_incomplete_support,
        args.normalize_negative_support_phase,
        Path(args.manifest_path) if args.manifest_path else None,
        args.assume_model_blind_from_manifest,
        args.canonicalize_completed_review,
        args.bind_support_intervals_from_manifest,
        args.rebind_identity_from_manifest,
    ), ensure_ascii=False, sort_keys=True))
