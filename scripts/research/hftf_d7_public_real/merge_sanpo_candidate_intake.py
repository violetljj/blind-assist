#!/usr/bin/env python3
"""Merge a validated SANPO candidate intake into the D7 assignment surface.

The merge is deliberately assignment-only.  It appends new SANPO candidates,
frames, sessions, event shells, and review assignments after a complete
preflight, preserves all existing completed reviews, and writes a recoverable
backup.  It never creates an admitted event or a split assignment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


DATASET = "SANPO-Real"
REVIEW_ROLES = (
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
    "GEOMETRY_EVIDENCE_REVIEWER",
    "COUNTEREXAMPLE_REVIEWER",
    "FINAL_ADJUDICATOR",
)
REVIEW_FILES = {
    "RGB_REVIEWER_A": "reviews/review_a.jsonl",
    "RGB_REVIEWER_B": "reviews/review_b.jsonl",
    "RGB_REVIEWER_C": "reviews/review_c.jsonl",
    "GEOMETRY_EVIDENCE_REVIEWER": "reviews/geometry_review.jsonl",
    "COUNTEREXAMPLE_REVIEWER": "reviews/counterexample_review.jsonl",
    "FINAL_ADJUDICATOR": "adjudication/final_adjudicator_assignments.jsonl",
}
TARGETS = {
    "BLOCKING_BODY_POSITIVE": 1500,
    "BOUNDARY_LEVEL_CHANGE_POSITIVE": 1000,
    "DYNAMIC_INTRUSION_POSITIVE": 1000,
    "HEAD_HAZARD_POSITIVE": 500,
    "PARALLEL_STRUCTURE_NEGATIVE": 1500,
    "SIDE_OBJECT_NONBLOCKING_NEGATIVE": 1000,
    "NORMAL_WALKABLE_NEGATIVE": 2000,
    "EGOMOTION_VISUAL_HARD_NEGATIVE": 1000,
    "HEAD_NONACTIONABLE_NEGATIVE": 500,
}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise ContractError(f"required JSONL missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ContractError(f"JSONL object required: {path}:{line_number}")
            yield value


def _line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _atomic_merge_jsonl(path: Path, new_rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            if path.is_file():
                with path.open("r", encoding="utf-8") as source:
                    shutil.copyfileobj(source, handle)
            for row in new_rows:
                handle.write(_line(row))
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _copy_backup(root: Path, backup_root: Path, relative_paths: Iterable[str]) -> None:
    for relative in relative_paths:
        source = root / relative
        if not source.is_file():
            raise ContractError(f"cannot back up missing D7 artifact: {source}")
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _candidate_ids(path: Path) -> set[str]:
    values: set[str] = set()
    for row in _iter_jsonl(path):
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in values:
            raise ContractError(f"duplicate or missing candidate_id in {path}: {candidate_id}")
        values.add(candidate_id)
    return values


def _frame_ids(path: Path) -> set[str]:
    values: set[str] = set()
    for row in _iter_jsonl(path):
        frame_id = str(row.get("frame_id") or "")
        if not frame_id or frame_id in values:
            raise ContractError(f"duplicate or missing frame_id in {path}: {frame_id}")
        values.add(frame_id)
    return values


def _new_review_row(candidate: dict[str, Any], *, role: str, event_id: str, reason: str) -> dict[str, Any]:
    geometry_only = role == "GEOMETRY_EVIDENCE_REVIEWER"
    counterexample = role == "COUNTEREXAMPLE_REVIEWER"
    return {
        "schema": "hftf_d7_public_real_review_record_v1",
        "record_kind": "ASSIGNMENT_ONLY",
        "review_role": role,
        "event_id": event_id,
        "candidate_id": candidate["candidate_id"],
        "dataset_id": DATASET,
        "source_session_id": candidate["source_session_id"],
        "review_completed": False,
        "decision": "PENDING",
        "event_bucket": None,
        "phase_intervals": None,
        "model_output_visible": False,
        "source_native_geometry_only": geometry_only,
        "counterexample_search_required": counterexample,
        "not_evaluable_reason": reason,
        "rgb_local_path": None,
    }


def _new_event_row(candidate: dict[str, Any], *, reason: str) -> dict[str, Any]:
    event_id = str(candidate.get("parent_event_id") or stable_id("d7parent", candidate["candidate_id"]))
    return {
        "schema": "hftf_d7_public_real_event_manifest_v1",
        "record_kind": "CANDIDATE_EVENT_SHELL",
        "event_id": event_id,
        "parent_event_id": event_id,
        "candidate_id": candidate["candidate_id"],
        "dataset_id": DATASET,
        "source_session_id": candidate["source_session_id"],
        "ancestry_group": candidate.get("ancestry_group"),
        "frame_ids": candidate.get("frame_ids", []),
        "start_timestamp_ns": None,
        "end_timestamp_ns": None,
        "event_bucket": "NOT_EVALUABLE",
        "truth_status": "NOT_EVALUABLE",
        "admission_status": "PENDING_REVIEW",
        "review_state": "NOT_RUN",
        "pre_interval": None,
        "alertable_interval": None,
        "passed_clearance_interval": None,
        "continuous_negative_interval": None,
        "candidate_selection_model_visible": False,
        "review_model_output_visible": False,
        "geometry_model_output_visible": False,
        "not_evaluable_reason": reason,
        "required_review_roles": list(REVIEW_ROLES),
        "source_license": candidate.get("source_license"),
        "source_hash": candidate.get("source_hash"),
        "rgb_local_path": None,
        "source_metadata": candidate.get("source_metadata"),
    }


def _new_queue_row(candidate: dict[str, Any], *, reason: str, event_id: str) -> dict[str, Any]:
    return {
        "schema": "hftf_d7_public_real_review_assignment_v1",
        "record_kind": "ASSIGNMENT_ONLY",
        "event_id": event_id,
        "candidate_id": candidate["candidate_id"],
        "dataset_id": DATASET,
        "source_session_id": candidate["source_session_id"],
        "roles": list(REVIEW_ROLES),
        "rgb_model_output_visible": False,
        "geometry_model_output_visible": False,
        "assignment_status": "PENDING",
        "not_evaluable_reason": reason,
    }


def _write_assignment_reports(root: Path, *, candidate_count: int, dataset_counts: Counter[str], pending_reasons: Counter[str]) -> None:
    generated = utc_now()
    quality = [
        "# HFTF D7 dataset quality report",
        "",
        f"Generated: `{generated}`",
        "",
        "## Terminal",
        "",
        "`NOT_COMPLETE`: SANPO source-coverage candidates are merged as assignment-only rows; no new event truth is admitted.",
        "",
        "| Check | Result | Evidence |",
        "| --- | --- | --- |",
        f"| Candidate windows | `{candidate_count}` | `candidates/candidate_index.jsonl` |",
        "| Admitted parent events | `0` | `adjudication/adjudicated_events.jsonl` is empty |",
        "| Event truth | `NOT_AUTHORIZED` | independent review/adjudication gate incomplete |",
        "| Training eligibility | `false` | all splits remain blocked |",
        "| Confirmation eligibility | `false` | output-blind event labels are not present |",
        "| Negative evidence from missing data | `false` | missingness remains NOT_EVALUABLE |",
        "",
        "## Quality gates",
        "",
        "- SANPO candidates use model-blind contiguous RGB+depth source coverage; segmentation is optional evidence.",
        "- SANPO timestamp values remain relative nominal frame-rate derivations; capture timestamp and pose-row binding are not authoritative.",
        "- Existing model-selected candidate reports remain Development discovery only.",
        "- No model output was used in SANPO candidate selection.",
        "",
        "## Candidate dataset counts",
        "",
    ]
    for dataset, count in sorted(dataset_counts.items()):
        quality.append(f"- `{dataset}`: `{count}`")
    quality.extend(["", "## Pending reasons", ""])
    for reason, count in sorted(pending_reasons.items()):
        quality.append(f"- `{reason}`: `{count}`")
    (root / "reports" / "dataset_quality_report.md").write_text("\n".join(quality) + "\n", encoding="utf-8")

    class_lines = [
        "# HFTF D7 class balance report",
        "",
        "No class balance claim is authorized because no candidate passed adjudication.",
        "",
        "| Event bucket | Admitted count | Target | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for bucket, target in TARGETS.items():
        class_lines.append(f"| {bucket} | 0 | {target} | `NOT_EVALUABLE` |")
    class_lines.append(f"| NOT_EVALUABLE | {candidate_count} | — | `pending/rejected terminal; not a negative class` |")
    (root / "reports" / "class_balance_report.md").write_text("\n".join(class_lines) + "\n", encoding="utf-8")

    review_completed: dict[str, int] = {}
    for role, relative in REVIEW_FILES.items():
        review_completed[role] = sum(1 for row in _iter_jsonl(root / relative) if row.get("record_kind") == "COMPLETED_REVIEW" and row.get("review_completed") is True)
    label_lines = [
        "# HFTF D7 label agreement report",
        "",
        "Status: `PILOT_ONLY`.",
        "",
        f"- RGB A/B/C completed reviews: `{review_completed.get('RGB_REVIEWER_A', 0)}` / `{review_completed.get('RGB_REVIEWER_B', 0)}` / `{review_completed.get('RGB_REVIEWER_C', 0)}`.",
        f"- Geometry completed reviews: `{review_completed.get('GEOMETRY_EVIDENCE_REVIEWER', 0)}`.",
        f"- Counterexample completed reviews: `{review_completed.get('COUNTEREXAMPLE_REVIEWER', 0)}`.",
        "- Final adjudications: `0` admitted events; the existing pilot terminal remains NOT_EVALUABLE.",
        "- Agreement statistics, 10% re-review, and conflict rates are not complete and are not treated as zero.",
    ]
    (root / "reports" / "label_agreement_report.md").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    (root / "reports" / "duplicate_audit_report.md").write_text(
        "\n".join([
            "# HFTF D7 duplicate audit report",
            "",
            "Status: `PARTIAL_INTAKE_ONLY`.",
            "",
            f"- Candidate rows audited for registry identity: `{candidate_count}`.",
            "- Candidate and frame IDs were preflighted for collisions before the SANPO merge.",
            "- Temporal overlap/near-duplicate parent graph remains pending independent review.",
            "- Stereo/camera-view collapse and ancestry adjacency remain review-gated.",
            "- No candidate was admitted after this merge.",
        ]) + "\n",
        encoding="utf-8",
    )


def _write_source_report(root: Path, registry: dict[str, Any], candidate_counts: Counter[str], frame_count: int, session_count: int) -> None:
    catalog = registry.get("catalog") if isinstance(registry.get("catalog"), dict) else {}
    sources = catalog.get("sources") if isinstance(catalog.get("sources"), list) else []
    source_receipts: dict[str, dict[str, Any]] = {}
    receipts_path = root / "receipts" / "source_receipts.jsonl"
    if receipts_path.is_file():
        for row in _iter_jsonl(receipts_path):
            dataset = str(row.get("dataset_id") or "")
            if dataset:
                source_receipts[dataset] = row
    stats = registry.get("source_stats") if isinstance(registry.get("source_stats"), dict) else {}
    lines = [
        "# HFTF D7 source coverage report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "This is an intake snapshot. It does not grant event truth or Confirmation authority.",
        "",
        "| Dataset | Access status | Ledger rows | RGB frames | Mask frames | Depth frames | Pose frames | Candidate windows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in sorted((item for item in sources if isinstance(item, dict)), key=lambda item: str(item.get("dataset_id"))):
        dataset = str(source.get("dataset_id"))
        receipt = source_receipts.get(dataset, {})
        status = str(receipt.get("access_status") or source.get("access_status") or "UNKNOWN")
        row_stats = stats.get(dataset) if isinstance(stats.get(dataset), dict) else {}
        lines.append(
            f"| {dataset} | {status} | {int(row_stats.get('ledger_rows', 0))} | {int(row_stats.get('rgb_frames', 0))} | {int(row_stats.get('mask_frames', 0))} | {int(row_stats.get('depth_frames', 0))} | {int(row_stats.get('pose_frames', 0))} | {candidate_counts.get(dataset, 0)} |"
        )
    lines.extend([
        "",
        f"- Top-level session rows: `{session_count}`",
        f"- Top-level candidate rows: `{sum(candidate_counts.values())}`",
        f"- Top-level frame rows: `{frame_count}`",
        "- SANPO frame-span metadata is source object evidence; it does not claim local media download or authoritative capture timestamps.",
        "- Model-selected candidate reports remain Development discovery only.",
        "- Missing tracks, ancestry, event labels, or lawful source terms remain UNKNOWN/NOT_EVALUABLE.",
    ])
    (root / "reports" / "source_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    candidate_path = root / "candidates" / "candidate_index.jsonl"
    frame_path = root / "canonical" / "frame_registry.jsonl"
    session_path = root / "manifests" / "session_registry.jsonl"
    new_candidate_path = Path(args.candidate_artifact).resolve()
    new_frame_path = Path(args.frame_artifact).resolve()
    new_session_path = Path(args.session_artifact).resolve()
    for path in (candidate_path, frame_path, session_path, new_candidate_path, new_frame_path, new_session_path):
        if not path.is_file():
            raise ContractError(f"required merge artifact missing: {path}")

    new_candidates = load_jsonl(new_candidate_path)
    new_sessions = load_jsonl(new_session_path)
    if not new_candidates or not new_sessions:
        raise ContractError("SANPO merge artifacts must not be empty")
    old_candidate_ids = _candidate_ids(candidate_path)
    new_candidate_ids: set[str] = set()
    for row in new_candidates:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in new_candidate_ids or candidate_id in old_candidate_ids:
            raise ContractError(f"candidate collision during SANPO merge: {candidate_id}")
        if row.get("dataset_id") != DATASET or row.get("model_output_visible_to_selector") is not False:
            raise ContractError(f"SANPO candidate is not model-blind: {candidate_id}")
        if row.get("event_bucket") != "NOT_EVALUABLE" or row.get("truth_status") != "NOT_EVALUABLE":
            raise ContractError(f"SANPO candidate is not an intake terminal: {candidate_id}")
        new_candidate_ids.add(candidate_id)
    old_session_ids = {str(row.get("source_session_id") or "") for row in _iter_jsonl(session_path)}
    new_session_ids: set[str] = set()
    for row in new_sessions:
        source_session_id = str(row.get("source_session_id") or "")
        if not source_session_id or source_session_id in new_session_ids or source_session_id in old_session_ids:
            raise ContractError(f"session collision during SANPO merge: {source_session_id}")
        if row.get("dataset_id") != DATASET:
            raise ContractError(f"SANPO session dataset mismatch: {source_session_id}")
        new_session_ids.add(source_session_id)
    if {str(row.get("source_session_id")) for row in new_candidates} - new_session_ids:
        raise ContractError("SANPO candidate references a session absent from its session artifact")

    old_frame_ids = _frame_ids(frame_path)
    new_frame_ids: set[str] = set()
    new_frame_count = 0
    for row in _iter_jsonl(new_frame_path):
        frame_id = str(row.get("frame_id") or "")
        if not frame_id or frame_id in new_frame_ids or frame_id in old_frame_ids:
            raise ContractError(f"frame collision during SANPO merge: {frame_id}")
        if row.get("dataset_id") != DATASET or str(row.get("source_session_id")) not in new_session_ids:
            raise ContractError(f"SANPO frame identity mismatch: {frame_id}")
        new_frame_ids.add(frame_id)
        new_frame_count += 1
    referenced_frames = {str(frame_id) for row in new_candidates for frame_id in row.get("frame_ids", [])}
    if referenced_frames != new_frame_ids:
        raise ContractError(f"SANPO candidate/frame set mismatch: referenced={len(referenced_frames)} frames={len(new_frame_ids)}")

    # Verify all existing review surfaces have the same candidate set before
    # adding rows; a pre-existing partial surface must not be made less clear.
    for relative in REVIEW_FILES.values():
        if _candidate_ids(root / relative) != old_candidate_ids:
            raise ContractError(f"existing review surface candidate set mismatch: {relative}")
    queue_path = root / "reviews" / "review_queue.jsonl"
    if _candidate_ids(queue_path) != old_candidate_ids:
        raise ContractError("existing review queue candidate set mismatch")

    backup_relative = [
        "candidates/candidate_index.jsonl",
        "canonical/frame_registry.jsonl",
        "manifests/session_registry.jsonl",
        "manifests/dataset_registry.json",
        "manifests/event_manifest.jsonl",
        "reviews/review_queue.jsonl",
        "reviews/review_a.jsonl",
        "reviews/review_b.jsonl",
        "reviews/review_c.jsonl",
        "reviews/geometry_review.jsonl",
        "reviews/counterexample_review.jsonl",
        "adjudication/final_adjudicator_assignments.jsonl",
        "adjudication/rejected_events.jsonl",
        "manifests/pending_package_manifest.json",
    ]
    backup_root = root / "manifests" / "merge_backups" / args.run_id
    if backup_root.exists():
        raise ContractError(f"merge backup already exists: {backup_root}")
    backup_root.mkdir(parents=True, exist_ok=False)
    _copy_backup(root, backup_root, backup_relative)

    reason = "INDEPENDENT_RGB_GEOMETRY_REVIEW_NOT_RUN_AND_SOURCE_ROLE_NOT_FROZEN"
    events: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reviews_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in new_candidates:
        event = _new_event_row(candidate, reason=reason)
        event_id = str(event["event_id"])
        events.append(event)
        queue.append(_new_queue_row(candidate, reason=reason, event_id=event_id))
        for role in REVIEW_ROLES:
            reviews_by_role[role].append(_new_review_row(candidate, role=role, event_id=event_id, reason=reason))
        rejected.append({
            "schema": "hftf_d7_public_real_rejected_event_v1",
            "record_kind": "NOT_EVALUABLE_TERMINAL",
            "event_id": event_id,
            "candidate_id": candidate["candidate_id"],
            "dataset_id": DATASET,
            "source_session_id": candidate["source_session_id"],
            "terminal_state": "NOT_EVALUABLE",
            "negative_evidence": False,
            "training_eligible": False,
            "confirmation_eligible": False,
            "reason": reason,
        })

    # All validation above is complete before any primary artifact is changed.
    _atomic_merge_jsonl(candidate_path, new_candidates)
    _atomic_merge_jsonl(frame_path, _iter_jsonl(new_frame_path))
    _atomic_merge_jsonl(session_path, new_sessions)
    _atomic_merge_jsonl(root / "manifests" / "event_manifest.jsonl", events)
    _atomic_merge_jsonl(queue_path, queue)
    for role, relative in REVIEW_FILES.items():
        _atomic_merge_jsonl(root / relative, reviews_by_role[role])
    _atomic_merge_jsonl(root / "adjudication" / "rejected_events.jsonl", rejected)

    registry_path = root / "manifests" / "dataset_registry.json"
    registry = load_json(registry_path)
    if not isinstance(registry, dict):
        raise ContractError("dataset registry is not an object")
    discovery = registry.setdefault("candidate_discovery", {})
    if isinstance(discovery, dict):
        discovery["total_candidate_count"] = int(discovery.get("total_candidate_count", len(old_candidate_ids))) + len(new_candidates)
        discovery["total_frame_count"] = int(discovery.get("total_frame_count", 0)) + new_frame_count
        imports = discovery.setdefault("imports", [])
        if isinstance(imports, list):
            imports.append({
                "authority": "DEVELOPMENT_DISCOVERY_ONLY",
                "candidate_count": len(new_candidates),
                "frame_count": new_frame_count,
                "model_blind": True,
                "path": str(new_candidate_path),
                "receipt_sha256": sha256_file(new_candidate_path),
                "source": "SANPO-Real/public-gcs-frame-span",
            })
    registry["session_count"] = int(registry.get("session_count", len(old_session_ids))) + len(new_sessions)
    source_stats = registry.setdefault("source_stats", {})
    if isinstance(source_stats, dict):
        sanpo_stats = source_stats.setdefault(DATASET, {})
        if isinstance(sanpo_stats, dict):
            sanpo_stats["candidate_windows"] = int(sanpo_stats.get("candidate_windows", 0)) + len(new_candidates)
            sanpo_stats["frame_span_complete_frames"] = int(sanpo_stats.get("frame_span_complete_frames", 0)) + int(new_frame_count)
            sanpo_stats["candidate_rgb_depth_frames"] = int(sanpo_stats.get("candidate_rgb_depth_frames", 0)) + int(new_frame_count)
    write_json(registry_path, registry)

    # Refresh assignment-surface manifests and reports from the now-current
    # primary rows.  These reports still cannot authorize any label or split.
    all_candidate_counts: Counter[str] = Counter()
    all_pending_reasons: Counter[str] = Counter()
    all_candidate_count = 0
    for row in _iter_jsonl(candidate_path):
        all_candidate_count += 1
        all_candidate_counts[str(row.get("dataset_id") or "UNKNOWN")] += 1
    for row in _iter_jsonl(root / "manifests" / "event_manifest.jsonl"):
        all_pending_reasons[str(row.get("not_evaluable_reason") or "UNKNOWN")] += 1
    _write_assignment_reports(root, candidate_count=all_candidate_count, dataset_counts=all_candidate_counts, pending_reasons=all_pending_reasons)
    _write_source_report(root, registry, all_candidate_counts, frame_count=len(old_frame_ids) + new_frame_count, session_count=len(old_session_ids) + len(new_sessions))

    pending_path = root / "manifests" / "pending_package_manifest.json"
    pending = load_json(pending_path)
    if not isinstance(pending, dict):
        raise ContractError("pending package manifest is not an object")
    pending.update({
        "generated_at_utc": utc_now(),
        "status": "NOT_COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "candidate_count": all_candidate_count,
        "admitted_parent_event_count": 0,
        "source_session_count": len(old_session_ids) + len(new_sessions),
        "source_coverage": {
            dataset: {"candidates": count, "not_evaluable": count, "admitted": 0}
            for dataset, count in sorted(all_candidate_counts.items())
        },
        "candidate_index_sha256": sha256_file(candidate_path),
        "session_registry_sha256": sha256_file(session_path),
        "review_assignments_are_not_labels": True,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
    })
    write_json(pending_path, pending)

    receipt_path = root / "receipts" / f"sanpo_candidate_merge_receipt_{args.run_id}.json"
    if receipt_path.exists():
        raise ContractError(f"merge receipt already exists: {receipt_path}")
    receipt = {
        "schema": "hftf_d7_public_real_sanpo_candidate_merge_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "dataset_id": DATASET,
        "status": "MERGED_ASSIGNMENT_ONLY_NO_EVENT_TRUTH",
        "new_candidate_count": len(new_candidates),
        "new_frame_count": new_frame_count,
        "new_session_count": len(new_sessions),
        "top_level_candidate_count": all_candidate_count,
        "top_level_frame_count": len(old_frame_ids) + new_frame_count,
        "top_level_session_count": len(old_session_ids) + len(new_sessions),
        "admitted_parent_events": 0,
        "selection_authority": "MODEL_BLIND_UNIFORM_CONTIGUOUS_RGB_DEPTH",
        "event_truth_authority": False,
        "review_assignments_are_not_labels": True,
        "backup_root": str(backup_root),
        "candidate_index_sha256": sha256_file(candidate_path),
        "frame_registry_sha256": sha256_file(frame_path),
        "session_registry_sha256": sha256_file(session_path),
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--frame-artifact", required=True)
    parser.add_argument("--session-artifact", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
