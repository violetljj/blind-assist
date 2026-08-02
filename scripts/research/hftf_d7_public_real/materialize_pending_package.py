#!/usr/bin/env python3
"""Create the complete D7 output surface without fabricating review labels.

The command materializes candidate event shells, isolated review assignments,
and explicit ``NOT_EVALUABLE`` terminals.  It intentionally writes no admitted
event.  A later lawful review run must replace assignment records with signed
review outputs and pass the validator before any split can contain an event.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json


REVIEW_ROLES = (
    "RGB_REVIEWER_A",
    "RGB_REVIEWER_B",
    "RGB_REVIEWER_C",
    "GEOMETRY_EVIDENCE_REVIEWER",
    "COUNTEREXAMPLE_REVIEWER",
    "FINAL_ADJUDICATOR",
)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"JSONL row is not an object: {path}:{line_number}")
            yield row


def _write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _reason(candidate: dict[str, Any], *, output_root: Path) -> str:
    dataset_id = str(candidate.get("dataset_id", "UNKNOWN"))
    if dataset_id == "EgoWalk":
        rgb_receipts = sorted((output_root / "receipts").glob("egowalk_rgb_receipt_*.json"))
        if rgb_receipts:
            latest = load_json(rgb_receipts[-1])
            if isinstance(latest, dict) and latest.get("status") == "PUBLIC_EXTRACTED_RGB_DOWNLOADED":
                return "INDEPENDENT_RGB_GEOMETRY_REVIEW_NOT_RUN"
        return "RAW_RGB_OR_EXTRACTED_VIDEO_REVIEW_NOT_LAWFULLY_CLOSED"
    if dataset_id in {"SANPO-Real", "JRDB", "THOR", "THOR-MAGNI"}:
        return "INDEPENDENT_RGB_GEOMETRY_REVIEW_NOT_RUN_AND_SOURCE_ROLE_NOT_FROZEN"
    if str(candidate.get("selection_role", "")).startswith("DEVELOPMENT"):
        return "DISCOVERY_ONLY_CANDIDATE_CANNOT_BE_PROMOTED"
    return "INDEPENDENT_REVIEW_NOT_RUN"


def _rgb_local_path(candidate: dict[str, Any], *, output_root: Path) -> str | None:
    if str(candidate.get("dataset_id")) != "EgoWalk":
        return None
    source_id = str(candidate.get("source_id") or "")
    if not source_id:
        return None
    path = output_root / "raw" / "egowalk-rgb" / f"{source_id}__rgb.mp4"
    return str(path) if path.is_file() else None


def _source_coverage(candidates: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: {"candidates": 0, "not_evaluable": 0, "admitted": 0})
    for row in candidates:
        dataset = str(row.get("dataset_id", "UNKNOWN"))
        coverage[dataset]["candidates"] += 1
        coverage[dataset]["not_evaluable"] += 1
    return {key: dict(value) for key, value in sorted(coverage.items())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    candidate_path = root / "candidates" / "candidate_index.jsonl"
    session_path = root / "manifests" / "session_registry.jsonl"
    catalog_path = root / "manifests" / "dataset_registry.json"
    for path in (candidate_path, session_path, catalog_path):
        if not path.is_file():
            raise ContractError(f"required D7 input missing: {path}")
    sessions = {str(row.get("source_session_id")): row for row in load_jsonl(session_path)}
    registry = load_json(catalog_path)
    if not isinstance(registry, dict):
        raise ContractError("dataset registry is not an object")
    catalog = registry.get("catalog") if isinstance(registry.get("catalog"), dict) else {}
    catalog_sources = {
        str(item.get("dataset_id")): item
        for item in catalog.get("sources", [])
        if isinstance(item, dict) and item.get("dataset_id")
    }
    for directory in ("manifests", "reviews", "adjudication", "splits", "reports"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    output_names = {
        "event_manifest": root / "manifests" / "event_manifest.jsonl",
        "review_a": root / "reviews" / "review_a.jsonl",
        "review_b": root / "reviews" / "review_b.jsonl",
        "review_c": root / "reviews" / "review_c.jsonl",
        "geometry": root / "reviews" / "geometry_review.jsonl",
        "counterexample": root / "reviews" / "counterexample_review.jsonl",
        "final_assignment": root / "adjudication" / "final_adjudicator_assignments.jsonl",
        "adjudicated": root / "adjudication" / "adjudicated_events.jsonl",
        "rejected": root / "adjudication" / "rejected_events.jsonl",
        "queue": root / "reviews" / "review_queue.jsonl",
    }
    handles = {key: path.open("w", encoding="utf-8", newline="\n") for key, path in output_names.items()}
    candidate_count = 0
    source_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    pending_reasons: Counter[str] = Counter()
    try:
        for candidate in _iter_jsonl(candidate_path):
            candidate_id = str(candidate.get("candidate_id", ""))
            if not candidate_id:
                raise ContractError("candidate row missing candidate_id")
            source_session_id = str(candidate.get("source_session_id", ""))
            if not source_session_id or source_session_id not in sessions:
                raise ContractError(f"candidate references unknown source session: {candidate_id} -> {source_session_id}")
            dataset_id = str(candidate.get("dataset_id", "UNKNOWN"))
            parent_event_id = str(candidate.get("parent_event_id") or stable_id("d7parent", candidate_id))
            reason = _reason(candidate, output_root=root)
            frame_ids = candidate.get("frame_ids") if isinstance(candidate.get("frame_ids"), list) else []
            rgb_local_path = _rgb_local_path(candidate, output_root=root)
            source_license = (
                catalog_sources.get(dataset_id, {}).get("license")
                or candidate.get("source_license")
                or sessions[source_session_id].get("source_license_status")
            )
            base = {
                "schema": "hftf_d7_public_real_event_manifest_v1",
                "record_kind": "CANDIDATE_EVENT_SHELL",
                "event_id": parent_event_id,
                "parent_event_id": parent_event_id,
                "candidate_id": candidate_id,
                "dataset_id": dataset_id,
                "source_session_id": source_session_id,
                "ancestry_group": candidate.get("ancestry_group") or sessions[source_session_id].get("ancestry_group"),
                "frame_ids": frame_ids,
                "start_timestamp_ns": candidate.get("start_timestamp_ns"),
                "end_timestamp_ns": candidate.get("end_timestamp_ns"),
                "event_bucket": "NOT_EVALUABLE",
                "truth_status": "NOT_EVALUABLE",
                "admission_status": "PENDING_REVIEW",
                "review_state": "NOT_RUN",
                "pre_interval": None,
                "alertable_interval": None,
                "passed_clearance_interval": None,
                "continuous_negative_interval": None,
                "candidate_selection_model_visible": bool(candidate.get("model_output_visible_to_selector", False)),
                "review_model_output_visible": False,
                "geometry_model_output_visible": False,
                "not_evaluable_reason": reason,
                "required_review_roles": list(REVIEW_ROLES),
                "source_license": source_license,
                "source_hash": candidate.get("source_hash"),
                "rgb_local_path": rgb_local_path,
            }
            _write_row(handles["event_manifest"], base)
            _write_row(handles["queue"], {
                "schema": "hftf_d7_public_real_review_assignment_v1",
                "record_kind": "ASSIGNMENT_ONLY",
                "event_id": parent_event_id,
                "candidate_id": candidate_id,
                "dataset_id": dataset_id,
                "source_session_id": source_session_id,
                "roles": list(REVIEW_ROLES),
                "rgb_model_output_visible": False,
                "geometry_model_output_visible": False,
                "assignment_status": "PENDING",
                "not_evaluable_reason": reason,
            })
            for role, key in (
                ("RGB_REVIEWER_A", "review_a"),
                ("RGB_REVIEWER_B", "review_b"),
                ("RGB_REVIEWER_C", "review_c"),
                ("GEOMETRY_EVIDENCE_REVIEWER", "geometry"),
                ("COUNTEREXAMPLE_REVIEWER", "counterexample"),
                ("FINAL_ADJUDICATOR", "final_assignment"),
            ):
                row = {
                    "schema": "hftf_d7_public_real_review_record_v1",
                    "record_kind": "ASSIGNMENT_ONLY",
                    "review_role": role,
                    "event_id": parent_event_id,
                    "candidate_id": candidate_id,
                    "dataset_id": dataset_id,
                    "source_session_id": source_session_id,
                    "review_completed": False,
                    "decision": "PENDING",
                    "event_bucket": None,
                    "phase_intervals": None,
                    "model_output_visible": False,
                    "source_native_geometry_only": role == "GEOMETRY_EVIDENCE_REVIEWER",
                    "counterexample_search_required": role == "COUNTEREXAMPLE_REVIEWER",
                    "not_evaluable_reason": reason,
                    "rgb_local_path": rgb_local_path,
                }
                _write_row(handles[key], row)
            _write_row(handles["rejected"], {
                "schema": "hftf_d7_public_real_rejected_event_v1",
                "record_kind": "NOT_EVALUABLE_TERMINAL",
                "event_id": parent_event_id,
                "candidate_id": candidate_id,
                "dataset_id": dataset_id,
                "source_session_id": source_session_id,
                "terminal_state": "NOT_EVALUABLE",
                "negative_evidence": False,
                "training_eligible": False,
                "confirmation_eligible": False,
                "reason": reason,
            })
            candidate_count += 1
            source_counts[dataset_id] += 1
            session_counts[source_session_id] += 1
            pending_reasons[reason] += 1
    finally:
        for handle in handles.values():
            handle.close()

    empty_adjudicated = output_names["adjudicated"].stat().st_size == 0
    if not empty_adjudicated:
        raise ContractError("pending package unexpectedly emitted adjudicated rows")

    split_payload = {
        "schema": "hftf_d7_public_real_split_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "status": "BLOCKED_NO_ADMITTED_EVENTS",
        "training_authorized": False,
        "confirmation_authorized": False,
        "counts": {"train_development": 0, "model_selection_development": 0, "same_domain_confirmation": 0, "cross_dataset_confirmation": 0},
        "event_ids": {"train_development": [], "model_selection_development": [], "same_domain_confirmation": [], "cross_dataset_confirmation": []},
        "reason": "No event has completed the independent RGB A/B/C, geometry, counterexample, and adjudication gates.",
        "session_disjoint_check": {"status": "VACUOUS", "overlap_count": 0},
    }
    write_json(root / "splits" / "development_split.json", split_payload)
    write_json(root / "splits" / "confirmation_split.json", {**split_payload, "schema": "hftf_d7_public_real_confirmation_split_v1"})
    write_json(root / "splits" / "leave_one_dataset_out_splits.json", {
        "schema": "hftf_d7_public_real_leave_one_dataset_out_v1",
        "status": "BLOCKED_NO_ADMITTED_EVENTS",
        "training_authorized": False,
        "splits": {},
        "reason": "No adjudicated event can enter a dataset-held-out role.",
    })
    write_json(root / "manifests" / "review_role_manifest.json", {
        "schema": "hftf_d7_public_real_review_role_manifest_v1",
        "generated_at_utc": utc_now(),
        "review_roles": list(REVIEW_ROLES),
        "assignments_are_decisions": False,
        "rgb_reviewers_model_output_visible": False,
        "geometry_reviewer_model_output_visible": False,
        "counterexample_reviewer_model_output_visible": False,
        "final_adjudicator_input_includes_all_raw_reviews": True,
        "status": "ASSIGNMENT_SURFACE_ONLY",
    })

    total_candidate_count = candidate_count
    dataset_quality = [
        "# HFTF D7 dataset quality report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Terminal",
        "",
        "`NOT_COMPLETE`: this run has candidate coverage and review assignment surfaces, but zero admitted parent events.",
        "",
        "| Check | Result | Evidence |",
        "| --- | --- | --- |",
        f"| Candidate windows | `{total_candidate_count}` | `candidates/candidate_index.jsonl` |",
        "| Admitted parent events | `0` | `adjudication/adjudicated_events.jsonl` is empty |",
        "| Event truth | `NOT_AUTHORIZED` | no completed review/adjudication gate |",
        "| Training eligibility | `false` | all splits are blocked |",
        "| Confirmation eligibility | `false` | output-blind event labels not present |",
        "| Negative evidence from missing data | `false` | missingness is preserved as NOT_EVALUABLE |",
        "",
        "## Quality gates",
        "",
        "- Timestamp monotonicity and pose-gap cuts are recorded for EgoWalk metadata; they do not establish obstacle truth.",
        "- Existing model-selected candidate reports remain Development discovery only.",
        "- No model output was read during EgoWalk uniform coverage selection.",
        "",
        "## Pending reasons",
        "",
    ]
    egowalk_rgb_receipts = sorted((root / "receipts").glob("egowalk_rgb_receipt_*.json"))
    if egowalk_rgb_receipts:
        latest_rgb_receipt = load_json(egowalk_rgb_receipts[-1])
        if isinstance(latest_rgb_receipt, dict) and latest_rgb_receipt.get("status") == "PUBLIC_EXTRACTED_RGB_DOWNLOADED":
            dataset_quality.insert(
                dataset_quality.index("- Existing model-selected candidate reports remain Development discovery only."),
                "- EgoWalk extracted RGB has a complete public-repository receipt; raw-recordings access remains blocked and RGB still requires independent review.",
            )
        else:
            dataset_quality.insert(
                dataset_quality.index("- Existing model-selected candidate reports remain Development discovery only."),
                "- EgoWalk extracted RGB download is partial or unresolved; raw-recordings access remains blocked.",
            )
    else:
        dataset_quality.insert(
            dataset_quality.index("- Existing model-selected candidate reports remain Development discovery only."),
            "- Source license/access remains unresolved for EgoWalk extracted media and gated for raw recordings.",
        )
    for reason, count in sorted(pending_reasons.items()):
        dataset_quality.append(f"- `{reason}`: `{count}`")
    (root / "reports" / "dataset_quality_report.md").write_text("\n".join(dataset_quality) + "\n", encoding="utf-8")

    class_lines = [
        "# HFTF D7 class balance report",
        "",
        "No class balance claim is authorized because no candidate passed adjudication.",
        "",
        "| Event bucket | Admitted count | Target | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    targets = {
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
    for bucket, target in targets.items():
        class_lines.append(f"| {bucket} | 0 | {target} | `NOT_EVALUABLE` |")
    class_lines.append("| NOT_EVALUABLE | " + str(total_candidate_count) + " | — | `pending/rejected terminal; not a negative class` |")
    (root / "reports" / "class_balance_report.md").write_text("\n".join(class_lines) + "\n", encoding="utf-8")

    (root / "reports" / "duplicate_audit_report.md").write_text(
        "\n".join([
            "# HFTF D7 duplicate audit report",
            "",
            "Status: `PARTIAL_INTAKE_ONLY`.",
            "",
            f"- Candidate rows audited for registry identity: `{total_candidate_count}`.",
            "- Candidate IDs and frame IDs were checked for duplicate IDs during the EgoWalk merge.",
            "- Temporal overlap/near-duplicate graph has not been admitted as event truth because RGB review and cross-source ancestry are incomplete.",
            "- Stereo/camera-view collapse is not applicable to the extracted EgoWalk trajectory-only intake; SANPO/JRDB source-native view collapse remains pending.",
            "- No candidate was admitted after this report.",
        ]) + "\n", encoding="utf-8")

    (root / "reports" / "label_agreement_report.md").write_text(
        "\n".join([
            "# HFTF D7 label agreement report",
            "",
            "Status: `NOT_RUN`.",
            "",
            "- RGB A/B/C completed reviews: `0`.",
            "- Geometry completed reviews: `0`.",
            "- Counterexample completed reviews: `0`.",
            "- Final adjudications: `0`.",
            "- Agreement, Cohen/Fleiss statistics, 10% re-review, and conflict rates are therefore undefined, not zero.",
        ]) + "\n", encoding="utf-8")

    write_json(root / "manifests" / "pending_package_manifest.json", {
        "schema": "hftf_d7_public_real_pending_package_manifest_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "generated_at_utc": utc_now(),
        "status": "NOT_COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "candidate_count": total_candidate_count,
        "admitted_parent_event_count": 0,
        "source_session_count": len(session_counts),
        "source_coverage": _source_coverage(list(_iter_jsonl(candidate_path))),
        "required_outputs": {key: str(path) for key, path in output_names.items()},
        "review_assignments_are_not_labels": True,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
        "candidate_index_sha256": sha256_file(candidate_path),
        "session_registry_sha256": sha256_file(session_path),
    })
    return {
        "schema": "hftf_d7_public_real_pending_package_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "candidate_count": total_candidate_count,
        "admitted_parent_events": 0,
        "source_sessions": len(session_counts),
        "status": "NOT_COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "training_authorized": False,
        "confirmation_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
