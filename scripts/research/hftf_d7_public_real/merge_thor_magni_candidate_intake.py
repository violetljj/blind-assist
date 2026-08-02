#!/usr/bin/env python3
"""Merge THOR-MAGNI synchronized window intake into D7 assignment surfaces.

This command is append-only at the logical-record level.  It accepts only
source-native THOR-MAGNI window manifests whose QTM time/SceneFNr binding has
already been audited, preflights all candidate/frame/session/review collisions,
and then appends NOT_EVALUABLE event shells plus assignment-only review rows.
It never creates event truth, an admitted parent event, or a split assignment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json


DATASET = "THOR-MAGNI"
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


def _atomic_append(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            if path.is_file():
                with path.open("r", encoding="utf-8") as source:
                    shutil.copyfileobj(source, handle)
            for row in rows:
                handle.write(_line(row))
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _id_set(path: Path, field: str) -> set[str]:
    values: set[str] = set()
    for row in _iter_jsonl(path):
        value = str(row.get(field) or "")
        if not value or value in values:
            raise ContractError(f"duplicate or missing {field} in {path}: {value}")
        values.add(value)
    return values


def _assert_model_blind(value: Any, *, path: str = "$") -> None:
    forbidden = {
        "model_hint",
        "candidate_selection_model_visible",
        "model_output_visible_to_selector",
        "native_geometry_used_for_selection",
        "required_confirmation_selection",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                if key == "candidate_selection_model_visible" and child is False:
                    pass
                elif key == "model_output_visible_to_selector" and child is False:
                    pass
                elif key == "native_geometry_used_for_selection" and child is False:
                    pass
                elif key == "required_confirmation_selection" and child == "MODEL_BLIND":
                    pass
                else:
                    raise ContractError(f"forbidden model/discovery field drift: {path}.{key}")
            if key in {"model_output_visible", "review_model_output_visible", "geometry_model_output_visible"} and child is not False:
                raise ContractError(f"model visibility is not false: {path}.{key}")
            _assert_model_blind(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_model_blind(child, path=f"{path}[{index}]")


def _load_thor_source_receipt(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "receipts" / "source_receipts.jsonl"
    rows = [row for row in _iter_jsonl(path) if row.get("dataset_id") == DATASET]
    if len(rows) != 1:
        raise ContractError(f"expected exactly one THOR-MAGNI source receipt: {path}")
    row = rows[0]
    if row.get("license") != "CC-BY-4.0":
        raise ContractError(f"THOR-MAGNI source license is not verified as CC-BY-4.0: {row.get('license')}")
    if row.get("event_truth_authority") is not False:
        raise ContractError("THOR-MAGNI source receipt grants event truth authority")
    return path, row


def _load_archive_receipt(root: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted((root / "receipts").glob("thor_magni_archive_receipt_*.json"))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        value = load_json(path)
        if isinstance(value, dict) and value.get("dataset_id") == DATASET:
            matches.append((path, value))
    if not matches:
        raise ContractError("expected at least one THOR-MAGNI archive receipt")
    fingerprints = {
        (
            value.get("archive_checksum"),
            value.get("archive_size"),
            value.get("record_id"),
            value.get("license"),
        )
        for _, value in matches
    }
    if len(fingerprints) != 1:
        raise ContractError("THOR-MAGNI archive receipts disagree on checksum/size/revision/license")
    def generated_at(item: tuple[Path, dict[str, Any]]) -> datetime:
        value = str(item[1].get("generated_at_utc") or "")
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

    path, receipt = max(matches, key=generated_at)
    if receipt.get("archive_checksum") is None or receipt.get("license") != "CC-BY-4.0":
        raise ContractError(f"THOR-MAGNI archive receipt lacks checksum/license: {path}")
    return path, receipt


def _bind_window_receipt(root: Path, manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "receipts").glob("thor_magni_window_receipt_*.json")):
        value = load_json(path)
        if not isinstance(value, dict):
            continue
        bound = str(value.get("manifest_path") or "")
        if bound and Path(bound).resolve() == manifest_path.resolve():
            matches.append((path, value))
    if len(matches) != 1:
        raise ContractError(f"expected exactly one window receipt for {manifest_path}, found {len(matches)}")
    return matches[0]


def _bind_member_receipt(
    root: Path,
    *,
    manifest: dict[str, Any],
    archive_receipt: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    scenario_hash = str(manifest.get("scenario_csv_sha256") or "")
    rgb_hash = str(manifest.get("rgb_sha256") or "")
    scenario_path = Path(str(manifest.get("scenario_csv") or "")).resolve()
    rgb_path = Path(str(manifest.get("rgb_path") or "")).resolve()
    if not scenario_hash or not rgb_hash or not scenario_path.is_file() or not rgb_path.is_file():
        raise ContractError(f"THOR-MAGNI window lacks materialized scenario/RGB members: {manifest.get('run_id')}")
    matches: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for path in sorted((root / "receipts").glob("thor_magni_member_receipt_*.json")):
        value = load_json(path)
        if not isinstance(value, dict) or value.get("dataset_id") != DATASET:
            continue
        if value.get("archive_checksum") != archive_receipt.get("archive_checksum"):
            continue
        members = value.get("members")
        if not isinstance(members, list):
            continue
        scenario_members = [member for member in members if isinstance(member, dict) and member.get("sha256") == scenario_hash]
        rgb_members = [member for member in members if isinstance(member, dict) and member.get("sha256") == rgb_hash]
        if len(scenario_members) != 1 or len(rgb_members) != 1:
            continue
        scenario_member = scenario_members[0]
        rgb_member = rgb_members[0]
        if Path(str(scenario_member.get("local_path") or "")).resolve() != scenario_path:
            continue
        if Path(str(rgb_member.get("local_path") or "")).resolve() != rgb_path:
            continue
        if scenario_member.get("source_revision") != archive_receipt.get("archive_checksum") or rgb_member.get("source_revision") != archive_receipt.get("archive_checksum"):
            raise ContractError(f"THOR-MAGNI selected member source revision drift: {path}")
        if sha256_file(scenario_path) != scenario_hash or sha256_file(rgb_path) != rgb_hash:
            raise ContractError(f"materialized THOR-MAGNI member hash drift: {path}")
        matches.append((path, value, scenario_member, rgb_member))
    if len(matches) != 1:
        raise ContractError(f"expected exactly one selected-member receipt for {manifest.get('run_id')}, found {len(matches)}")
    return matches[0]


def _load_artifacts(
    root: Path,
    manifest_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not manifest_paths:
        raise ContractError("at least one --window-manifest is required")
    source_receipt_path, source_receipt = _load_thor_source_receipt(root)
    archive_receipt_path, archive_receipt = _load_archive_receipt(root)
    all_candidates: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("dataset_id") != DATASET:
            raise ContractError(f"invalid THOR-MAGNI window manifest: {manifest_path}")
        window_receipt_path, window_receipt = _bind_window_receipt(root, manifest_path)
        if window_receipt.get("dataset_id") != DATASET or window_receipt.get("event_truth_authority") is not False:
            raise ContractError(f"window receipt is not development-only: {window_receipt_path}")
        if window_receipt.get("manifest_sha256") != sha256_file(manifest_path):
            raise ContractError(f"window manifest hash drift: {manifest_path}")
        if window_receipt.get("counts", {}).get("candidate_windows") != manifest.get("candidate_count"):
            raise ContractError(f"window candidate count receipt drift: {manifest_path}")
        if window_receipt.get("counts", {}).get("frame_rows") != manifest.get("frame_count"):
            raise ContractError(f"window frame count receipt drift: {manifest_path}")
        candidate_path = Path(str(window_receipt.get("candidate_index_path") or "")).resolve()
        frame_path = Path(str(window_receipt.get("frame_registry_path") or "")).resolve()
        for path in (candidate_path, frame_path):
            if not path.is_file():
                raise ContractError(f"window manifest points to missing artifact: {path}")
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ContractError(f"THOR-MAGNI artifact escapes output root: {path}") from exc
        candidates = load_jsonl(candidate_path)
        frames = load_jsonl(frame_path)
        if not candidates or not frames:
            raise ContractError(f"THOR-MAGNI window artifact is empty: {manifest_path}")
        expected_candidates = int(manifest.get("candidate_count", -1))
        expected_frames = int(manifest.get("frame_count", -1))
        if expected_candidates != len(candidates) or expected_frames != len(frames):
            raise ContractError(f"window manifest counts drifted: {manifest_path}")
        session_id = str(manifest.get("source_session_id") or "")
        ancestry = str(manifest.get("ancestry_group") or "")
        if not session_id or not ancestry:
            raise ContractError(f"window manifest lacks source identity: {manifest_path}")
        member_receipt_path, member_receipt, scenario_member, rgb_member = _bind_member_receipt(
            root,
            manifest=manifest,
            archive_receipt=archive_receipt,
        )
        source_evidence = {
            str(Path(str(value)).resolve())
            for value in (source_receipt.get("local_evidence_paths") or [])
            if value
        }
        if str(member_receipt_path.resolve()) not in source_evidence:
            raise ContractError(f"THOR-MAGNI member receipt is not bound into source receipt: {member_receipt_path}")
        if member_receipt.get("status") != "PUBLIC_SELECTED_MEMBERS_MATERIALIZED":
            raise ContractError(f"selected THOR-MAGNI members are not materialized: {member_receipt_path}")
        if member_receipt.get("archive_checksum") != archive_receipt.get("archive_checksum"):
            raise ContractError(f"THOR-MAGNI archive checksum drift: {member_receipt_path}")
        if member_receipt.get("record_id") != archive_receipt.get("record_id"):
            raise ContractError(f"THOR-MAGNI source revision drift: {member_receipt_path}")
        source_hashes = [
            str(value)
            for value in (manifest.get("source_hash"), manifest.get("scenario_csv_sha256"), manifest.get("rgb_sha256"))
            if value
        ]
        sessions.append({
            "schema": "hftf_d7_public_real_session_v1",
            "dataset_id": DATASET,
            "source_session_id": session_id,
            "ancestry_group": ancestry,
            "session_root": manifest.get("scenario_csv"),
            "data_role": "DEVELOPMENT_CANDIDATE_DISCOVERY",
            "history_roles": ["thor_magni_public_selected_member_intake"],
            "source_license_status": "CC-BY-4.0_MEMBER_TERMS_REVIEW_REQUIRED",
            "rgb_count": int(manifest.get("unique_scene_frame_count", 0) or 0),
            "mask_count": 0,
            "depth_count": 0,
            "pose_count": len(frames),
            "source_hashes": sorted(set(source_hashes)),
            "source_truth_status": "METADATA_GEOMETRY_ONLY",
            "candidate_count": len(candidates),
        })
        for candidate in candidates:
            _validate_candidate(candidate, session_id=session_id, ancestry=ancestry)
        for frame in frames:
            _validate_frame(frame, session_id=session_id, ancestry=ancestry)
        all_candidates.extend(candidates)
        all_frames.extend(frames)
        manifest_rows.append({
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "frame_path": str(frame_path),
            "frame_sha256": sha256_file(frame_path),
            "source_session_id": session_id,
            "ancestry_group": ancestry,
            "candidate_count": len(candidates),
            "frame_count": len(frames),
            "file_id": manifest.get("file_id"),
            "rgb_sha256": manifest.get("rgb_sha256"),
            "scenario_csv_sha256": manifest.get("scenario_csv_sha256"),
            "window_receipt_path": str(window_receipt_path.resolve()),
            "window_receipt_sha256": sha256_file(window_receipt_path),
            "member_receipt_path": str(member_receipt_path.resolve()),
            "member_receipt_sha256": sha256_file(member_receipt_path),
            "archive_receipt_path": str(archive_receipt_path.resolve()),
            "archive_receipt_sha256": sha256_file(archive_receipt_path),
            "source_receipt_path": str(source_receipt_path.resolve()),
            "source_receipt_sha256": sha256_file(source_receipt_path),
            "archive_checksum": archive_receipt.get("archive_checksum"),
            "source_license": source_receipt.get("license"),
            "scenario_member_id": scenario_member.get("member_id"),
            "rgb_member_id": rgb_member.get("member_id"),
            "scenario_member_sha256": scenario_member.get("sha256"),
            "rgb_member_sha256": rgb_member.get("sha256"),
        })
    candidate_by_id = {str(row.get("candidate_id")): row for row in all_candidates}
    frame_by_id = {str(row.get("frame_id")): row for row in all_frames}
    if len(candidate_by_id) != len(all_candidates) or len(frame_by_id) != len(all_frames):
        raise ContractError("THOR-MAGNI candidate/frame IDs are not globally unique")
    frame_reference_counts: Counter[str] = Counter()
    candidates_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in all_candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        frame_ids = [str(value) for value in candidate.get("frame_ids", [])]
        if len(frame_ids) != len(set(frame_ids)):
            raise ContractError(f"THOR-MAGNI candidate repeats frame IDs: {candidate_id}")
        candidates_by_session[str(candidate["source_session_id"])].append(candidate)
        for frame_id in frame_ids:
            frame = frame_by_id.get(frame_id)
            if frame is None:
                raise ContractError(f"THOR-MAGNI candidate references missing frame: {candidate_id}/{frame_id}")
            if frame.get("source_session_id") != candidate.get("source_session_id") or frame.get("ancestry_group") != candidate.get("ancestry_group"):
                raise ContractError(f"THOR-MAGNI frame ownership drift: {candidate_id}/{frame_id}")
            frame_reference_counts[frame_id] += 1
    for session_id, session_candidates in candidates_by_session.items():
        ordered = sorted(session_candidates, key=lambda row: (int(row["start_timestamp_ns"]), int(row["end_timestamp_ns"]), str(row["candidate_id"])))
        for previous, current in zip(ordered, ordered[1:]):
            if int(current["start_timestamp_ns"]) < int(previous["end_timestamp_ns"]):
                raise ContractError(f"THOR-MAGNI source-time windows overlap within session: {session_id}")
    referenced_frame_ids = set(frame_reference_counts)
    for manifest_row in manifest_rows:
        session_id = manifest_row["source_session_id"]
        manifest_frame_ids = {str(row.get("frame_id")) for row in all_frames if row.get("source_session_id") == session_id}
        manifest_referenced = {
            frame_id for frame_id in referenced_frame_ids if frame_by_id[frame_id].get("source_session_id") == session_id
        }
        manifest_row["referenced_frame_count"] = len(manifest_referenced)
        manifest_row["unreferenced_frame_count"] = len(manifest_frame_ids - manifest_referenced)
        manifest_row["reused_frame_reference_count"] = sum(max(0, count - 1) for frame_id, count in frame_reference_counts.items() if frame_by_id[frame_id].get("source_session_id") == session_id)
    return all_candidates, all_frames, sessions, manifest_rows


def _validate_candidate(candidate: dict[str, Any], *, session_id: str, ancestry: str) -> None:
    if candidate.get("schema") != "hftf_d7_public_real_candidate_v1":
        raise ContractError(f"THOR-MAGNI candidate schema drift: {candidate.get('candidate_id')}")
    if not candidate.get("candidate_id") or not candidate.get("parent_event_id"):
        raise ContractError(f"THOR-MAGNI candidate lacks candidate/parent identity: {candidate.get('candidate_id')}")
    if candidate.get("dataset_id") != DATASET or candidate.get("source_session_id") != session_id or candidate.get("ancestry_group") != ancestry:
        raise ContractError(f"THOR-MAGNI candidate identity drift: {candidate.get('candidate_id')}")
    if candidate.get("event_bucket") != "NOT_EVALUABLE" or candidate.get("truth_status") != "NOT_EVALUABLE":
        raise ContractError(f"THOR-MAGNI candidate is not intake-only: {candidate.get('candidate_id')}")
    if candidate.get("model_output_visible_to_selector") is not False or candidate.get("native_geometry_used_for_selection") is not False:
        raise ContractError(f"THOR-MAGNI candidate is not model-blind: {candidate.get('candidate_id')}")
    if candidate.get("candidate_selection") != "MODEL_BLIND_UNIFORM_QTM_100HZ_SOURCE_COVERAGE":
        raise ContractError(f"unexpected THOR-MAGNI candidate selection: {candidate.get('candidate_id')}")
    if candidate.get("timestamp_semantics") != "SOURCE_SYNCHRONIZED_QTM_TIME":
        raise ContractError(f"THOR-MAGNI candidate timestamp semantics drift: {candidate.get('candidate_id')}")
    if not candidate.get("rgb_uri") or not candidate.get("source_hash"):
        raise ContractError(f"THOR-MAGNI candidate lacks source binding: {candidate.get('candidate_id')}")
    try:
        if int(candidate["start_timestamp_ns"]) >= int(candidate["end_timestamp_ns"]):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid THOR-MAGNI candidate time range: {candidate.get('candidate_id')}") from exc
    frame_ids = candidate.get("frame_ids")
    if not isinstance(frame_ids, list) or not frame_ids:
        raise ContractError(f"THOR-MAGNI candidate lacks frame_ids: {candidate.get('candidate_id')}")
    _assert_model_blind(candidate)


def _validate_frame(frame: dict[str, Any], *, session_id: str, ancestry: str) -> None:
    if frame.get("schema") != "hftf_d7_public_real_frame_v1":
        raise ContractError(f"THOR-MAGNI frame schema drift: {frame.get('frame_id')}")
    if frame.get("dataset_id") != DATASET or frame.get("source_session_id") != session_id or frame.get("ancestry_group") != ancestry:
        raise ContractError(f"THOR-MAGNI frame identity drift: {frame.get('frame_id')}")
    if frame.get("timestamp_ns") is None or not frame.get("source_hash"):
        raise ContractError(f"THOR-MAGNI frame lacks source timestamp/hash: {frame.get('frame_id')}")
    _assert_model_blind(frame)


def _new_event(candidate: dict[str, Any], *, reason: str, root: Path) -> dict[str, Any]:
    event_id = str(candidate.get("parent_event_id") or stable_id("d7parent", candidate["candidate_id"]))
    rgb_uri = str(candidate.get("rgb_uri") or "")
    rgb_path = (root / rgb_uri).resolve() if rgb_uri else None
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
        "candidate_selection_model_visible": False,
        "review_model_output_visible": False,
        "geometry_model_output_visible": False,
        "not_evaluable_reason": reason,
        "required_review_roles": list(REVIEW_ROLES),
        "source_license": candidate.get("source_license"),
        "source_hash": candidate.get("source_hash"),
        "rgb_local_path": str(rgb_path) if rgb_path is not None and rgb_path.is_file() else None,
        "source_metadata": candidate.get("source_metadata"),
    }


def _new_review(candidate: dict[str, Any], *, role: str, event_id: str, reason: str, root: Path) -> dict[str, Any]:
    rgb_uri = str(candidate.get("rgb_uri") or "")
    rgb_path = (root / rgb_uri).resolve() if rgb_uri else None
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
        "source_native_geometry_only": role == "GEOMETRY_EVIDENCE_REVIEWER",
        "counterexample_search_required": role == "COUNTEREXAMPLE_REVIEWER",
        "not_evaluable_reason": reason,
        "rgb_local_path": str(rgb_path) if rgb_path is not None and rgb_path.is_file() else None,
    }


def _new_queue(candidate: dict[str, Any], *, event_id: str, reason: str) -> dict[str, Any]:
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


def _copy_backup(root: Path, backup_root: Path, relative_paths: Iterable[str]) -> None:
    for relative in relative_paths:
        source = root / relative
        if not source.is_file():
            raise ContractError(f"cannot back up missing D7 artifact: {source}")
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _restore_backup(root: Path, backup_root: Path, relative_paths: Iterable[str]) -> None:
    for relative in relative_paths:
        source = backup_root / relative
        destination = root / relative
        if not source.is_file():
            raise ContractError(f"merge backup is incomplete: {source}")
        shutil.copy2(source, destination)


def _check_postconditions(
    root: Path,
    *,
    expected_candidate_ids: set[str],
    expected_frame_ids: set[str],
    expected_session_ids: set[str],
    expected_event_ids: set[str],
    expected_parent_event_ids: set[str],
) -> dict[str, Any]:
    candidate_ids = _id_set(root / "candidates" / "candidate_index.jsonl", "candidate_id")
    frame_ids = _id_set(root / "canonical" / "frame_registry.jsonl", "frame_id")
    session_ids = _id_set(root / "manifests" / "session_registry.jsonl", "source_session_id")
    event_ids = _id_set(root / "manifests" / "event_manifest.jsonl", "event_id")
    parent_event_ids = _id_set(root / "manifests" / "event_manifest.jsonl", "parent_event_id")
    event_candidate_ids = _id_set(root / "manifests" / "event_manifest.jsonl", "candidate_id")
    queue_candidate_ids = _id_set(root / "reviews" / "review_queue.jsonl", "candidate_id")
    rejected_candidate_ids = _id_set(root / "adjudication" / "rejected_events.jsonl", "candidate_id")
    expected = {
        "candidate_ids": (candidate_ids, expected_candidate_ids),
        "frame_ids": (frame_ids, expected_frame_ids),
        "session_ids": (session_ids, expected_session_ids),
        "event_ids": (event_ids, expected_event_ids),
        "parent_event_ids": (parent_event_ids, expected_parent_event_ids),
        "event_candidate_ids": (event_candidate_ids, expected_candidate_ids),
        "queue_candidate_ids": (queue_candidate_ids, expected_candidate_ids),
        "rejected_candidate_ids": (rejected_candidate_ids, expected_candidate_ids),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ContractError(f"THOR-MAGNI postcondition mismatch for {label}: actual={len(actual)} expected={len(wanted)}")
    for relative in REVIEW_FILES.values():
        actual = _id_set(root / relative, "candidate_id")
        if actual != expected_candidate_ids:
            raise ContractError(f"THOR-MAGNI postcondition mismatch for review surface: {relative}")
    return {
        "candidate_count": len(candidate_ids),
        "frame_count": len(frame_ids),
        "session_count": len(session_ids),
        "event_count": len(event_ids),
        "parent_event_count": len(parent_event_ids),
        "review_surface_count": len(REVIEW_FILES),
        "candidate_index_sha256": sha256_file(root / "candidates" / "candidate_index.jsonl"),
        "frame_registry_sha256": sha256_file(root / "canonical" / "frame_registry.jsonl"),
        "session_registry_sha256": sha256_file(root / "manifests" / "session_registry.jsonl"),
        "event_manifest_sha256": sha256_file(root / "manifests" / "event_manifest.jsonl"),
        "review_queue_sha256": sha256_file(root / "reviews" / "review_queue.jsonl"),
        "rejected_events_sha256": sha256_file(root / "adjudication" / "rejected_events.jsonl"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    manifest_paths = [Path(value).resolve() for value in args.window_manifest]
    candidates, frames, sessions, manifest_rows = _load_artifacts(root, manifest_paths)
    excluded_manifest_paths = [Path(value).resolve() for value in getattr(args, "excluded_window_manifest", [])]
    if set(excluded_manifest_paths) & set(manifest_paths):
        raise ContractError("a window manifest cannot be both selected and explicitly excluded")
    excluded_rows: list[dict[str, Any]] = []
    if excluded_manifest_paths:
        _, _, _, excluded_rows = _load_artifacts(root, excluded_manifest_paths)
        selected_identities = {
            (
                row.get("source_session_id"),
                row.get("ancestry_group"),
                row.get("file_id"),
                row.get("rgb_sha256"),
                row.get("scenario_csv_sha256"),
            ): row
            for row in manifest_rows
        }
        for excluded in excluded_rows:
            identity = (
                excluded.get("source_session_id"),
                excluded.get("ancestry_group"),
                excluded.get("file_id"),
                excluded.get("rgb_sha256"),
                excluded.get("scenario_csv_sha256"),
            )
            canonical = selected_identities.get(identity)
            if canonical is None:
                raise ContractError(f"excluded THOR-MAGNI window has no selected canonical replacement: {excluded.get('path')}")
            excluded["exclusion_reason"] = "DUPLICATE_SOURCE_SESSION_CANONICAL_SELECTED"
            excluded["canonical_manifest_path"] = canonical["path"]
    if not candidates or not frames:
        raise ContractError("THOR-MAGNI merge artifacts must not be empty")

    old_candidate_path = root / "candidates" / "candidate_index.jsonl"
    old_frame_path = root / "canonical" / "frame_registry.jsonl"
    old_session_path = root / "manifests" / "session_registry.jsonl"
    required = [old_candidate_path, old_frame_path, old_session_path, root / "manifests" / "dataset_registry.json", root / "manifests" / "event_manifest.jsonl", root / "reviews" / "review_queue.jsonl", *[root / value for value in REVIEW_FILES.values()], root / "adjudication" / "rejected_events.jsonl", root / "adjudication" / "adjudicated_events.jsonl", root / "manifests" / "pending_package_manifest.json"]
    for path in required:
        if not path.is_file():
            raise ContractError(f"required D7 merge artifact missing: {path}")

    old_candidate_ids = _id_set(old_candidate_path, "candidate_id")
    new_candidate_ids = _id_set_from_rows(candidates, "candidate_id")
    if old_candidate_ids & new_candidate_ids:
        raise ContractError(f"THOR-MAGNI candidate collision: {sorted(old_candidate_ids & new_candidate_ids)[:5]}")
    old_frame_ids = _id_set(old_frame_path, "frame_id")
    new_frame_ids = _id_set_from_rows(frames, "frame_id")
    if old_frame_ids & new_frame_ids:
        raise ContractError(f"THOR-MAGNI frame collision: {sorted(old_frame_ids & new_frame_ids)[:5]}")
    old_session_ids = _id_set(old_session_path, "source_session_id")
    new_session_ids = _id_set_from_rows(sessions, "source_session_id")
    if old_session_ids & new_session_ids or len(new_session_ids) != len(sessions):
        raise ContractError("THOR-MAGNI source-session collision")
    new_ancestry = {str(row.get("ancestry_group") or "") for row in sessions}
    old_ancestry = {str(row.get("ancestry_group") or "") for row in _iter_jsonl(old_session_path)}
    if "" in new_ancestry or new_ancestry & old_ancestry or len(new_ancestry) != len(sessions):
        raise ContractError("THOR-MAGNI ancestry-group collision or missing identity")
    event_path = root / "manifests" / "event_manifest.jsonl"
    old_event_ids = _id_set(event_path, "event_id")
    old_parent_event_ids = _id_set(event_path, "parent_event_id")
    old_adjudicated_event_ids = _id_set(root / "adjudication" / "adjudicated_events.jsonl", "event_id")
    old_rejected_event_ids = _id_set(root / "adjudication" / "rejected_events.jsonl", "event_id")
    old_event_ids |= old_adjudicated_event_ids | old_rejected_event_ids
    old_adjudicated_parent_ids = _id_set(root / "adjudication" / "adjudicated_events.jsonl", "parent_event_id")
    old_rejected_parent_ids = {
        str(row.get("parent_event_id"))
        for row in _iter_jsonl(root / "adjudication" / "rejected_events.jsonl")
        if row.get("parent_event_id")
    }
    old_parent_event_ids |= old_adjudicated_parent_ids | old_rejected_parent_ids
    new_parent_event_ids = _id_set_from_rows(candidates, "parent_event_id")
    if old_parent_event_ids & new_parent_event_ids:
        raise ContractError(f"THOR-MAGNI parent-event collision: {sorted(old_parent_event_ids & new_parent_event_ids)[:5]}")
    referenced_frames = {str(frame_id) for candidate in candidates for frame_id in candidate.get("frame_ids", [])}
    if not referenced_frames <= new_frame_ids:
        raise ContractError(f"THOR-MAGNI candidate/frame set mismatch: referenced={len(referenced_frames)} frames={len(new_frame_ids)}")
    if {str(candidate.get("source_session_id")) for candidate in candidates} != new_session_ids:
        raise ContractError("THOR-MAGNI candidate/session set mismatch")
    existing_candidate_sets = {relative: _id_set(root / relative, "candidate_id") for relative in REVIEW_FILES.values()}
    for relative, values in existing_candidate_sets.items():
        if values != old_candidate_ids:
            raise ContractError(f"existing review surface candidate set mismatch: {relative}")
    if _id_set(root / "reviews" / "review_queue.jsonl", "candidate_id") != old_candidate_ids:
        raise ContractError("existing review queue candidate set mismatch")
    if _id_set(root / "adjudication" / "rejected_events.jsonl", "candidate_id") != old_candidate_ids:
        raise ContractError("existing rejected-event candidate set mismatch")

    registry_path = root / "manifests" / "dataset_registry.json"
    registry = load_json(registry_path)
    if not isinstance(registry, dict):
        raise ContractError("dataset registry is not an object")
    discovery = registry.setdefault("candidate_discovery", {})
    if not isinstance(discovery, dict):
        raise ContractError("candidate_discovery is not an object")
    imports = discovery.setdefault("imports", [])
    if not isinstance(imports, list):
        raise ContractError("candidate_discovery.imports is not a list")
    source_stats = registry.setdefault("source_stats", {})
    if not isinstance(source_stats, dict):
        raise ContractError("source_stats is not an object")
    pending_path = root / "manifests" / "pending_package_manifest.json"
    pending = load_json(pending_path)
    if not isinstance(pending, dict):
        raise ContractError("pending package manifest is not an object")

    reason = "THOR_MAGNI_INDEPENDENT_REVIEW_NOT_RUN"
    events = [_new_event(candidate, reason=reason, root=root) for candidate in candidates]
    new_event_ids = _id_set_from_rows(events, "event_id")
    if old_event_ids & new_event_ids:
        raise ContractError(f"THOR-MAGNI event collision: {sorted(old_event_ids & new_event_ids)[:5]}")
    if new_event_ids != new_parent_event_ids:
        raise ContractError("THOR-MAGNI event/parent ID mismatch")
    event_by_candidate = {str(event["candidate_id"]): event for event in events}
    queue = [_new_queue(candidate, event_id=str(event_by_candidate[candidate["candidate_id"]]["event_id"]), reason=reason) for candidate in candidates]
    reviews_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        event_id = str(event_by_candidate[candidate["candidate_id"]]["event_id"])
        for role in REVIEW_ROLES:
            reviews_by_role[role].append(_new_review(candidate, role=role, event_id=event_id, reason=reason, root=root))
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

    if getattr(args, "dry_run", False):
        return {
            "status": "DRY_RUN_PREFLIGHT_OK",
            "dataset_id": DATASET,
            "window_manifest_count": len(manifest_paths),
            "candidate_count": len(candidates),
            "frame_count": len(frames),
            "session_count": len(sessions),
            "excluded_window_manifest_count": len(excluded_rows),
            "excluded_window_manifests": excluded_rows,
            "reused_frame_reference_count": sum(int(row.get("reused_frame_reference_count", 0) or 0) for row in manifest_rows),
            "event_truth_authority": False,
            "review_assignments_are_not_labels": True,
        }

    backup_relative = [
        "candidates/candidate_index.jsonl",
        "canonical/frame_registry.jsonl",
        "manifests/session_registry.jsonl",
        "manifests/dataset_registry.json",
        "manifests/event_manifest.jsonl",
        "reviews/review_queue.jsonl",
        *REVIEW_FILES.values(),
        "adjudication/rejected_events.jsonl",
        "manifests/pending_package_manifest.json",
    ]
    backup_root = root / "manifests" / "merge_backups" / args.run_id
    if backup_root.exists():
        raise ContractError(f"merge backup already exists: {backup_root}")
    receipt_path = root / "receipts" / f"thor_magni_candidate_merge_receipt_{args.run_id}.json"
    if receipt_path.exists():
        raise ContractError(f"merge receipt already exists: {receipt_path}")
    backup_root.mkdir(parents=True, exist_ok=False)
    _copy_backup(root, backup_root, backup_relative)

    expected_candidate_ids = old_candidate_ids | new_candidate_ids
    expected_frame_ids = old_frame_ids | new_frame_ids
    expected_session_ids = old_session_ids | new_session_ids
    expected_event_ids = old_event_ids | new_event_ids
    expected_parent_event_ids = old_parent_event_ids | new_parent_event_ids
    try:
        _atomic_append(old_candidate_path, candidates)
        _atomic_append(old_frame_path, frames)
        _atomic_append(old_session_path, sessions)
        _atomic_append(event_path, events)
        _atomic_append(root / "reviews" / "review_queue.jsonl", queue)
        for role, relative in REVIEW_FILES.items():
            _atomic_append(root / relative, reviews_by_role[role])
        _atomic_append(root / "adjudication" / "rejected_events.jsonl", rejected)

        discovery["total_candidate_count"] = len(expected_candidate_ids)
        discovery["total_frame_count"] = len(expected_frame_ids)
        imports.extend({
            "authority": "DEVELOPMENT_DISCOVERY_ONLY",
            "candidate_count": row["candidate_count"],
            "frame_count": row["frame_count"],
            "model_blind": True,
            "path": row["candidate_path"],
            "candidate_artifact_sha256": row["candidate_sha256"],
            "frame_artifact_path": row["frame_path"],
            "frame_artifact_sha256": row["frame_sha256"],
            "window_receipt_path": row["window_receipt_path"],
            "window_receipt_sha256": row["window_receipt_sha256"],
            "member_receipt_path": row["member_receipt_path"],
            "member_receipt_sha256": row["member_receipt_sha256"],
            "archive_checksum": row["archive_checksum"],
            "source": "THOR-MAGNI/public-selected-members-source-time",
        } for row in manifest_rows)
        registry["session_count"] = len(expected_session_ids)
        stats = source_stats.setdefault(DATASET, {})
        if not isinstance(stats, dict):
            raise ContractError("THOR-MAGNI source_stats is not an object")
        stats["candidate_windows"] = int(stats.get("candidate_windows", 0)) + len(candidates)
        stats["rgb_frames"] = int(stats.get("rgb_frames", 0)) + sum(int(row.get("rgb_count", 0) or 0) for row in sessions)
        stats["pose_frames"] = int(stats.get("pose_frames", 0)) + len(frames)
        stats["media_rows"] = int(stats.get("media_rows", 0)) + len(sessions)
        stats["ledger_rows"] = int(stats.get("ledger_rows", 0)) + len(sessions)
        stats["source_synchronized_frames"] = int(stats.get("source_synchronized_frames", 0)) + len(frames)
        stats["reused_frame_reference_count"] = int(stats.get("reused_frame_reference_count", 0)) + sum(int(row.get("reused_frame_reference_count", 0) or 0) for row in manifest_rows)
        write_json(registry_path, registry)

        dataset_counts: Counter[str] = Counter()
        for row in _iter_jsonl(old_candidate_path):
            dataset_counts[str(row.get("dataset_id") or "UNKNOWN")] += 1
        pending.update({
            "generated_at_utc": utc_now(),
            "status": "NOT_COMPLETE_PENDING_INDEPENDENT_REVIEW",
            "candidate_count": sum(dataset_counts.values()),
            "admitted_parent_event_count": 0,
            "source_session_count": len(expected_session_ids),
            "source_coverage": {dataset: {"candidates": count, "not_evaluable": count, "admitted": 0} for dataset, count in sorted(dataset_counts.items())},
            "candidate_index_sha256": sha256_file(old_candidate_path),
            "session_registry_sha256": sha256_file(old_session_path),
            "review_assignments_are_not_labels": True,
            "training_authorized": False,
            "confirmation_authorized": False,
            "production_authorized": False,
        })
        write_json(pending_path, pending)

        postconditions = _check_postconditions(
            root,
            expected_candidate_ids=expected_candidate_ids,
            expected_frame_ids=expected_frame_ids,
            expected_session_ids=expected_session_ids,
            expected_event_ids=expected_event_ids,
            expected_parent_event_ids=expected_parent_event_ids,
        )
        receipt = {
            "schema": "hftf_d7_public_real_thor_magni_candidate_merge_receipt_v1",
            "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
            "run_id": args.run_id,
            "generated_at_utc": utc_now(),
            "dataset_id": DATASET,
            "status": "MERGED_ASSIGNMENT_ONLY_NO_EVENT_TRUTH",
            "window_manifests": manifest_rows,
            "excluded_window_manifests": excluded_rows,
            "new_candidate_count": len(candidates),
            "new_frame_count": len(frames),
            "new_session_count": len(sessions),
            "top_level_candidate_count": sum(dataset_counts.values()),
            "top_level_frame_count": postconditions["frame_count"],
            "top_level_session_count": postconditions["session_count"],
            "top_level_event_count": postconditions["event_count"],
            "admitted_parent_events": 0,
            "selection_authority": "MODEL_BLIND_UNIFORM_QTM_100HZ_SOURCE_COVERAGE",
            "event_truth_authority": False,
            "review_assignments_are_not_labels": True,
            "backup_root": str(backup_root),
            "reused_frame_reference_count": sum(int(row.get("reused_frame_reference_count", 0) or 0) for row in manifest_rows),
            "postconditions": postconditions,
            "candidate_index_sha256": postconditions["candidate_index_sha256"],
            "frame_registry_sha256": postconditions["frame_registry_sha256"],
            "session_registry_sha256": postconditions["session_registry_sha256"],
            "training_authorized": False,
            "confirmation_authorized": False,
            "production_authorized": False,
        }
        write_json(receipt_path, receipt)
    except Exception:
        if receipt_path.exists():
            receipt_path.unlink(missing_ok=True)
        _restore_backup(root, backup_root, backup_relative)
        raise
    return receipt


def _id_set_from_rows(rows: Iterable[dict[str, Any]], field: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        value = str(row.get(field) or "")
        if not value or value in values:
            raise ContractError(f"duplicate or missing {field} in intake rows: {value}")
        values.add(value)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--window-manifest", action="append", default=[])
    parser.add_argument("--excluded-window-manifest", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
