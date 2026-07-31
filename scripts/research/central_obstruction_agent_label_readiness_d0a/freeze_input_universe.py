#!/usr/bin/env python3
"""Write-once D0-A0 continuous-RGB input-universe freeze."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = MODULE_DIR / "source_universe_r3.json"
MANIFEST_NAME = "input-universe-manifest.json"
RECEIPT_NAME = "input-universe-receipt.json"
ROLE_LEDGER_NAME = "reuse-role-ledger.jsonl"
FITNESS_REVIEW_NAME = "reuse-fitness-review.json"
MANIFEST_SCHEMA = "blindassist.central_obstruction_d0a0_input_universe.v1"
RECEIPT_SCHEMA = "blindassist.central_obstruction_d0a0_input_universe_receipt.v1"


class FreezeError(ValueError):
    """Fail-closed D0-A0 contract error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def load_object(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeError(f"{where}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise FreezeError(f"{where}: expected a JSON object")
    return value


def load_source_spec(repo_root: Path, spec_path: Path) -> tuple[dict[str, Any], dict[str, str] | None]:
    return _load_source_spec(repo_root, spec_path, visited=set())


def _load_source_spec(
    repo_root: Path,
    spec_path: Path,
    *,
    visited: set[Path],
) -> tuple[dict[str, Any], dict[str, str] | None]:
    spec_path = spec_path.resolve()
    if spec_path in visited:
        raise FreezeError("source spec inheritance contains a cycle")
    visited.add(spec_path)
    raw = load_object(spec_path, where="source spec")
    inherited = raw.get("inherits_path")
    if inherited is None:
        return raw, None
    base_path = repo_file(repo_root, inherited, where="source spec predecessor")
    base, _ = _load_source_spec(repo_root, base_path, visited=visited)
    effective = json.loads(json.dumps(base))
    for key in (
        "schema_version",
        "protocol_id",
        "phase",
        "evidence_instance",
        "output_root",
        "candidate_output_access",
        "eligibility_rule",
        "fitness_review",
        "reuse_discovery",
    ):
        if key in raw:
            effective[key] = raw[key]
    sessions = effective.get("sessions")
    if not isinstance(sessions, list):
        raise FreezeError("source spec predecessor sessions are invalid")
    by_id = {row.get("session_id"): row for row in sessions if isinstance(row, dict)}
    for override in raw.get("session_role_overrides", []):
        if not isinstance(override, dict) or override.get("session_id") not in by_id:
            raise FreezeError("source spec session_role_overrides contains an unknown session")
        by_id[override["session_id"]].update(override)
    additions = raw.get("additional_sessions", [])
    if not isinstance(additions, list) or any(not isinstance(row, dict) for row in additions):
        raise FreezeError("source spec additional_sessions are invalid")
    sessions.extend(additions)
    effective["sessions"] = sessions
    return effective, {
        "path": relative_posix(base_path, repo_root),
        "sha256": sha256_file(base_path),
    }


def repo_file(repo_root: Path, relative: str, *, where: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise FreezeError(f"{where}: expected a non-empty relative path")
    declared = Path(relative)
    if declared.is_absolute() or ".." in declared.parts:
        raise FreezeError(f"{where}: path escapes repository")
    candidate = repo_root.absolute() / declared
    if not candidate.is_file():
        raise FreezeError(f"{where}: file is missing: {relative}")
    return candidate


def repo_dir(repo_root: Path, relative: str, *, where: str) -> Path:
    declared = Path(relative)
    if declared.is_absolute() or ".." in declared.parts:
        raise FreezeError(f"{where}: path escapes repository")
    candidate = repo_root.absolute() / declared
    if not candidate.is_dir():
        raise FreezeError(f"{where}: directory is missing: {relative}")
    return candidate


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as error:
        raise FreezeError("declared path escapes repository") from error


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise FreezeError(f"{path}: blank JSONL row at line {line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise FreezeError(f"{path}: invalid JSON at line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise FreezeError(f"{path}: row {line_number} is not an object")
            yield line_number, row


def normalize_row(
    *,
    row: dict[str, Any],
    session: dict[str, Any],
    ledger_path: Path,
    payload_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], Path]:
    adapter = session["ledger_adapter"]
    if adapter == "CROWDBOT_FRAMES_V1":
        source_frame_id = row.get("frame_id")
        sequence_id = row.get("sequence_id")
        if sequence_id != session["session_id"]:
            raise FreezeError(f"{session['session_id']}: sequence identity mismatch")
        image_relative = row.get("rgb_path")
        expected_image_sha = row.get("rgb_sha256")
        width = session["expected_width"]
        height = session["expected_height"]
    elif adapter == "PUBLIC_VIDEO_REPLAY_RGB_V1":
        source_frame_id = row.get("frame_id")
        image_relative = row.get("image_path")
        expected_image_sha = row.get("image_sha256")
        width = row.get("width")
        height = row.get("height")
    else:
        raise FreezeError(f"{session['session_id']}: unsupported ledger adapter {adapter}")

    try:
        frame_index = int(source_frame_id)
    except (TypeError, ValueError) as error:
        raise FreezeError(f"{session['session_id']}: invalid frame_id {source_frame_id!r}") from error
    timestamp = row.get("source_capture_timestamp_ns")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
        raise FreezeError(f"{session['session_id']} frame {frame_index}: invalid timestamp")
    if row.get("source_id") != session["source_id"]:
        raise FreezeError(f"{session['session_id']} frame {frame_index}: source identity mismatch")
    if width != session["expected_width"] or height != session["expected_height"]:
        raise FreezeError(f"{session['session_id']} frame {frame_index}: dimension mismatch")
    if not isinstance(image_relative, str) or not image_relative:
        raise FreezeError(f"{session['session_id']} frame {frame_index}: image path is missing")
    if Path(image_relative).is_absolute() or ".." in Path(image_relative).parts:
        raise FreezeError(f"{session['session_id']} frame {frame_index}: payload path escapes root")
    image_path = ledger_path.parent / image_relative
    try:
        image_path.resolve().relative_to(payload_root.resolve())
    except ValueError as error:
        raise FreezeError(f"{session['session_id']} frame {frame_index}: payload path escapes root") from error
    if not image_path.is_file():
        raise FreezeError(f"{session['session_id']} frame {frame_index}: payload is missing")
    if not isinstance(expected_image_sha, str) or len(expected_image_sha) != 64:
        raise FreezeError(f"{session['session_id']} frame {frame_index}: invalid ledger SHA-256")
    actual_image_sha = sha256_file(image_path)
    if actual_image_sha != expected_image_sha:
        raise FreezeError(f"{session['session_id']} frame {frame_index}: payload SHA-256 mismatch")

    normalized = {
        "frame_index": frame_index,
        "source_frame_id": str(source_frame_id),
        "source_capture_timestamp_ns": timestamp,
        "timestamp_semantics": session["timestamp_semantics"],
        "image_path": relative_posix(image_path, repo_root),
        "image_sha256": actual_image_sha,
        "width": width,
        "height": height,
    }
    return normalized, image_path


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception as error:  # Pillow provides format-specific diagnostics.
        raise FreezeError(f"cannot read image dimensions: {path}: {error}") from error


def role_fields(session: dict[str, Any], *, content_identity: str) -> dict[str, Any]:
    required = (
        "dataset_name",
        "independence_group",
        "ancestry",
        "current_task_fitness",
        "missing_current_task_requirements",
        "prior_content_access",
        "prior_algorithm_output_access",
        "claim_relevant_outcome_overlap",
        "selection_or_tuning_influence",
        "assigned_current_role",
        "admission_disposition",
        "exclusion_reason",
        "reuse_candidates",
    )
    missing = [key for key in required if key not in session]
    if missing:
        raise FreezeError(f"{session.get('session_id')}: missing reuse-first fields {missing}")
    return {
        "dataset_name": session["dataset_name"],
        "content_identity": content_identity,
        "independence_group": session["independence_group"],
        "ancestry": session["ancestry"],
        "current_task_fitness": session["current_task_fitness"],
        "missing_current_task_requirements": session["missing_current_task_requirements"],
        "prior_content_access": session["prior_content_access"],
        "prior_algorithm_output_access": session["prior_algorithm_output_access"],
        "claim_relevant_outcome_overlap": session["claim_relevant_outcome_overlap"],
        "selection_or_tuning_influence": session["selection_or_tuning_influence"],
        "assigned_current_role": session["assigned_current_role"],
        "admission_disposition": session["admission_disposition"],
        "exclusion_reason": session["exclusion_reason"],
        "reuse_candidates": session["reuse_candidates"],
    }


def audit_asset_row(
    *,
    source_id: str,
    session_id: str,
    dataset_name: str,
    content_identity: str,
    independence_group: str,
    ancestry: list[str],
    current_task_fitness: str,
    missing_current_task_requirements: list[str],
    prior_content_access: Any,
    prior_algorithm_output_access: Any,
    claim_relevant_outcome_overlap: str,
    selection_or_tuning_influence: str,
    assigned_current_role: str,
    admission_disposition: str,
    exclusion_reason: str | None,
    reuse_candidates: list[str],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "session_id": session_id,
        "dataset_name": dataset_name,
        "content_identity": content_identity,
        "independence_group": independence_group,
        "ancestry": ancestry,
        "current_task_fitness": current_task_fitness,
        "missing_current_task_requirements": missing_current_task_requirements,
        "prior_content_access": prior_content_access,
        "prior_algorithm_output_access": prior_algorithm_output_access,
        "claim_relevant_outcome_overlap": claim_relevant_outcome_overlap,
        "selection_or_tuning_influence": selection_or_tuning_influence,
        "assigned_current_role": assigned_current_role,
        "admission_disposition": admission_disposition,
        "exclusion_reason": exclusion_reason,
        "reuse_candidates": reuse_candidates,
    }


def discover_reuse_assets(
    *,
    repo_root: Path,
    spec: dict[str, Any],
    admitted_payload_shas: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    discovery = spec.get("reuse_discovery")
    if not isinstance(discovery, dict):
        return [], set()
    opened: set[str] = set()
    rows: list[dict[str, Any]] = []
    extensions = {str(value).lower() for value in discovery.get("video_extensions", [])}
    videos_by_sha: dict[str, list[Path]] = {}
    for root_text in discovery.get("video_roots", []):
        root = repo_dir(repo_root, root_text, where="reuse_discovery.video_root")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                digest = sha256_file(path)
                opened.add(relative_posix(path, repo_root))
                videos_by_sha.setdefault(digest, []).append(path)
    for digest, paths in sorted(videos_by_sha.items()):
        if digest in admitted_payload_shas:
            continue
        relative_paths = sorted(relative_posix(path, repo_root) for path in paths)
        first_parts = Path(relative_paths[0]).parts
        dataset_name = first_parts[2] if len(first_parts) > 2 else "local-video-asset"
        total_bytes = sum(path.stat().st_size for path in paths)
        rows.append(
            audit_asset_row(
                source_id=f"local_video_asset_{digest[:16]}",
                session_id=f"video_content_{digest[:16]}",
                dataset_name=dataset_name,
                content_identity=f"sha256:{digest};alias_count:{len(paths)};total_alias_bytes:{total_bytes}",
                independence_group=f"UNRESOLVED_VIDEO_ANCESTRY::{dataset_name}",
                ancestry=relative_paths,
                current_task_fitness=(
                    "LOCAL_CONTINUOUS_VIDEO_PAYLOAD_PRESENT; no complete per-frame D0-A timestamp/hash ledger "
                    "or bounded central-ROI fitness audit exists in the current evidence version."
                ),
                missing_current_task_requirements=[
                    "complete per-frame order/timestamp/payload-hash ledger",
                    "session-level central-image fitness canary",
                    "resolved ancestry/duplicate relationship where aliases or re-encodes exist",
                ],
                prior_content_access="YES_OR_HISTORICAL_UNKNOWN",
                prior_algorithm_output_access="UNKNOWN_FAIL_CLOSED",
                claim_relevant_outcome_overlap="UNKNOWN_FAIL_CLOSED_TO_CALIBRATION_ONLY",
                selection_or_tuning_influence="HISTORICAL_EVIDENCE_ASSET; EXACT_INFLUENCE_NOT_RECONSTRUCTED_AT_D0-A0",
                assigned_current_role="D0_A_CALIBRATION_CANDIDATE",
                admission_disposition="ADMIT_D0_A_CALIBRATION_ONLY",
                exclusion_reason=None,
                reuse_candidates=["D0-A1 excluded calibration", "visual stress case", "source characterization"],
            )
        )

    manifest_pattern = discovery.get("device_manifest_glob")
    if isinstance(manifest_pattern, str):
        for path in sorted(repo_root.glob(manifest_pattern)):
            manifest = load_object(path, where="reuse device manifest")
            opened.add(relative_posix(path, repo_root))
            frames = manifest.get("frames")
            if not isinstance(frames, list):
                raise FreezeError(f"reuse device manifest lacks frames: {path}")
            present = 0
            for frame in frames:
                if isinstance(frame, dict) and isinstance(frame.get("image_path"), str):
                    if (path.parent / frame["image_path"]).is_file():
                        present += 1
            payload_present = present == len(frames) and len(frames) > 0
            disposition = (
                "ADMIT_D0_A_CALIBRATION_ONLY" if payload_present else "NOT_EVALUABLE_FOR_CURRENT_QUESTION"
            )
            rows.append(
                audit_asset_row(
                    source_id=str(manifest.get("source_id")),
                    session_id=str(manifest.get("sequence_id")),
                    dataset_name="CrowdBot / USTRF canonical materialization",
                    content_identity=f"manifest_sha256:{sha256_file(path)};frames:{len(frames)};rgb_payload_present:{present}",
                    independence_group=f"{manifest.get('source_id')}::{manifest.get('sequence_id')}",
                    ancestry=["CrowdBot/Qolo", relative_posix(path, repo_root)],
                    current_task_fitness=(
                        "Complete device manifest and RGB payload are present."
                        if payload_present
                        else "Frame identities remain, but the current host evidence does not retain renderable RGB payload."
                    ),
                    missing_current_task_requirements=(
                        ["full-session central-image fitness canary"]
                        if payload_present
                        else ["renderable continuous RGB payload"]
                    ),
                    prior_content_access=True,
                    prior_algorithm_output_access=True,
                    claim_relevant_outcome_overlap="UNRELATED_ROUTE_TARGET_DETECTOR_MATERIALIZATION",
                    selection_or_tuning_influence="USED_IN_HISTORICAL_USTRF_DEVELOPMENT",
                    assigned_current_role=(
                        "D0_A_CALIBRATION_CANDIDATE" if payload_present else "SOURCE_CHARACTERIZATION_ONLY"
                    ),
                    admission_disposition=disposition,
                    exclusion_reason=(
                        None
                        if payload_present
                        else "Current RGB payload is absent; historical dataset use is not the exclusion reason."
                    ),
                    reuse_candidates=["source/session identity", "transport regression", "future RGB recovery audit"],
                )
            )

    for asset in discovery.get("additional_assets", []):
        if not isinstance(asset, dict):
            raise FreezeError("reuse_discovery.additional_assets row is invalid")
        asset_paths = []
        content_digest = hashlib.sha256()
        file_count = 0
        image_count = 0
        total_bytes = 0
        for path_text in asset.get("paths", []):
            declared = repo_root / path_text
            if declared.is_file():
                files = [declared]
            elif declared.is_dir():
                files = sorted(path for path in declared.rglob("*") if path.is_file())
            else:
                raise FreezeError(f"additional reuse asset path is missing: {path_text}")
            asset_paths.append(path_text)
            for path in files:
                rel = relative_posix(path, repo_root)
                digest = sha256_file(path)
                opened.add(rel)
                content_digest.update(canonical_bytes({"path": rel, "sha256": digest}))
                file_count += 1
                total_bytes += path.stat().st_size
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    image_count += 1
        rows.append(
            audit_asset_row(
                source_id=asset["source_id"],
                session_id=asset["session_id"],
                dataset_name=asset["dataset_name"],
                content_identity=(
                    f"ordered_asset_sha256:{content_digest.hexdigest()};files:{file_count};"
                    f"images:{image_count};bytes:{total_bytes}"
                ),
                independence_group=asset["independence_group"],
                ancestry=asset_paths,
                current_task_fitness=asset["current_task_fitness"],
                missing_current_task_requirements=asset["missing_current_task_requirements"],
                prior_content_access=asset["prior_content_access"],
                prior_algorithm_output_access=asset["prior_algorithm_output_access"],
                claim_relevant_outcome_overlap=asset["claim_relevant_outcome_overlap"],
                selection_or_tuning_influence=asset["selection_or_tuning_influence"],
                assigned_current_role=asset["assigned_current_role"],
                admission_disposition=asset["admission_disposition"],
                exclusion_reason=asset["exclusion_reason"],
                reuse_candidates=asset["reuse_candidates"],
            )
        )
    return rows, opened


def validate_control_files(
    *,
    repo_root: Path,
    spec: dict[str, Any],
) -> tuple[Path, Path]:
    if spec.get("protocol_id") != "CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A":
        raise FreezeError("source spec protocol_id mismatch")
    if spec.get("phase") != "D0-A0" or spec.get("candidate_output_access") is not False:
        raise FreezeError("source spec does not preserve the D0-A0 candidate-output firewall")
    protocol_path = repo_file(repo_root, spec["protocol_path"], where="protocol_path")
    protocol = load_object(protocol_path, where="protocol")
    if (
        protocol.get("protocol_id") != spec["protocol_id"]
        or protocol.get("execution_authorized") is not True
        or protocol.get("status") != "AUTHORIZED_NOT_RUN"
    ):
        raise FreezeError("protocol does not authorize the unconsumed D0-A execution")
    phase_rows = protocol.get("phase_plan")
    if not isinstance(phase_rows, list) or not any(
        row.get("phase") == "D0-A0" and row.get("candidate_output_access") is False
        for row in phase_rows
        if isinstance(row, dict)
    ):
        raise FreezeError("protocol does not define a candidate-blind D0-A0 phase")
    workflow_path = repo_file(repo_root, spec["workflow_path"], where="workflow_path")
    workflow = load_object(workflow_path, where="workflow")
    policy = workflow.get("workflows", {}).get("central_obstruction_agent_label_readiness_d0a_v1")
    if not isinstance(policy, dict) or policy.get("candidate_output_hidden_from_reviewers") is not True:
        raise FreezeError("AI workflow does not preserve the candidate-output firewall")
    return protocol_path, workflow_path


def build_freeze(
    *,
    repo_root: Path,
    spec_path: Path,
    frozen_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    repo_root = repo_root.resolve()
    spec_path = spec_path.resolve()
    spec, predecessor_spec = load_source_spec(repo_root, spec_path)
    protocol_path, workflow_path = validate_control_files(repo_root=repo_root, spec=spec)
    sessions = spec.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise FreezeError("source spec must enumerate at least one session")
    session_keys: set[tuple[str, str]] = set()
    all_frames: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    opened_inputs = {
        relative_posix(spec_path, repo_root),
        relative_posix(protocol_path, repo_root),
        relative_posix(workflow_path, repo_root),
    }
    total_payload_bytes = 0
    global_identity_digest = hashlib.sha256()
    admitted_role_rows: list[dict[str, Any]] = []
    fitness_input_sessions: list[dict[str, Any]] = []
    admitted_payload_shas: set[str] = set()
    if predecessor_spec is not None:
        opened_inputs.add(predecessor_spec["path"])

    for session in sessions:
        if not isinstance(session, dict):
            raise FreezeError("source spec session must be an object")
        source_id = session.get("source_id")
        session_id = session.get("session_id")
        if not isinstance(source_id, str) or not isinstance(session_id, str):
            raise FreezeError("session source_id/session_id must be strings")
        key = (source_id, session_id)
        if key in session_keys:
            raise FreezeError(f"duplicate source/session: {key}")
        session_keys.add(key)
        if session.get("prior_content_access") is not True:
            raise FreezeError(f"{session_id}: prior content access must be disclosed")

        ledger_path = repo_file(repo_root, session["ledger_path"], where=f"{session_id}.ledger")
        payload_root = repo_dir(repo_root, session["payload_root"], where=f"{session_id}.payload_root")
        ledger_sha = sha256_file(ledger_path)
        if ledger_sha != session.get("ledger_sha256"):
            raise FreezeError(f"{session_id}: source ledger SHA-256 mismatch")
        opened_inputs.add(relative_posix(ledger_path, repo_root))
        ancestry = []
        for index, relative in enumerate(session.get("ancestry_receipt_paths", [])):
            receipt_path = repo_file(repo_root, relative, where=f"{session_id}.ancestry[{index}]")
            load_object(receipt_path, where=f"{session_id}.ancestry[{index}]")
            opened_inputs.add(relative_posix(receipt_path, repo_root))
            ancestry.append(
                {
                    "path": relative_posix(receipt_path, repo_root),
                    "sha256": sha256_file(receipt_path),
                }
            )
        if not ancestry:
            raise FreezeError(f"{session_id}: ancestry receipts are required")
        source_payloads = []
        for index, binding in enumerate(session.get("source_payload_paths", [])):
            if not isinstance(binding, dict):
                raise FreezeError(f"{session_id}.source_payload_paths[{index}] is invalid")
            payload_path = repo_file(repo_root, binding["path"], where=f"{session_id}.source_payload[{index}]")
            payload_sha = sha256_file(payload_path)
            if payload_sha != binding.get("sha256"):
                raise FreezeError(f"{session_id}: source payload SHA-256 mismatch")
            opened_inputs.add(relative_posix(payload_path, repo_root))
            admitted_payload_shas.add(payload_sha)
            source_payloads.append(
                {
                    "path": relative_posix(payload_path, repo_root),
                    "sha256": payload_sha,
                    "bytes": payload_path.stat().st_size,
                }
            )

        previous_timestamp: int | None = None
        session_digest = hashlib.sha256()
        session_frames: list[dict[str, Any]] = []
        if session["ledger_adapter"] == "IMAGE_DIRECTORY_TIMESTAMP_STEM_V1":
            image_paths = sorted(payload_root.glob(session.get("payload_glob", "*.jpg")))
            raw_rows = []
            for index, image_path in enumerate(image_paths):
                try:
                    timestamp = int(image_path.stem)
                except ValueError as error:
                    raise FreezeError(f"{session_id}: image filename is not a timestamp: {image_path.name}") from error
                width, height = image_dimensions(image_path)
                dimension = f"{width}x{height}"
                if dimension not in set(session.get("allowed_dimensions", [])):
                    raise FreezeError(f"{session_id}: unexpected native dimension {dimension}")
                raw_rows.append(
                    (
                        index + 1,
                        {
                            "frame_id": index,
                            "source_id": session["source_id"],
                            "source_capture_timestamp_ns": timestamp,
                            "image_path": image_path.name,
                            "image_sha256": sha256_file(image_path),
                            "width": width,
                            "height": height,
                        },
                    )
                )
        else:
            raw_rows = iter_jsonl(ledger_path)
        for line_number, row in raw_rows:
            if session["ledger_adapter"] == "IMAGE_DIRECTORY_TIMESTAMP_STEM_V1":
                image_path = payload_root / row["image_path"]
                normalized = {
                    "frame_index": row["frame_id"],
                    "source_frame_id": str(row["source_capture_timestamp_ns"]),
                    "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
                    "timestamp_semantics": session["timestamp_semantics"],
                    "image_path": relative_posix(image_path, repo_root),
                    "image_sha256": row["image_sha256"],
                    "width": row["width"],
                    "height": row["height"],
                }
            else:
                normalized, image_path = normalize_row(
                    row=row,
                    session=session,
                    ledger_path=ledger_path,
                    payload_root=payload_root,
                    repo_root=repo_root,
                )
            expected_index = len(session_frames)
            if normalized["frame_index"] != expected_index:
                raise FreezeError(
                    f"{session_id}: frame index {normalized['frame_index']} at line {line_number}; expected {expected_index}"
                )
            timestamp = normalized["source_capture_timestamp_ns"]
            if previous_timestamp is not None:
                delta = timestamp - previous_timestamp
                if delta <= 0:
                    raise FreezeError(f"{session_id} frame {expected_index}: timestamp is not strictly increasing")
                expected_step = session.get("expected_frame_step_ns")
                if expected_step is not None and delta != expected_step:
                    raise FreezeError(f"{session_id} frame {expected_index}: fixed timestamp step mismatch")
            previous_timestamp = timestamp
            frame = {
                "source_id": source_id,
                "session_id": session_id,
                "source_ancestry_group": session["source_ancestry_group"],
                **normalized,
            }
            encoded = canonical_bytes(frame)
            session_digest.update(encoded)
            global_identity_digest.update(encoded)
            total_payload_bytes += image_path.stat().st_size
            session_frames.append(frame)
            all_frames.append(frame)

        expected_count = session.get("expected_frame_count")
        if len(session_frames) != expected_count:
            raise FreezeError(f"{session_id}: frame count {len(session_frames)} != {expected_count}")
        content_identity = f"ordered_frame_identity_sha256:{session_digest.hexdigest()};frames:{len(session_frames)}"
        current_role = role_fields(session, content_identity=content_identity)
        summary = {
                "source_id": source_id,
                "session_id": session_id,
                "source_ancestry_group": session["source_ancestry_group"],
                "source_kind": session["source_kind"],
                "source_uri": session["source_uri"],
                "license_or_rights_metadata": session["license_or_rights_metadata"],
                "ledger_path": relative_posix(ledger_path, repo_root),
                "ledger_sha256": ledger_sha,
                "payload_root": relative_posix(payload_root, repo_root),
                "frame_count": len(session_frames),
                "first_frame_index": session_frames[0]["frame_index"],
                "last_frame_index": session_frames[-1]["frame_index"],
                "first_timestamp_ns": session_frames[0]["source_capture_timestamp_ns"],
                "last_timestamp_ns": session_frames[-1]["source_capture_timestamp_ns"],
                "timestamp_semantics": session["timestamp_semantics"],
                "expected_frame_step_ns": session.get("expected_frame_step_ns"),
                "dimensions": (
                    f"{session['expected_width']}x{session['expected_height']}"
                    if session.get("expected_width") is not None
                    else sorted({f"{row['width']}x{row['height']}" for row in session_frames})
                ),
                "materialization": session["materialization"],
                "prior_access_state": session["prior_access_state"],
                "prior_content_access": session["prior_content_access"],
                "prior_candidate_output_access": session["prior_candidate_output_access"],
                "ordered_frame_identity_sha256": session_digest.hexdigest(),
                "ancestry_receipts": ancestry,
                "source_payloads": source_payloads,
                **current_role,
            }
        session_summaries.append(summary)
        admitted_role_rows.append(
            {
                "source_id": source_id,
                "session_id": session_id,
                **current_role,
                "source_frame_manifest": {
                    "frame_count": len(session_frames),
                    "ordered_frame_identity_sha256": session_digest.hexdigest(),
                    "ledger_path": relative_posix(ledger_path, repo_root),
                    "ledger_sha256": ledger_sha,
                },
            }
        )
        sample_indices = sorted({round(index * (len(session_frames) - 1) / 7) for index in range(8)})
        fitness_input_sessions.append(
            {
                "source_id": source_id,
                "session_id": session_id,
                "sample_rule": "8 evenly spaced frames including both endpoints",
                "samples": [
                    {
                        "frame_index": session_frames[index]["frame_index"],
                        "source_capture_timestamp_ns": session_frames[index]["source_capture_timestamp_ns"],
                        "image_path": session_frames[index]["image_path"],
                        "image_sha256": session_frames[index]["image_sha256"],
                    }
                    for index in sample_indices
                ],
            }
        )

    source_groups = sorted({row["source_ancestry_group"] for row in session_summaries})
    discovered_rows, discovered_inputs = discover_reuse_assets(
        repo_root=repo_root,
        spec=spec,
        admitted_payload_shas=admitted_payload_shas,
    )
    opened_inputs.update(discovered_inputs)
    role_rows = admitted_role_rows + discovered_rows
    role_rows.sort(key=lambda row: (row["admission_disposition"], row["dataset_name"], row["source_id"], row["session_id"]))
    role_keys = [(row["source_id"], row["session_id"]) for row in role_rows]
    if len(set(role_keys)) != len(role_keys):
        raise FreezeError("reuse-role ledger contains duplicate source/session identities")
    prompt_path = repo_file(repo_root, spec["fitness_review"]["prompt_path"], where="fitness review prompt")
    opened_inputs.add(relative_posix(prompt_path, repo_root))
    fitness_input = {
        "schema_version": "blindassist.central_obstruction_d0a0_fitness_input.v1",
        "protocol_id": spec["protocol_id"],
        "phase": "D0-A0",
        "candidate_output_visible": False,
        "truth_or_review_label_visible": False,
        "sessions": fitness_input_sessions,
    }
    fitness_input_sha = hashlib.sha256(canonical_bytes(fitness_input)).hexdigest()
    fitness_review = {
        "schema_version": "blindassist.central_obstruction_d0a0_fitness_review.v1",
        "protocol_id": spec["protocol_id"],
        "phase": "D0-A0",
        "evidence_instance": spec["evidence_instance"],
        **spec["fitness_review"],
        "prompt_sha256": sha256_file(prompt_path),
        "input_sha256": fitness_input_sha,
        "other_review_visible_before_submission": False,
        "labels_generated": False,
        "session_reviews": [
            {
                "source_id": session["source_id"],
                "session_id": session["session_id"],
                "sample_count": len(fitness["samples"]),
                "current_task_fitness": session["current_task_fitness"],
                "admission_disposition": session["admission_disposition"],
            }
            for session, fitness in zip(spec["sessions"], fitness_input_sessions)
        ],
        "input": fitness_input,
        "claim_ceiling": "BOUNDED_MODEL_REVIEWED_SOURCE_FITNESS_CANARY_ONLY",
    }
    disposition_counts: dict[str, int] = {}
    for row in role_rows:
        disposition = row["admission_disposition"]
        disposition_counts[disposition] = disposition_counts.get(disposition, 0) + 1
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_id": spec["protocol_id"],
        "phase": "D0-A0",
        "evidence_instance": spec["evidence_instance"],
        "status": "INPUT_UNIVERSE_FROZEN",
        "frozen_at_utc": frozen_at_utc,
        "eligibility_rule": spec["eligibility_rule"],
        "source_spec": {
            "path": relative_posix(spec_path, repo_root),
            "sha256": sha256_file(spec_path),
        },
        "predecessor_source_spec": predecessor_spec,
        "protocol": {
            "path": relative_posix(protocol_path, repo_root),
            "sha256": sha256_file(protocol_path),
        },
        "workflow": {
            "path": relative_posix(workflow_path, repo_root),
            "sha256": sha256_file(workflow_path),
        },
        "candidate_output_access": False,
        "labels_generated": False,
        "analysis_unit_future_only": "PARENT_NATURAL_EVENT; no event labels exist at D0-A0",
        "source_ancestry_group_count": len(source_groups),
        "source_ancestry_groups": source_groups,
        "session_count": len(session_summaries),
        "frame_count": len(all_frames),
        "ordered_frame_identity_sha256": global_identity_digest.hexdigest(),
        "sessions": session_summaries,
        "frames": all_frames,
        "reuse_role_summary": {
            "row_count": len(role_rows),
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "required_fields": [
                "source_id",
                "session_id",
                "dataset_name",
                "content_identity",
                "independence_group",
                "ancestry",
                "current_task_fitness",
                "missing_current_task_requirements",
                "prior_content_access",
                "prior_algorithm_output_access",
                "claim_relevant_outcome_overlap",
                "selection_or_tuning_influence",
                "assigned_current_role",
                "admission_disposition",
                "exclusion_reason",
                "reuse_candidates",
            ],
        },
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_id": spec["protocol_id"],
        "phase": "D0-A0",
        "evidence_instance": spec["evidence_instance"],
        "status": "D0_A0_INPUT_UNIVERSE_FROZEN",
        "execution_validity": "VALID",
        "scientific_outcome": "NOT_RUN_NO_LABELS",
        "created_at_utc": frozen_at_utc,
        "candidate_output_access": False,
        "truth_or_review_label_access": False,
        "labels_generated": False,
        "source_ancestry_group_count": len(source_groups),
        "session_count": len(session_summaries),
        "frame_count": len(all_frames),
        "payload_total_bytes": total_payload_bytes,
        "ordered_frame_identity_sha256": global_identity_digest.hexdigest(),
        "opened_input_paths": sorted(opened_inputs),
        "output_manifest": {
            "path": MANIFEST_NAME,
            "sha256": None,
        },
        "reuse_role_ledger": {
            "path": ROLE_LEDGER_NAME,
            "sha256": None,
            "row_count": len(role_rows),
            "disposition_counts": dict(sorted(disposition_counts.items())),
        },
        "fitness_review": {
            "path": FITNESS_REVIEW_NAME,
            "sha256": None,
            "input_sha256": fitness_input_sha,
        },
        "next_permitted_action": "D0-A1_EXCLUDED_CALIBRATION_AND_LOCK_ONLY",
        "d0a2_production_labeling_started": False,
        "d0b_model_execution_authorized": False,
        "claim_ceiling": "IMMUTABLE_CONTINUOUS_RGB_INPUT_UNIVERSE_IDENTITY_ONLY",
    }
    return manifest, receipt, role_rows, fitness_review


def write_freeze(
    *,
    repo_root: Path,
    spec_path: Path,
    output_root: Path,
    frozen_at_utc: str,
) -> tuple[Path, Path]:
    manifest_path = output_root / MANIFEST_NAME
    receipt_path = output_root / RECEIPT_NAME
    role_path = output_root / ROLE_LEDGER_NAME
    fitness_path = output_root / FITNESS_REVIEW_NAME
    if any(path.exists() for path in (manifest_path, receipt_path, role_path, fitness_path)):
        raise FreezeError("formal D0-A0 outputs already exist; write-once freeze refuses overwrite")
    manifest, receipt, role_rows, fitness_review = build_freeze(
        repo_root=repo_root, spec_path=spec_path, frozen_at_utc=frozen_at_utc
    )
    output_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    manifest_tmp = output_root / f".{MANIFEST_NAME}.{token}.tmp"
    receipt_tmp = output_root / f".{RECEIPT_NAME}.{token}.tmp"
    role_tmp = output_root / f".{ROLE_LEDGER_NAME}.{token}.tmp"
    fitness_tmp = output_root / f".{FITNESS_REVIEW_NAME}.{token}.tmp"
    try:
        role_tmp.write_bytes(b"".join(canonical_bytes(row) for row in role_rows))
        fitness_tmp.write_bytes(canonical_bytes(fitness_review))
        role_sha = sha256_file(role_tmp)
        fitness_sha = sha256_file(fitness_tmp)
        manifest["reuse_role_ledger"] = {
            "path": ROLE_LEDGER_NAME,
            "sha256": role_sha,
            "row_count": len(role_rows),
        }
        manifest["fitness_review"] = {
            "path": FITNESS_REVIEW_NAME,
            "sha256": fitness_sha,
            "input_sha256": fitness_review["input_sha256"],
        }
        receipt["reuse_role_ledger"]["sha256"] = role_sha
        receipt["fitness_review"]["sha256"] = fitness_sha
        manifest_tmp.write_bytes(canonical_bytes(manifest))
        receipt["output_manifest"]["sha256"] = sha256_file(manifest_tmp)
        receipt_tmp.write_bytes(canonical_bytes(receipt))
        os.replace(manifest_tmp, manifest_path)
        os.replace(receipt_tmp, receipt_path)
        os.replace(role_tmp, role_path)
        os.replace(fitness_tmp, fitness_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)
        receipt_tmp.unlink(missing_ok=True)
        role_tmp.unlink(missing_ok=True)
        fitness_tmp.unlink(missing_ok=True)
    return manifest_path, receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=MODULE_DIR.parents[2])
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--frozen-at-utc")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else (repo_root / args.spec)
    spec, _ = load_source_spec(repo_root, spec_path)
    output_root = args.output_root or (repo_root / spec["output_root"])
    frozen_at = args.frozen_at_utc or dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_path, receipt_path = write_freeze(
        repo_root=repo_root,
        spec_path=spec_path,
        output_root=output_root.resolve(),
        frozen_at_utc=frozen_at,
    )
    print(
        json.dumps(
            {
                "status": "D0_A0_INPUT_UNIVERSE_FROZEN",
                "manifest": str(manifest_path),
                "receipt": str(receipt_path),
                "manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
