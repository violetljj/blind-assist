#!/usr/bin/env python3
"""Independently validate the write-once D0-A0 input-universe freeze."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from scripts.research.central_obstruction_agent_label_readiness_d0a.freeze_input_universe import (
    discover_reuse_assets,
    load_source_spec,
    role_fields,
)

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = MODULE_DIR / "source_universe_r3.json"
MANIFEST_NAME = "input-universe-manifest.json"
RECEIPT_NAME = "input-universe-receipt.json"
ROLE_LEDGER_NAME = "reuse-role-ledger.jsonl"
FITNESS_REVIEW_NAME = "reuse-fitness-review.json"
VALIDATION_NAME = "input-universe-validation.json"
VALIDATION_SCHEMA = "blindassist.central_obstruction_d0a0_independent_validation.v1"


class ValidationError(ValueError):
    """Fail-closed independent validation error."""


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
        raise ValidationError(f"{where}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"{where}: expected object")
    return value


def inside(root: Path, relative: str, *, file: bool, where: str) -> Path:
    declared = Path(relative)
    if declared.is_absolute() or ".." in declared.parts:
        raise ValidationError(f"{where}: path escapes repository")
    candidate = root.absolute() / declared
    if file and not candidate.is_file():
        raise ValidationError(f"{where}: missing file")
    if not file and not candidate.is_dir():
        raise ValidationError(f"{where}: missing directory")
    return candidate


def normalized_source_row(
    row: dict[str, Any],
    *,
    session: dict[str, Any],
    ledger: Path,
    payload_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], int]:
    if session["ledger_adapter"] == "CROWDBOT_FRAMES_V1":
        frame_raw = row.get("frame_id")
        if row.get("sequence_id") != session["session_id"]:
            raise ValidationError(f"{session['session_id']}: sequence mismatch")
        rel = row.get("rgb_path")
        expected_sha = row.get("rgb_sha256")
        width, height = session["expected_width"], session["expected_height"]
    elif session["ledger_adapter"] == "PUBLIC_VIDEO_REPLAY_RGB_V1":
        frame_raw = row.get("frame_id")
        rel = row.get("image_path")
        expected_sha = row.get("image_sha256")
        width, height = row.get("width"), row.get("height")
    else:
        raise ValidationError(f"{session['session_id']}: unknown adapter")
    try:
        frame_index = int(frame_raw)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{session['session_id']}: bad frame id") from error
    timestamp = row.get("source_capture_timestamp_ns")
    if row.get("source_id") != session["source_id"]:
        raise ValidationError(f"{session['session_id']}: source mismatch")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
        raise ValidationError(f"{session['session_id']}: bad timestamp")
    if width != session["expected_width"] or height != session["expected_height"]:
        raise ValidationError(f"{session['session_id']}: dimension mismatch")
    if not isinstance(rel, str):
        raise ValidationError(f"{session['session_id']}: bad payload path")
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValidationError(f"{session['session_id']}: payload escape")
    payload = ledger.parent / rel
    try:
        payload.resolve().relative_to(payload_root.resolve())
    except ValueError as error:
        raise ValidationError(f"{session['session_id']}: payload escape") from error
    if not payload.is_file():
        raise ValidationError(f"{session['session_id']}: payload missing")
    actual_sha = sha256_file(payload)
    if actual_sha != expected_sha:
        raise ValidationError(f"{session['session_id']} frame {frame_index}: payload hash mismatch")
    record = {
        "source_id": session["source_id"],
        "session_id": session["session_id"],
        "source_ancestry_group": session["source_ancestry_group"],
        "frame_index": frame_index,
        "source_frame_id": str(frame_raw),
        "source_capture_timestamp_ns": timestamp,
        "timestamp_semantics": session["timestamp_semantics"],
        "image_path": payload.absolute().relative_to(repo_root.absolute()).as_posix(),
        "image_sha256": actual_sha,
        "width": width,
        "height": height,
    }
    return record, payload.stat().st_size


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception as error:
        raise ValidationError(f"cannot read image dimensions: {path}: {error}") from error


def load_jsonl(path: Path, *, where: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise ValidationError(f"{where}: blank row {line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValidationError(f"{where}: invalid JSON row {line_number}") from error
            if not isinstance(row, dict):
                raise ValidationError(f"{where}: non-object row {line_number}")
            rows.append(row)
    return rows


def validate(
    *,
    repo_root: Path,
    spec_path: Path,
    output_root: Path,
    validated_at_utc: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    spec_path = spec_path.resolve()
    spec, predecessor_spec = load_source_spec(repo_root, spec_path)
    manifest_path = output_root / MANIFEST_NAME
    receipt_path = output_root / RECEIPT_NAME
    role_path = output_root / ROLE_LEDGER_NAME
    fitness_path = output_root / FITNESS_REVIEW_NAME
    manifest = load_object(manifest_path, where="manifest")
    receipt = load_object(receipt_path, where="receipt")
    if manifest.get("schema_version") != "blindassist.central_obstruction_d0a0_input_universe.v1":
        raise ValidationError("manifest schema mismatch")
    if receipt.get("schema_version") != "blindassist.central_obstruction_d0a0_input_universe_receipt.v1":
        raise ValidationError("receipt schema mismatch")
    if receipt.get("output_manifest", {}).get("sha256") != sha256_file(manifest_path):
        raise ValidationError("manifest receipt hash mismatch")
    if manifest.get("source_spec", {}).get("sha256") != sha256_file(spec_path):
        raise ValidationError("source spec binding mismatch")
    if manifest.get("candidate_output_access") is not False or receipt.get("candidate_output_access") is not False:
        raise ValidationError("candidate-output firewall was not preserved")
    if manifest.get("labels_generated") is not False or receipt.get("labels_generated") is not False:
        raise ValidationError("D0-A0 must not generate labels")
    if receipt.get("truth_or_review_label_access") is not False:
        raise ValidationError("truth/review label firewall was not preserved")
    protocol_path = inside(repo_root, spec["protocol_path"], file=True, where="protocol")
    protocol = load_object(protocol_path, where="protocol")
    if manifest.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValidationError("current protocol binding mismatch")
    required_before = protocol.get("artifact_contract", {}).get("required_before_d0a2")
    if not isinstance(required_before, list) or ROLE_LEDGER_NAME not in required_before:
        raise ValidationError("current protocol does not expose the mandatory reuse-role ledger")
    phase_rows = protocol.get("phase_plan")
    if not isinstance(phase_rows, list) or not any(
        isinstance(row, dict)
        and row.get("phase") == "D0-A0"
        and row.get("candidate_output_access") is False
        and all(
            token in str(row.get("output", "")).lower()
            for token in ("reuse", "role")
        )
        for row in phase_rows
    ):
        raise ValidationError(
            "current protocol D0-A0 does not bind reuse-first roles with candidate-output access disabled"
        )
    workflow_path = inside(repo_root, spec["workflow_path"], file=True, where="workflow")
    if manifest.get("workflow", {}).get("sha256") != sha256_file(workflow_path):
        raise ValidationError("current workflow binding mismatch")
    if manifest.get("predecessor_source_spec") != predecessor_spec:
        raise ValidationError("source-spec predecessor binding mismatch")
    if receipt.get("reuse_role_ledger", {}).get("sha256") != sha256_file(role_path):
        raise ValidationError("reuse-role ledger receipt hash mismatch")
    if manifest.get("reuse_role_ledger", {}).get("sha256") != sha256_file(role_path):
        raise ValidationError("reuse-role ledger manifest hash mismatch")
    if receipt.get("fitness_review", {}).get("sha256") != sha256_file(fitness_path):
        raise ValidationError("fitness-review receipt hash mismatch")
    if manifest.get("fitness_review", {}).get("sha256") != sha256_file(fitness_path):
        raise ValidationError("fitness-review manifest hash mismatch")

    expected_frames: list[dict[str, Any]] = []
    expected_sessions: list[dict[str, Any]] = []
    admitted_role_rows: list[dict[str, Any]] = []
    fitness_input_sessions: list[dict[str, Any]] = []
    admitted_payload_shas: set[str] = set()
    global_digest = hashlib.sha256()
    total_bytes = 0
    opened = {
        spec_path.relative_to(repo_root).as_posix(),
        spec["protocol_path"],
        spec["workflow_path"],
    }
    if predecessor_spec is not None:
        opened.add(predecessor_spec["path"])
    keys: set[tuple[str, str]] = set()
    for session in spec["sessions"]:
        key = (session["source_id"], session["session_id"])
        if key in keys:
            raise ValidationError(f"duplicate session {key}")
        keys.add(key)
        ledger = inside(repo_root, session["ledger_path"], file=True, where=f"{key}.ledger")
        payload_root = inside(repo_root, session["payload_root"], file=False, where=f"{key}.payload")
        if sha256_file(ledger) != session["ledger_sha256"]:
            raise ValidationError(f"{key}: ledger hash mismatch")
        opened.add(session["ledger_path"])
        ancestry = []
        for path_text in session["ancestry_receipt_paths"]:
            path = inside(repo_root, path_text, file=True, where=f"{key}.ancestry")
            load_object(path, where=f"{key}.ancestry")
            opened.add(path_text)
            ancestry.append({"path": path_text, "sha256": sha256_file(path)})
        source_payloads = []
        for binding in session.get("source_payload_paths", []):
            path = inside(repo_root, binding["path"], file=True, where=f"{key}.source_payload")
            digest = sha256_file(path)
            if digest != binding.get("sha256"):
                raise ValidationError(f"{key}: source payload hash mismatch")
            opened.add(binding["path"])
            admitted_payload_shas.add(digest)
            source_payloads.append({"path": binding["path"], "sha256": digest, "bytes": path.stat().st_size})

        rows = []
        session_digest = hashlib.sha256()
        previous_timestamp: int | None = None
        if session["ledger_adapter"] == "IMAGE_DIRECTORY_TIMESTAMP_STEM_V1":
            image_paths = sorted(payload_root.glob(session.get("payload_glob", "*.jpg")))
            source_rows = []
            for index, image_path in enumerate(image_paths):
                try:
                    timestamp = int(image_path.stem)
                except ValueError as error:
                    raise ValidationError(f"{key}: filename timestamp mismatch") from error
                width, height = image_dimensions(image_path)
                if f"{width}x{height}" not in set(session.get("allowed_dimensions", [])):
                    raise ValidationError(f"{key}: native dimension mismatch")
                source_rows.append(
                    (
                        index + 1,
                        {
                            "source_id": session["source_id"],
                            "session_id": session["session_id"],
                            "source_ancestry_group": session["source_ancestry_group"],
                            "frame_index": index,
                            "source_frame_id": str(timestamp),
                            "source_capture_timestamp_ns": timestamp,
                            "timestamp_semantics": session["timestamp_semantics"],
                            "image_path": image_path.absolute().relative_to(repo_root.absolute()).as_posix(),
                            "image_sha256": sha256_file(image_path),
                            "width": width,
                            "height": height,
                        },
                        image_path.stat().st_size,
                    )
                )
        else:
            source_rows = []
            with ledger.open("r", encoding="utf-8") as stream:
                for line_number, raw in enumerate(stream, start=1):
                    if not raw.strip():
                        raise ValidationError(f"{key}: blank row {line_number}")
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError as error:
                        raise ValidationError(f"{key}: bad JSON row {line_number}") from error
                    if not isinstance(row, dict):
                        raise ValidationError(f"{key}: non-object row {line_number}")
                    record, payload_bytes = normalized_source_row(
                        row,
                        session=session,
                        ledger=ledger,
                        payload_root=payload_root,
                        repo_root=repo_root,
                    )
                    source_rows.append((line_number, record, payload_bytes))
        for line_number, record, payload_bytes in source_rows:
            if record["frame_index"] != len(rows):
                raise ValidationError(f"{key}: non-contiguous frame index")
            timestamp = record["source_capture_timestamp_ns"]
            if previous_timestamp is not None:
                delta = timestamp - previous_timestamp
                if delta <= 0:
                    raise ValidationError(f"{key}: non-increasing timestamp")
                expected_step = session.get("expected_frame_step_ns")
                if expected_step is not None and delta != expected_step:
                    raise ValidationError(f"{key}: timestamp step mismatch")
            previous_timestamp = timestamp
            encoded = canonical_bytes(record)
            session_digest.update(encoded)
            global_digest.update(encoded)
            total_bytes += payload_bytes
            rows.append(record)
            expected_frames.append(record)
        if len(rows) != session["expected_frame_count"]:
            raise ValidationError(f"{key}: frame count mismatch")
        content_identity = f"ordered_frame_identity_sha256:{session_digest.hexdigest()};frames:{len(rows)}"
        current_role = role_fields(session, content_identity=content_identity)
        expected_sessions.append(
            {
                "source_id": session["source_id"],
                "session_id": session["session_id"],
                "source_ancestry_group": session["source_ancestry_group"],
                "source_kind": session["source_kind"],
                "source_uri": session["source_uri"],
                "license_or_rights_metadata": session["license_or_rights_metadata"],
                "ledger_path": session["ledger_path"],
                "ledger_sha256": session["ledger_sha256"],
                "payload_root": session["payload_root"],
                "frame_count": len(rows),
                "first_frame_index": rows[0]["frame_index"],
                "last_frame_index": rows[-1]["frame_index"],
                "first_timestamp_ns": rows[0]["source_capture_timestamp_ns"],
                "last_timestamp_ns": rows[-1]["source_capture_timestamp_ns"],
                "timestamp_semantics": session["timestamp_semantics"],
                "expected_frame_step_ns": session.get("expected_frame_step_ns"),
                "dimensions": (
                    f"{session['expected_width']}x{session['expected_height']}"
                    if session.get("expected_width") is not None
                    else sorted({f"{row['width']}x{row['height']}" for row in rows})
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
        )
        admitted_role_rows.append(
            {
                "source_id": session["source_id"],
                "session_id": session["session_id"],
                **current_role,
                "source_frame_manifest": {
                    "frame_count": len(rows),
                    "ordered_frame_identity_sha256": session_digest.hexdigest(),
                    "ledger_path": session["ledger_path"],
                    "ledger_sha256": session["ledger_sha256"],
                },
            }
        )
        sample_indices = sorted({round(index * (len(rows) - 1) / 7) for index in range(8)})
        fitness_input_sessions.append(
            {
                "source_id": session["source_id"],
                "session_id": session["session_id"],
                "sample_rule": "8 evenly spaced frames including both endpoints",
                "samples": [
                    {
                        "frame_index": rows[index]["frame_index"],
                        "source_capture_timestamp_ns": rows[index]["source_capture_timestamp_ns"],
                        "image_path": rows[index]["image_path"],
                        "image_sha256": rows[index]["image_sha256"],
                    }
                    for index in sample_indices
                ],
            }
        )

    if manifest.get("frames") != expected_frames:
        raise ValidationError("manifest frame ledger differs from independent reconstruction")
    if manifest.get("sessions") != expected_sessions:
        raise ValidationError("manifest session summaries differ from independent reconstruction")
    identity_sha = global_digest.hexdigest()
    if manifest.get("ordered_frame_identity_sha256") != identity_sha:
        raise ValidationError("global ordered frame identity mismatch")
    if receipt.get("ordered_frame_identity_sha256") != identity_sha:
        raise ValidationError("receipt ordered frame identity mismatch")
    if receipt.get("frame_count") != len(expected_frames) or receipt.get("session_count") != len(expected_sessions):
        raise ValidationError("receipt count mismatch")
    if receipt.get("payload_total_bytes") != total_bytes:
        raise ValidationError("receipt payload byte count mismatch")
    groups = sorted({session["source_ancestry_group"] for session in spec["sessions"]})
    if manifest.get("source_ancestry_groups") != groups or receipt.get("source_ancestry_group_count") != len(groups):
        raise ValidationError("source ancestry group count mismatch")
    if receipt.get("d0a2_production_labeling_started") is not False:
        raise ValidationError("D0-A2 must remain not started")
    if receipt.get("d0b_model_execution_authorized") is not False:
        raise ValidationError("D0-B must remain unauthorized")

    discovered_rows, discovered_opened = discover_reuse_assets(
        repo_root=repo_root,
        spec=spec,
        admitted_payload_shas=admitted_payload_shas,
    )
    opened.update(discovered_opened)
    prompt_path = inside(repo_root, spec["fitness_review"]["prompt_path"], file=True, where="fitness prompt")
    opened.add(spec["fitness_review"]["prompt_path"])
    expected_role_rows = admitted_role_rows + discovered_rows
    expected_role_rows.sort(
        key=lambda row: (row["admission_disposition"], row["dataset_name"], row["source_id"], row["session_id"])
    )
    actual_role_rows = load_jsonl(role_path, where="reuse-role ledger")
    if actual_role_rows != expected_role_rows:
        raise ValidationError("reuse-role ledger differs from independent reconstruction")
    required_role_fields = set(protocol["reuse_first_admission_policy"]["required_session_manifest_fields"])
    for index, row in enumerate(actual_role_rows):
        if not required_role_fields.issubset(row):
            raise ValidationError(f"reuse-role row {index} is missing protocol-required fields")
    disposition_counts: dict[str, int] = {}
    for row in actual_role_rows:
        value = row["admission_disposition"]
        disposition_counts[value] = disposition_counts.get(value, 0) + 1
    disposition_counts = dict(sorted(disposition_counts.items()))
    if receipt.get("reuse_role_ledger", {}).get("row_count") != len(actual_role_rows):
        raise ValidationError("reuse-role receipt row count mismatch")
    if receipt.get("reuse_role_ledger", {}).get("disposition_counts") != disposition_counts:
        raise ValidationError("reuse-role disposition count mismatch")
    if manifest.get("reuse_role_summary", {}).get("disposition_counts") != disposition_counts:
        raise ValidationError("manifest reuse-role disposition count mismatch")

    fitness_input = {
        "schema_version": "blindassist.central_obstruction_d0a0_fitness_input.v1",
        "protocol_id": spec["protocol_id"],
        "phase": "D0-A0",
        "candidate_output_visible": False,
        "truth_or_review_label_visible": False,
        "sessions": fitness_input_sessions,
    }
    fitness_input_sha = hashlib.sha256(canonical_bytes(fitness_input)).hexdigest()
    expected_fitness_review = {
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
    if load_object(fitness_path, where="fitness review") != expected_fitness_review:
        raise ValidationError("fitness review differs from independently reconstructed input binding")
    if receipt.get("fitness_review", {}).get("input_sha256") != fitness_input_sha:
        raise ValidationError("fitness review input SHA mismatch")
    if receipt.get("opened_input_paths") != sorted(opened):
        raise ValidationError("opened-input declaration mismatch")

    return {
        "schema_version": VALIDATION_SCHEMA,
        "protocol_id": spec["protocol_id"],
        "phase": "D0-A0",
        "evidence_instance": spec["evidence_instance"],
        "status": "VALID",
        "decision": "D0_A0_INPUT_UNIVERSE_FREEZE_VALID",
        "validated_at_utc": validated_at_utc,
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
        "source_spec_sha256": sha256_file(spec_path),
        "source_ancestry_group_count": len(groups),
        "session_count": len(expected_sessions),
        "frame_count": len(expected_frames),
        "payload_total_bytes": total_bytes,
        "ordered_frame_identity_sha256": identity_sha,
        "reuse_role_row_count": len(actual_role_rows),
        "reuse_role_disposition_counts": disposition_counts,
        "fitness_review_input_sha256": fitness_input_sha,
        "candidate_output_access": False,
        "truth_or_review_label_access": False,
        "labels_generated": False,
        "next_permitted_action": "D0-A1_EXCLUDED_CALIBRATION_AND_LOCK_ONLY",
        "d0a2_production_labeling_started": False,
        "d0b_model_execution_authorized": False,
        "errors": [],
    }


def write_validation(
    *,
    repo_root: Path,
    spec_path: Path,
    output_root: Path,
    validated_at_utc: str,
) -> Path:
    destination = output_root / VALIDATION_NAME
    if destination.exists():
        raise ValidationError("formal independent validation already exists; refusing overwrite")
    result = validate(
        repo_root=repo_root,
        spec_path=spec_path,
        output_root=output_root,
        validated_at_utc=validated_at_utc,
    )
    token = uuid.uuid4().hex
    temporary = output_root / f".{VALIDATION_NAME}.{token}.tmp"
    try:
        temporary.write_bytes(canonical_bytes(result))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=MODULE_DIR.parents[2])
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validated-at-utc")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else (repo_root / args.spec)
    spec = load_object(spec_path, where="source spec")
    output_root = args.output_root or (repo_root / spec["output_root"])
    validated_at = args.validated_at_utc or dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    path = write_validation(
        repo_root=repo_root,
        spec_path=spec_path,
        output_root=output_root.resolve(),
        validated_at_utc=validated_at,
    )
    print(json.dumps({"status": "VALID", "validation": str(path), "sha256": sha256_file(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
