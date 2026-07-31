"""Freeze the content-blind fresh fixed-clip calibration input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from bisect import bisect_left
from pathlib import Path
from typing import Any

import cv2

from .freeze_d0a1_pilot import draw_review_frame, make_contact_sheet
from .freeze_input_universe import canonical_bytes, sha256_file


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "docs/research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR_PROTOCOL_2026-07-31.json"
DEFAULT_D0A0_MANIFEST = REPO_ROOT / "artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a-r3/input-universe-manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a-successor-r0"
MANIFEST_NAME = "calibration-input-manifest.json"
RECEIPT_NAME = "calibration-input-receipt.json"
WINDOW_DURATION_NS = 1_000_000_000
SLOT_OFFSETS_NS = (0, 250_000_000, 500_000_000, 750_000_000)
ROI = [0.25, 0.15, 0.75, 0.95]
MIN_ROI_WIDTH = 64
MIN_ROI_HEIGHT = 64


class SuccessorFreezeError(ValueError):
    """Raised when the successor input cannot be frozen fail-closed."""


def load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SuccessorFreezeError(f"{where}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise SuccessorFreezeError(f"{where}: expected JSON object")
    return value


def relative_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise SuccessorFreezeError(f"path escapes repository: {path}") from error


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def selected_session_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = protocol.get("fresh_calibration_selection", {}).get("sessions")
    if not isinstance(rows, list) or len(rows) != 3:
        raise SuccessorFreezeError("successor protocol must freeze exactly three sessions")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SuccessorFreezeError("successor session selection row is invalid")
        key = (row.get("source_id"), row.get("session_id"))
        if key in seen or not all(isinstance(part, str) and part for part in key):
            raise SuccessorFreezeError("successor session selection is duplicated or incomplete")
        seen.add(key)
        if row.get("window_ordinals") != [0, 1]:
            raise SuccessorFreezeError(f"successor selection must contain windows 0 and 1: {key}")
    return rows


def index_d0a0_frames(d0a0: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    sessions = d0a0.get("sessions")
    frames = d0a0.get("frames")
    if not isinstance(sessions, list) or not isinstance(frames, list):
        raise SuccessorFreezeError("D0-A0 manifest sessions or frames are missing")
    session_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sessions:
        if not isinstance(row, dict):
            raise SuccessorFreezeError("D0-A0 session row is invalid")
        key = (row.get("source_id"), row.get("session_id"))
        if key in session_index:
            raise SuccessorFreezeError(f"D0-A0 session identity is duplicated: {key}")
        session_index[key] = row
    frame_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in frames:
        if not isinstance(row, dict):
            raise SuccessorFreezeError("D0-A0 frame row is invalid")
        key = (row.get("source_id"), row.get("session_id"))
        frame_index.setdefault(key, []).append(row)
    for key, rows in frame_index.items():
        rows.sort(key=lambda item: item.get("frame_index", -1))
        indices = [row.get("frame_index") for row in rows]
        timestamps = [row.get("source_capture_timestamp_ns") for row in rows]
        if indices != list(range(len(indices))):
            raise SuccessorFreezeError(f"D0-A0 frame indices are not contiguous: {key}")
        if any(not isinstance(value, int) for value in timestamps) or timestamps != sorted(timestamps):
            raise SuccessorFreezeError(f"D0-A0 timestamps are not ordered: {key}")
    return session_index, frame_index


def choose_slot_frames(
    rows: list[dict[str, Any]],
    *,
    start_fraction: float,
    max_jitter_ns: int,
) -> tuple[int, list[dict[str, Any]]]:
    if len(rows) < 8:
        raise SuccessorFreezeError("selected session is too short for the frozen windows")
    timestamps = [row["source_capture_timestamp_ns"] for row in rows]
    first = timestamps[0]
    last = timestamps[-1]
    available = last - first - WINDOW_DURATION_NS
    if available <= 0:
        raise SuccessorFreezeError("selected session has no one-second calibration window")
    start_ns = first + round(available * start_fraction)
    picks: list[dict[str, Any]] = []
    for offset_ns in SLOT_OFFSETS_NS:
        requested_ns = start_ns + offset_ns
        position = bisect_left(timestamps, requested_ns)
        if position >= len(rows):
            raise SuccessorFreezeError("frozen slot is past the selected session")
        pick = rows[position]
        jitter = abs(pick["source_capture_timestamp_ns"] - requested_ns)
        if jitter > max_jitter_ns:
            raise SuccessorFreezeError(
                f"frozen slot timestamp jitter exceeds bound: {pick['frame_index']} / {jitter} ns"
            )
        if picks and pick["frame_index"] <= picks[-1]["frame_index"]:
            raise SuccessorFreezeError("frozen slot frames are not strictly increasing")
        picks.append(pick)
    return start_ns, picks


def source_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source_id",
        "session_id",
        "dataset_name",
        "dimensions",
        "frame_count",
        "ordered_frame_identity_sha256",
        "ledger_path",
        "ledger_sha256",
        "payload_root",
        "timestamp_semantics",
        "source_ancestry_group",
        "independence_group",
    )
    return {key: row.get(key) for key in keys}


def freeze_bundle(
    *,
    repo_root: Path,
    protocol_path: Path,
    d0a0_manifest_path: Path,
    output_root: Path,
    frozen_at_utc: str,
) -> tuple[Path, Path]:
    if output_root.exists():
        raise SuccessorFreezeError(f"successor output root already exists: {output_root}")
    protocol = load_json(protocol_path, where="successor protocol")
    d0a0 = load_json(d0a0_manifest_path, where="D0-A0 manifest")
    if protocol.get("protocol_id") != "CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR":
        raise SuccessorFreezeError("successor protocol id mismatch")
    if d0a0.get("candidate_output_access") is not False or d0a0.get("labels_generated") is not False:
        raise SuccessorFreezeError("D0-A0 candidate-output or label firewall is open")
    selection = selected_session_rows(protocol)
    burned = set(protocol.get("predecessors", {}).get("burned_source_ids", []))
    selected_ids = {row["source_id"] for row in selection}
    if selected_ids.intersection(burned):
        raise SuccessorFreezeError("fresh successor selection overlaps a burned D0-A1 source")
    session_index, frame_index = index_d0a0_frames(d0a0)
    max_jitter_ns = int(protocol["analysis_unit"]["maximum_slot_jitter_ns"])
    window_fractions = protocol["analysis_unit"]["window_start_fractions"]
    if window_fractions != [0.25, 0.75]:
        raise SuccessorFreezeError("successor window fractions drifted from the frozen protocol")

    selected_source_summaries: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    used_frame_keys: set[tuple[str, str, int]] = set()
    for selection_row in selection:
        key = (selection_row["source_id"], selection_row["session_id"])
        if key not in session_index or key not in frame_index:
            raise SuccessorFreezeError(f"selected D0-A0 session is missing: {key}")
        source = session_index[key]
        if source.get("admission_disposition") != "ADMIT_D0_A_PRODUCTION_LABELING":
            raise SuccessorFreezeError(f"selected session is not D0-A0 production-universe: {key}")
        rows = frame_index[key]
        selected_source_summaries.append(source_summary(source))
        for window_ordinal, fraction in enumerate(window_fractions):
            start_ns, picks = choose_slot_frames(
                rows,
                start_fraction=float(fraction),
                max_jitter_ns=max_jitter_ns,
            )
            unit_id = f"{selection_row['session_id']}__fixed_clip_{window_ordinal:02d}"
            observations: list[dict[str, Any]] = []
            for slot_ordinal, (offset_ns, frame) in enumerate(zip(SLOT_OFFSETS_NS, picks, strict=True)):
                frame_key = (frame["source_id"], frame["session_id"], frame["frame_index"])
                if frame_key in used_frame_keys:
                    raise SuccessorFreezeError(f"frame selected twice: {frame_key}")
                used_frame_keys.add(frame_key)
                image_path = (repo_root / frame["image_path"]).resolve()
                if not image_path.is_file():
                    raise SuccessorFreezeError(f"source RGB payload is missing: {image_path}")
                actual_sha = sha256_file(image_path)
                if actual_sha != frame.get("image_sha256"):
                    raise SuccessorFreezeError(f"source RGB hash mismatch: {image_path}")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise SuccessorFreezeError(f"cannot decode source RGB payload: {image_path}")
                height, width = image.shape[:2]
                roi_width = round((ROI[2] - ROI[0]) * width)
                roi_height = round((ROI[3] - ROI[1]) * height)
                if roi_width < MIN_ROI_WIDTH or roi_height < MIN_ROI_HEIGHT:
                    raise SuccessorFreezeError(f"native ROI is too small: {unit_id}/{slot_ordinal}")
                requested_ns = start_ns + offset_ns
                observations.append(
                    {
                        "unit_id": unit_id,
                        "clip_id": unit_id,
                        "source_id": frame["source_id"],
                        "session_id": frame["session_id"],
                        "slot_ordinal": slot_ordinal,
                        "source_frame_index": frame["frame_index"],
                        "source_capture_timestamp_ns": frame["source_capture_timestamp_ns"],
                        "requested_timestamp_ns": requested_ns,
                        "timestamp_jitter_ns": frame["source_capture_timestamp_ns"] - requested_ns,
                        "width": width,
                        "height": height,
                        "source_image_path": frame["image_path"],
                        "source_image_sha256": frame["image_sha256"],
                        "roi_normalized_xyxy": ROI,
                        "claim_critical": True,
                        "candidate_output_visible": False,
                        "prior_review_visible": False,
                        "review_image_path": None,
                        "review_image_sha256": None,
                    }
                )
            units.append(
                {
                    "unit_id": unit_id,
                    "clip_id": unit_id,
                    "source_id": key[0],
                    "session_id": key[1],
                    "window_ordinal": window_ordinal,
                    "window_start_fraction": float(window_fractions[window_ordinal]),
                    "window_start_timestamp_ns": start_ns,
                    "window_end_timestamp_ns": start_ns + WINDOW_DURATION_NS,
                    "boundary_rule": "PROGRAM_RULE_FROM_FROZEN_SESSION_TIMESTAMP_AND_SLOT_OFFSETS",
                    "observation_count": len(observations),
                    "observations": observations,
                }
            )

    if len(units) != 6 or sum(len(unit["observations"]) for unit in units) != 24:
        raise SuccessorFreezeError("successor calibration count drifted from 6 units / 24 observations")
    observations = [observation for unit in units for observation in unit["observations"]]
    manifest = {
        "schema_version": "blindassist.central_obstruction_d0a_successor_calibration_input.v1",
        "protocol_id": protocol["protocol_id"],
        "phase": protocol["phase"],
        "evidence_instance": protocol["evidence_instance"],
        "status": "FRESH_CALIBRATION_INPUTS_FROZEN",
        "frozen_at_utc": frozen_at_utc,
        "protocol": {
            "path": relative_posix(protocol_path, repo_root),
            "sha256": sha256_file(protocol_path),
        },
        "d0a0_manifest": {
            "path": relative_posix(d0a0_manifest_path, repo_root),
            "sha256": sha256_file(d0a0_manifest_path),
        },
        "candidate_output_access": False,
        "labels_generated": False,
        "natural_event_grouping_used": False,
        "analysis_unit": protocol["analysis_unit"],
        "observation_contract": protocol["observation_contract"],
        "freshness": {
            "burned_d0a1_source_overlap_count": 0,
            "production_source_overlap_count": len(selected_source_summaries),
            "production_frame_overlap_count": 0,
            "clip_level_fresh": True,
            "session_level_independence_claim": False,
        },
        "calibration_source_count": len(selected_source_summaries),
        "fixed_clip_count": len(units),
        "observation_count": len(observations),
        "source_sessions": selected_source_summaries,
        "fixed_units": units,
        "observations": observations,
        "contact_sheets": [],
        "next_permitted_action": "FRESH_ISOLATED_OBSERVATION_LABEL_REVIEW",
        "d0a2_authorized": False,
        "d0a3_authorized": False,
        "d0a4_authorized": False,
    }
    receipt = {
        "schema_version": "blindassist.central_obstruction_d0a_successor_calibration_input_receipt.v1",
        "protocol_id": protocol["protocol_id"],
        "evidence_instance": protocol["evidence_instance"],
        "status": "FRESH_CALIBRATION_INPUTS_FROZEN",
        "created_at_utc": frozen_at_utc,
        "protocol_sha256": sha256_file(protocol_path),
        "d0a0_manifest_sha256": sha256_file(d0a0_manifest_path),
        "candidate_output_access": False,
        "labels_generated": False,
        "fixed_clip_count": len(units),
        "observation_count": len(observations),
        "output_manifest": {"path": MANIFEST_NAME, "sha256": None},
    }

    output_root.mkdir(parents=True, exist_ok=False)
    inputs_root = output_root / "calibration-inputs"
    inputs_root.mkdir()
    try:
        for unit in units:
            unit_root = inputs_root / unit["unit_id"]
            unit_root.mkdir()
            rendered = []
            for observation in unit["observations"]:
                image = cv2.imread(str(repo_root / observation["source_image_path"]), cv2.IMREAD_COLOR)
                shown = draw_review_frame(
                    image,
                    clip_id=unit["unit_id"],
                    frame_index=observation["source_frame_index"],
                    xyxy=ROI,
                )
                destination = unit_root / f"slot-{observation['slot_ordinal']:02d}.png"
                if not cv2.imwrite(str(destination), shown):
                    raise SuccessorFreezeError(f"cannot write review image: {destination}")
                observation["review_image_path"] = relative_posix(destination, repo_root)
                observation["review_image_sha256"] = sha256_file(destination)
                rendered.append(shown)
            sheet = make_contact_sheet(rendered)
            sheet_path = unit_root / "contact-sheet.jpg"
            if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise SuccessorFreezeError(f"cannot write contact sheet: {sheet_path}")
            manifest["contact_sheets"].append(
                {
                    "unit_id": unit["unit_id"],
                    "path": relative_posix(sheet_path, repo_root),
                    "sha256": sha256_file(sheet_path),
                }
            )
        # The manifest contains the review paths added above; keep its byte form LF-bound.
        manifest_path = output_root / MANIFEST_NAME
        receipt_path = output_root / RECEIPT_NAME
        manifest_tmp = output_root / f".{MANIFEST_NAME}.{uuid.uuid4().hex}.tmp"
        receipt_tmp = output_root / f".{RECEIPT_NAME}.{uuid.uuid4().hex}.tmp"
        manifest_tmp.write_bytes(canonical_bytes(manifest))
        receipt["output_manifest"]["sha256"] = sha256_file(manifest_tmp)
        receipt_tmp.write_bytes(canonical_bytes(receipt))
        os.replace(manifest_tmp, manifest_path)
        os.replace(receipt_tmp, receipt_path)
    except Exception:
        for temporary in output_root.glob(".*.tmp"):
            temporary.unlink(missing_ok=True)
        raise
    return manifest_path, receipt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--d0a0-manifest", type=Path, default=DEFAULT_D0A0_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frozen-at-utc", default="2026-07-31T10:50:00Z")
    args = parser.parse_args()
    manifest_path, receipt_path = freeze_bundle(
        repo_root=args.repo_root.resolve(),
        protocol_path=args.protocol.resolve(),
        d0a0_manifest_path=args.d0a0_manifest.resolve(),
        output_root=args.output_root.resolve(),
        frozen_at_utc=args.frozen_at_utc,
    )
    print(
        json.dumps(
            {
                "status": "FRESH_CALIBRATION_INPUTS_FROZEN",
                "manifest": relative_posix(manifest_path, args.repo_root.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "receipt": relative_posix(receipt_path, args.repo_root.resolve()),
                "receipt_sha256": sha256_file(receipt_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
