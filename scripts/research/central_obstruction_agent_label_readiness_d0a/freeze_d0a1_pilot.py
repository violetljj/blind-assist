#!/usr/bin/env python3
"""Materialize the write-once, candidate-output-blind D0-A1 pilot input bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .freeze_input_universe import canonical_bytes, sha256_file


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_LOCK = MODULE_DIR / "d0a1_lock_r2.json"
MANIFEST_NAME = "pilot-input-manifest.json"
RECEIPT_NAME = "pilot-input-receipt.json"
VALID_CALIBRATION_DISPOSITION = "ADMIT_D0_A_CALIBRATION_ONLY"
PRODUCTION_DISPOSITION = "ADMIT_D0_A_PRODUCTION_LABELING"


class PilotFreezeError(ValueError):
    """Fail-closed D0-A1 input-freeze error."""


def load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotFreezeError(f"{where}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise PilotFreezeError(f"{where}: expected JSON object")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if key == "inherits_path":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_lock(
    repo_root: Path,
    lock_path: Path,
    *,
    visited: set[Path] | None = None,
) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    seen = set() if visited is None else visited
    if lock_path in seen:
        raise PilotFreezeError("D0-A1 lock inheritance contains a cycle")
    seen.add(lock_path)
    raw = load_json(lock_path, where="D0-A1 lock")
    inherited = raw.get("inherits_path")
    if inherited is None:
        return raw
    base_path = repo_path(repo_root, inherited, where="D0-A1 predecessor lock")
    if not base_path.is_file():
        raise PilotFreezeError("D0-A1 predecessor lock is missing")
    base = load_lock(repo_root, base_path, visited=seen)
    return deep_merge(base, raw)


def relative_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise PilotFreezeError(f"path escapes repository: {path}") from error


def repo_path(repo_root: Path, declared: str, *, where: str) -> Path:
    if not isinstance(declared, str) or not declared:
        raise PilotFreezeError(f"{where}: expected relative path")
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        raise PilotFreezeError(f"{where}: path escapes repository")
    return repo_root.resolve() / relative


def load_jsonl(path: Path, *, where: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PilotFreezeError(f"{where}:{line_number}: expected object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise PilotFreezeError(f"{where}: cannot read JSONL: {error}") from error
    return rows


def validate_bindings(repo_root: Path, lock: dict[str, Any]) -> tuple[Path, Path]:
    if (
        lock.get("protocol_id") != "CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A"
        or lock.get("phase") != "D0-A1"
        or lock.get("candidate_output_access") is not False
    ):
        raise PilotFreezeError("lock does not preserve the D0-A1 candidate-output firewall")
    bindings = lock.get("bindings")
    if not isinstance(bindings, dict):
        raise PilotFreezeError("lock bindings are missing")
    for name in ("protocol", "ai_workflow", "d0a0_manifest", "d0a0_reuse_role_ledger"):
        binding = bindings.get(name)
        if not isinstance(binding, dict):
            raise PilotFreezeError(f"missing {name} binding")
        path = repo_path(repo_root, binding.get("path"), where=name)
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise PilotFreezeError(f"{name} binding mismatch")
    protocol_path = repo_path(repo_root, bindings["protocol"]["path"], where="protocol")
    protocol = load_json(protocol_path, where="protocol")
    phase_rows = protocol.get("phase_plan")
    if not isinstance(phase_rows, list) or not any(
        isinstance(row, dict)
        and row.get("phase") == "D0-A1"
        and row.get("candidate_output_access") is False
        for row in phase_rows
    ):
        raise PilotFreezeError("current protocol does not authorize candidate-blind D0-A1")
    workflow_path = repo_path(repo_root, bindings["ai_workflow"]["path"], where="workflow")
    workflow = load_json(workflow_path, where="workflow")
    workflow_row = workflow.get("workflows", {}).get("central_obstruction_agent_label_readiness_d0a_v1")
    if not isinstance(workflow_row, dict) or workflow_row.get("candidate_output_hidden_from_reviewers") is not True:
        raise PilotFreezeError("AI workflow candidate-output firewall mismatch")
    prompt_binding = bindings.get("review_prompt")
    if not isinstance(prompt_binding, dict):
        raise PilotFreezeError("review prompt binding is missing")
    prompt_path = repo_path(repo_root, prompt_binding.get("path"), where="review_prompt")
    if not prompt_path.is_file():
        raise PilotFreezeError("review prompt is missing")
    role_path = repo_path(repo_root, bindings["d0a0_reuse_role_ledger"]["path"], where="role ledger")
    return prompt_path, role_path


def ordered_asset_identity(repo_root: Path, directory: Path) -> str:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    images = 0
    for path in files:
        digest.update(
            canonical_bytes(
                {
                    "path": relative_posix(path, repo_root),
                    "sha256": sha256_file(path),
                }
            )
        )
        total_bytes += path.stat().st_size
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            images += 1
    return (
        f"ordered_asset_sha256:{digest.hexdigest()};files:{len(files)};"
        f"images:{images};bytes:{total_bytes}"
    )


def validate_sources(
    *,
    repo_root: Path,
    lock: dict[str, Any],
    role_path: Path,
) -> dict[str, dict[str, Any]]:
    role_rows = load_jsonl(role_path, where="D0-A0 reuse-role ledger")
    roles = {row.get("source_id"): row for row in role_rows}
    production_ids = {
        row.get("source_id")
        for row in role_rows
        if row.get("admission_disposition") == PRODUCTION_DISPOSITION
    }
    sources = lock.get("calibration_sources")
    if not isinstance(sources, list) or len(sources) < 3:
        raise PilotFreezeError("at least three calibration sources are required")
    by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
            raise PilotFreezeError("invalid calibration source row")
        source_id = source["source_id"]
        if source_id in by_id or source_id in production_ids:
            raise PilotFreezeError(f"duplicate or production source selected: {source_id}")
        role = roles.get(source_id)
        if (
            not isinstance(role, dict)
            or role.get("admission_disposition") != VALID_CALIBRATION_DISPOSITION
            or role.get("session_id") != source.get("session_id")
        ):
            raise PilotFreezeError(f"source is not D0-A0 calibration-only: {source_id}")
        path = repo_path(repo_root, source.get("path"), where=f"source {source_id}")
        kind = source.get("kind")
        if kind == "video":
            if not path.is_file() or sha256_file(path) != source.get("sha256"):
                raise PilotFreezeError(f"video identity mismatch: {source_id}")
        elif kind == "image_sequence":
            if not path.is_dir() or ordered_asset_identity(repo_root, path) != source.get("content_identity"):
                raise PilotFreezeError(f"image-sequence identity mismatch: {source_id}")
        else:
            raise PilotFreezeError(f"unsupported source kind: {kind}")
        by_id[source_id] = {**source, "_path": path}
    return by_id


def validate_lock_contract(lock: dict[str, Any]) -> None:
    roi = lock.get("roi")
    if not isinstance(roi, dict):
        raise PilotFreezeError("ROI is missing")
    xyxy = roi.get("normalized_xyxy")
    if (
        not isinstance(xyxy, list)
        or len(xyxy) != 4
        or not all(isinstance(value, (int, float)) for value in xyxy)
        or not (0 <= xyxy[0] < xyxy[2] <= 1 and 0 <= xyxy[1] < xyxy[3] <= 1)
    ):
        raise PilotFreezeError("ROI normalized_xyxy is invalid")
    labels = lock.get("observation_contract", {}).get("labels")
    if labels != [
        "VISIBLE_CENTRAL_OBSTRUCTION_PRESENT",
        "NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE",
        "NOT_EVALUABLE",
    ]:
        raise PilotFreezeError("three-state observation vocabulary mismatch")
    audit = lock.get("audit_rule")
    if (
        not isinstance(audit, dict)
        or audit.get("claim_critical_review_passes") != 2
        or audit.get("fresh_context_per_independent_pass") is not True
        or audit.get("other_review_hidden_before_submission") is not True
        or audit.get("human_queue_required") is not False
    ):
        raise PilotFreezeError("risk-tiered audit contract mismatch")
    thresholds = lock.get("readiness_thresholds")
    if not isinstance(thresholds, dict):
        raise PilotFreezeError("readiness thresholds are missing")
    for key in (
        "overall_observation_label_agreement",
        "claim_critical_observation_label_agreement",
        "parent_event_match_rate",
    ):
        if not 0 < thresholds.get(key, 0) <= 1:
            raise PilotFreezeError(f"invalid readiness threshold: {key}")
    if "missing isolated-review evidence is NOT_READY" not in thresholds.get("decision_rule", ""):
        raise PilotFreezeError("empty-denominator fail-closed rule is missing")


def raw_pixel_sha(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes(order="C")).hexdigest()


def decode_video_frames(path: Path, indices: list[int]) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PilotFreezeError(f"cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded: dict[int, np.ndarray] = {}
    for index in sorted(set(indices)):
        if not 0 <= index < frame_count:
            capture.release()
            raise PilotFreezeError(f"video frame index out of range: {index}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise PilotFreezeError(f"cannot decode video frame {index}: {path}")
        decoded[index] = frame
    capture.release()
    return decoded, {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def draw_review_frame(frame: np.ndarray, *, clip_id: str, frame_index: int, xyxy: list[float]) -> np.ndarray:
    shown = frame.copy()
    height, width = shown.shape[:2]
    x1, y1, x2, y2 = (
        int(round(xyxy[0] * width)),
        int(round(xyxy[1] * height)),
        int(round(xyxy[2] * width)),
        int(round(xyxy[3] * height)),
    )
    cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        shown,
        f"{clip_id} / {frame_index}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return shown


def make_contact_sheet(frames: list[np.ndarray], *, target_width: int = 320) -> np.ndarray:
    rendered: list[np.ndarray] = []
    for frame in frames:
        scale = target_width / frame.shape[1]
        rendered.append(cv2.resize(frame, (target_width, max(1, round(frame.shape[0] * scale)))))
    target_height = max(frame.shape[0] for frame in rendered)
    padded = [
        cv2.copyMakeBorder(frame, 0, target_height - frame.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for frame in rendered
    ]
    return np.hstack(padded)


def build_bundle(
    *,
    repo_root: Path,
    lock_path: Path,
    frozen_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[tuple[int, np.ndarray]]]]:
    lock = load_lock(repo_root, lock_path)
    validate_lock_contract(lock)
    prompt_path, role_path = validate_bindings(repo_root, lock)
    sources = validate_sources(repo_root=repo_root, lock=lock, role_path=role_path)
    clips = lock.get("pilot_clips")
    if not isinstance(clips, list):
        raise PilotFreezeError("pilot clips are missing")
    thresholds = lock["readiness_thresholds"]
    if len(clips) < thresholds["minimum_pilot_clips"]:
        raise PilotFreezeError("pilot clip count is below the frozen minimum")
    clip_ids: set[str] = set()
    observations: list[dict[str, Any]] = []
    clip_frames: dict[str, list[tuple[int, np.ndarray]]] = {}
    metadata: dict[str, Any] = {}
    source_indices: dict[str, list[int]] = {}
    for clip in clips:
        if not isinstance(clip, dict) or not isinstance(clip.get("clip_id"), str):
            raise PilotFreezeError("invalid pilot clip")
        clip_id = clip["clip_id"]
        if clip_id in clip_ids:
            raise PilotFreezeError(f"duplicate clip id: {clip_id}")
        clip_ids.add(clip_id)
        source_id = clip.get("source_id")
        indices = clip.get("frame_indices")
        if source_id not in sources or not isinstance(indices, list) or len(indices) < 3:
            raise PilotFreezeError(f"invalid pilot clip source/indices: {clip_id}")
        if indices != sorted(set(indices)) or any(not isinstance(index, int) for index in indices):
            raise PilotFreezeError(f"pilot indices must be unique ascending integers: {clip_id}")
        source_indices.setdefault(source_id, []).extend(indices)
    decoded_by_source: dict[str, dict[int, np.ndarray]] = {}
    for source_id, indices in source_indices.items():
        source = sources[source_id]
        if source["kind"] == "video":
            decoded, source_metadata = decode_video_frames(source["_path"], indices)
        else:
            image_paths = sorted(
                path for path in source["_path"].rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            decoded = {}
            for index in sorted(set(indices)):
                if not 0 <= index < len(image_paths):
                    raise PilotFreezeError(f"image-sequence index out of range: {index}")
                frame = cv2.imread(str(image_paths[index]), cv2.IMREAD_COLOR)
                if frame is None:
                    raise PilotFreezeError(f"cannot decode image: {image_paths[index]}")
                decoded[index] = frame
            height, width = next(iter(decoded.values())).shape[:2]
            source_metadata = {
                "frame_count": len(image_paths),
                "fps": None,
                "width": width,
                "height": height,
            }
        decoded_by_source[source_id] = decoded
        metadata[source_id] = source_metadata
    xyxy = lock["roi"]["normalized_xyxy"]
    min_roi_width = lock["roi"]["minimum_native_roi_width_px"]
    min_roi_height = lock["roi"]["minimum_native_roi_height_px"]
    for clip in clips:
        clip_id = clip["clip_id"]
        source_id = clip["source_id"]
        source = sources[source_id]
        clip_frames[clip_id] = []
        for ordinal, frame_index in enumerate(clip["frame_indices"]):
            frame = decoded_by_source[source_id][frame_index]
            height, width = frame.shape[:2]
            roi_width = round((xyxy[2] - xyxy[0]) * width)
            roi_height = round((xyxy[3] - xyxy[1]) * height)
            if roi_width < min_roi_width or roi_height < min_roi_height:
                raise PilotFreezeError(f"native ROI is too small: {clip_id}/{frame_index}")
            observations.append(
                {
                    "clip_id": clip_id,
                    "source_id": source_id,
                    "session_id": source["session_id"],
                    "clip_observation_ordinal": ordinal,
                    "source_frame_index": frame_index,
                    "source_time_seconds": (
                        frame_index / metadata[source_id]["fps"]
                        if metadata[source_id]["fps"]
                        else None
                    ),
                    "width": width,
                    "height": height,
                    "raw_bgr_sha256": raw_pixel_sha(frame),
                    "roi_normalized_xyxy": xyxy,
                    "candidate_output_visible": False,
                    "prior_review_visible": False,
                }
            )
            clip_frames[clip_id].append(
                (frame_index, draw_review_frame(frame, clip_id=clip_id, frame_index=frame_index, xyxy=xyxy))
            )
    if len(observations) < thresholds["minimum_pilot_observations"]:
        raise PilotFreezeError("pilot observation count is below the frozen minimum")
    manifest = {
        "schema_version": "blindassist.central_obstruction_d0a1_pilot_input.v1",
        "protocol_id": lock["protocol_id"],
        "phase": "D0-A1",
        "evidence_instance": lock["evidence_instance"],
        "status": "D0_A1_PILOT_INPUTS_FROZEN",
        "frozen_at_utc": frozen_at_utc,
        "lock": {
            "path": relative_posix(lock_path, repo_root),
            "sha256": sha256_file(lock_path),
        },
        "prompt": {
            "path": relative_posix(prompt_path, repo_root),
            "sha256": sha256_file(prompt_path),
        },
        "d0a0_manifest_sha256": lock["bindings"]["d0a0_manifest"]["sha256"],
        "d0a0_reuse_role_ledger_sha256": lock["bindings"]["d0a0_reuse_role_ledger"]["sha256"],
        "candidate_output_access": False,
        "labels_generated": False,
        "production_source_overlap_count": 0,
        "calibration_source_count": len(sources),
        "pilot_clip_count": len(clips),
        "pilot_observation_count": len(observations),
        "calibration_sources": [
            {key: value for key, value in source.items() if key != "_path"}
            for source in sources.values()
        ],
        "source_metadata": metadata,
        "observations": observations,
        "contact_sheets": [],
        "next_permitted_action": "D0-A1_PRIMARY_SOURCE_ONLY_LABEL_PASS",
        "d0a2_production_labeling_authorized": False,
        "d0b_authorized": False,
    }
    receipt = {
        "schema_version": "blindassist.central_obstruction_d0a1_pilot_input_receipt.v1",
        "protocol_id": lock["protocol_id"],
        "phase": "D0-A1",
        "evidence_instance": lock["evidence_instance"],
        "status": "D0_A1_PILOT_INPUTS_FROZEN",
        "created_at_utc": frozen_at_utc,
        "lock_sha256": sha256_file(lock_path),
        "prompt_sha256": sha256_file(prompt_path),
        "output_manifest": {"path": MANIFEST_NAME, "sha256": None},
        "candidate_output_access": False,
        "labels_generated": False,
        "production_source_overlap_count": 0,
        "d0a2_production_labeling_authorized": False,
        "d0b_authorized": False,
    }
    return manifest, receipt, clip_frames


def write_bundle(
    *,
    repo_root: Path,
    lock_path: Path,
    output_root: Path,
    frozen_at_utc: str,
) -> tuple[Path, Path]:
    manifest_path = output_root / MANIFEST_NAME
    receipt_path = output_root / RECEIPT_NAME
    if manifest_path.exists() or receipt_path.exists():
        raise PilotFreezeError("formal D0-A1 pilot inputs already exist; refusing overwrite")
    manifest, receipt, clip_frames = build_bundle(
        repo_root=repo_root,
        lock_path=lock_path,
        frozen_at_utc=frozen_at_utc,
    )
    token = uuid.uuid4().hex
    output_root.mkdir(parents=True, exist_ok=True)
    frames_root = output_root / "pilot-inputs"
    frames_root.mkdir(exist_ok=False)
    manifest_tmp = output_root / f".{MANIFEST_NAME}.{token}.tmp"
    receipt_tmp = output_root / f".{RECEIPT_NAME}.{token}.tmp"
    try:
        observation_lookup = {
            (row["clip_id"], row["source_frame_index"]): row
            for row in manifest["observations"]
        }
        for clip_id, frames in clip_frames.items():
            clip_root = frames_root / clip_id
            clip_root.mkdir()
            rendered: list[np.ndarray] = []
            for frame_index, shown in frames:
                destination = clip_root / f"{frame_index:09d}.png"
                if not cv2.imwrite(str(destination), shown):
                    raise PilotFreezeError(f"cannot write pilot frame: {destination}")
                row = observation_lookup[(clip_id, frame_index)]
                row["review_image_path"] = relative_posix(destination, repo_root)
                row["review_image_sha256"] = sha256_file(destination)
                rendered.append(shown)
            sheet = make_contact_sheet(rendered)
            sheet_path = clip_root / "contact-sheet.jpg"
            if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise PilotFreezeError(f"cannot write contact sheet: {sheet_path}")
            manifest["contact_sheets"].append(
                {
                    "clip_id": clip_id,
                    "path": relative_posix(sheet_path, repo_root),
                    "sha256": sha256_file(sheet_path),
                }
            )
        manifest_tmp.write_bytes(canonical_bytes(manifest))
        receipt["output_manifest"]["sha256"] = sha256_file(manifest_tmp)
        receipt_tmp.write_bytes(canonical_bytes(receipt))
        os.replace(manifest_tmp, manifest_path)
        os.replace(receipt_tmp, receipt_path)
    except Exception:
        manifest_tmp.unlink(missing_ok=True)
        receipt_tmp.unlink(missing_ok=True)
        raise
    return manifest_path, receipt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = load_lock(repo_root, lock_path)
    output_root = args.output_root or repo_path(repo_root, lock["output_root"], where="output_root")
    frozen_at_utc = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    manifest, receipt = write_bundle(
        repo_root=repo_root,
        lock_path=lock_path,
        output_root=output_root.resolve(),
        frozen_at_utc=frozen_at_utc,
    )
    print(
        json.dumps(
            {
                "status": "D0_A1_PILOT_INPUTS_FROZEN",
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "receipt": str(receipt),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
