#!/usr/bin/env python3
"""Materialize model-blind THOR-MAGNI synchronized frame/window intake.

THOR-MAGNI CSV files merge QTM timestamps, rigid-body tracks, and eye-tracker
scene-frame indices.  This command uses the source ``SceneFNr`` column to bind
one explicitly selected scene recording to QTM time, then emits a non-
overlapping uniform 4-second candidate intake.  It does not inspect RGB
content, use a model, infer event buckets, or merge candidates into the D7
top-level event registry.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from pipeline import ContractError, sha256_file, stable_id, utc_now, write_json, write_jsonl


PAPER_URL = "https://journals.sagepub.com/doi/10.1177/02783649241274794"
LICENSE = "CC-BY-4.0 (Zenodo record metadata; verify member-specific terms before event use)"


def _parse_float(value: str) -> float | None:
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _parse_int(value: str) -> int | None:
    number = _parse_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _meta_values(rows: dict[str, list[str]], key: str) -> list[str]:
    values = rows.get(key) or []
    return [str(value).strip() for value in values[1:] if str(value).strip()]


def _scene_column(header: list[str], device: str) -> tuple[int, str, str]:
    needle = f"{device}_SceneFNr"
    matches = [(index, value) for index, value in enumerate(header) if str(value).endswith(needle)]
    if len(matches) != 1:
        raise ContractError(f"expected one {device} SceneFNr column, found {matches}")
    index, column = matches[0]
    body_name = str(column).split(" ", 1)[0]
    return index, str(column), body_name


def _body_centroid_columns(header: list[str], body_names: Iterable[str]) -> dict[str, tuple[int, int, int]]:
    result: dict[str, tuple[int, int, int]] = {}
    for body in body_names:
        expected = [
            header.index(f"{body} Centroid_X") if f"{body} Centroid_X" in header else -1,
            header.index(f"{body} Centroid_Y") if f"{body} Centroid_Y" in header else -1,
            header.index(f"{body} Centroid_Z") if f"{body} Centroid_Z" in header else -1,
        ]
        if all(index >= 0 for index in expected):
            result[body] = (expected[0], expected[1], expected[2])
    return result


def _read_synchronized_rows(path: Path, *, device: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"THOR-MAGNI scenario CSV not found: {path}")
    metadata: dict[str, list[str]] = {}
    header: list[str] | None = None
    unique_rows: dict[int, dict[str, Any]] = {}
    qtm_times: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if not raw:
                continue
            if raw[0] == "Frame" and len(raw) > 1 and raw[1] == "Time":
                header = raw
                break
            metadata[raw[0]] = raw
        if header is None:
            raise ContractError(f"THOR-MAGNI CSV lacks Frame/Time header: {path}")
        try:
            frame_index_column = header.index("Frame")
            time_column = header.index("Time")
        except ValueError as exc:
            raise ContractError(f"THOR-MAGNI CSV header lacks Frame/Time: {path}") from exc
        scene_index, scene_column, camera_body = _scene_column(header, device)
        body_names = _meta_values(metadata, "BODY_NAMES")
        body_roles = _meta_values(metadata, "BODY_ROLES")
        role_by_body = {body: body_roles[index] for index, body in enumerate(body_names) if index < len(body_roles)}
        centroid_columns = _body_centroid_columns(header, body_names)
        for raw in reader:
            if len(raw) <= max(frame_index_column, time_column, scene_index):
                continue
            qtm_frame = _parse_int(raw[frame_index_column])
            qtm_time_s = _parse_float(raw[time_column])
            scene_frame = _parse_int(raw[scene_index])
            if qtm_frame is None or qtm_time_s is None or scene_frame is None or scene_frame < 0:
                continue
            if qtm_times and qtm_time_s < qtm_times[-1]:
                raise ContractError(f"non-monotone QTM time in {path}: {qtm_time_s} < {qtm_times[-1]}")
            qtm_times.append(qtm_time_s)
            if scene_frame in unique_rows:
                continue
            bodies: dict[str, dict[str, Any]] = {}
            for body, indices in centroid_columns.items():
                values = [_parse_float(raw[index]) if index < len(raw) else None for index in indices]
                bodies[body] = {
                    "role": role_by_body.get(body),
                    "centroid_mm": values,
                }
            unique_rows[scene_frame] = {
                "scene_frame_index": scene_frame,
                "qtm_frame": qtm_frame,
                "qtm_time_s": qtm_time_s,
                "timestamp_ns": int(round(qtm_time_s * 1_000_000_000)),
                "camera_body": camera_body,
                "camera_body_role": role_by_body.get(camera_body),
                "bodies": bodies,
            }
    if not unique_rows:
        raise ContractError(f"no synchronized {device} SceneFNr rows found in {path}")
    ordered = [unique_rows[index] for index in sorted(unique_rows)]
    return {
        "metadata": metadata,
        "header": header,
        "scene_column": scene_column,
        "camera_body": camera_body,
        "body_names": body_names,
        "role_by_body": role_by_body,
        "rows": ordered,
        "source_row_count": len(qtm_times),
        "unique_scene_frame_count": len(ordered),
    }


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _build_window_rows(
    *,
    parsed: dict[str, Any],
    scenario_csv: Path,
    root: Path,
    rgb_path: Path,
    intrinsics_path: Path | None,
    run_id: str,
    window_seconds: float,
    scene_fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if window_seconds <= 0 or scene_fps <= 0:
        raise ContractError("window_seconds and scene_fps must be positive")
    nominal_frames_per_window = int(round(window_seconds * scene_fps))
    if nominal_frames_per_window < 2:
        raise ContractError("window must contain at least two nominal scene frames")
    rows = parsed["rows"]
    by_frame = {int(row["scene_frame_index"]): row for row in rows}
    timestamps = [int(row["timestamp_ns"]) for row in rows]
    if any(next_time <= current_time for current_time, next_time in zip(timestamps, timestamps[1:])):
        raise ContractError("deduplicated SceneFNr timestamps must be strictly increasing")
    scenario_hash = sha256_file(scenario_csv)
    rgb_hash = sha256_file(rgb_path) if rgb_path.is_file() else "MISSING"
    source_hash = hashlib.sha256(f"{scenario_hash}:{rgb_hash}".encode("ascii")).hexdigest()
    file_id = str(parsed["metadata"].get("FILE_ID", ["", ""])[1] if len(parsed["metadata"].get("FILE_ID", [])) > 1 else scenario_csv.stem)
    device = str(parsed["scene_column"]).rsplit(" ", 1)[-1].split("_SceneFNr", 1)[0]
    session_id = stable_id("d7sess", "THOR-MAGNI", file_id, device, rgb_path.name)
    ancestry_group = stable_id("d7anc", "THOR-MAGNI", file_id)
    frame_rows: list[dict[str, Any]] = []
    frame_ids: dict[int, str] = {}
    for row in rows:
        scene_frame = int(row["scene_frame_index"])
        frame_id = stable_id("d7frm", session_id, scene_frame, source_hash)
        frame_ids[scene_frame] = frame_id
        frame_rows.append({
            "schema": "hftf_d7_public_real_frame_v1",
            "dataset_id": "THOR-MAGNI",
            "source_session_id": session_id,
            "ancestry_group": ancestry_group,
            "frame_id": frame_id,
            "frame_index": scene_frame,
            "timestamp_ns": row["timestamp_ns"],
            "rgb_path": _relative_path(root, rgb_path),
            "intrinsics_optional": _relative_path(root, intrinsics_path) if intrinsics_path else None,
            "pose_optional": {
                "qtm_frame": row["qtm_frame"],
                "qtm_time_s": row["qtm_time_s"],
                "camera_body": row["camera_body"],
                "camera_body_role": row["camera_body_role"],
                "tracked_body_centroids_mm": row["bodies"],
            },
            "depth_optional": None,
            "segmentation_optional": None,
            "tracks_optional": {
                "source_csv": _relative_path(root, scenario_csv),
                "time_column": "Time",
                "qtm_frame_column": "Frame",
                "scene_frame_column": parsed["scene_column"],
            },
            "source_metadata": {
                "file_id": file_id,
                "camera_device": device,
                "camera_body": row["camera_body"],
                "camera_body_role": row["camera_body_role"],
                "source_csv": _relative_path(root, scenario_csv),
                "scene_frame_binding": "SOURCE_SCENEFNR_INDEXED_TO_QTM_TIME",
                "timestamp_semantics": "SOURCE_SYNCHRONIZED_QTM_TIME",
                "official_sync_contract": PAPER_URL,
                "source_truth_status": "METADATA_GEOMETRY_ONLY",
            },
            "source_license": LICENSE,
            "provider_revision": "zenodo-record-13865754",
            "source_hash": source_hash,
        })
    candidate_rows: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(rows) - 1:
        start_row = rows[cursor]
        target_timestamp = int(start_row["timestamp_ns"]) + int(round(window_seconds * 1_000_000_000))
        floor_index = bisect.bisect_right(timestamps, target_timestamp, lo=cursor) - 1
        ceil_index = min(floor_index + 1, len(rows) - 1)
        if floor_index <= cursor:
            break
        end_index = min(
            (floor_index, ceil_index),
            key=lambda index: abs(timestamps[index] - target_timestamp),
        )
        if end_index <= cursor:
            break
        selected = rows[cursor : end_index + 1]
        duration_ns = int(selected[-1]["timestamp_ns"]) - int(selected[0]["timestamp_ns"])
        if duration_ns < int(round(window_seconds * 0.95 * 1_000_000_000)):
            break
        start = int(selected[0]["scene_frame_index"])
        end = int(selected[-1]["scene_frame_index"])
        candidate_id = stable_id("d7cand", "THOR-MAGNI", session_id, start, end, source_hash)
        candidate_rows.append({
            "schema": "hftf_d7_public_real_candidate_v1",
            "candidate_id": candidate_id,
            "parent_event_id": stable_id("d7parent", candidate_id),
            "dataset_id": "THOR-MAGNI",
            "source_session_id": session_id,
            "ancestry_group": ancestry_group,
            "source_id": file_id,
            "segment_index": 0,
            "start_frame_index": start,
            "end_frame_index": end,
            "start_timestamp_ns": selected[0]["timestamp_ns"],
            "end_timestamp_ns": selected[-1]["timestamp_ns"],
            "timestamp_semantics": "SOURCE_SYNCHRONIZED_QTM_TIME",
            "frame_ids": [frame_ids[index] for index in range(start, end + 1)],
            "frame_count": len(selected),
            "candidate_selection": "MODEL_BLIND_UNIFORM_SOURCE_VIDEO_COVERAGE",
            "model_output_visible_to_selector": False,
            "native_geometry_used_for_selection": False,
            "native_geometry_available": True,
            "event_bucket": "NOT_EVALUABLE",
            "truth_status": "NOT_EVALUABLE",
            "parent_independence_status": "UNVERIFIED",
            "required_confirmation_selection": "MODEL_BLIND",
            "source_license": LICENSE,
            "source_hash": source_hash,
            "provider_revision": "zenodo-record-13865754",
            "rgb_uri": _relative_path(root, rgb_path),
            "geometry_uri": _relative_path(root, scenario_csv),
            "source_metadata": {
                "file_id": file_id,
                "camera_device": device,
                "scene_frame_column": parsed["scene_column"],
                "source_sync_contract": PAPER_URL,
                "candidate_truth_boundary": "INTAKE_ONLY_UNREVIEWED",
            },
        })
        cursor = end_index + 1
    durations_s = [
        (int(row["end_timestamp_ns"]) - int(row["start_timestamp_ns"])) / 1_000_000_000
        for row in candidate_rows
    ]
    manifest = {
        "schema": "hftf_d7_public_real_thor_magni_window_manifest_v1",
        "run_id": run_id,
        "dataset_id": "THOR-MAGNI",
        "source_session_id": session_id,
        "ancestry_group": ancestry_group,
        "file_id": file_id,
        "camera_device": device,
        "scene_frame_column": parsed["scene_column"],
        "camera_body": parsed["camera_body"],
        "camera_body_role": parsed["role_by_body"].get(parsed["camera_body"]),
        "scenario_csv": str(scenario_csv),
        "rgb_path": str(rgb_path),
        "intrinsics_path": str(intrinsics_path) if intrinsics_path else None,
        "source_hash": source_hash,
        "scenario_csv_sha256": scenario_hash,
        "rgb_sha256": rgb_hash,
        "source_row_count": parsed["source_row_count"],
        "unique_scene_frame_count": parsed["unique_scene_frame_count"],
        "window_seconds": window_seconds,
        "scene_fps": scene_fps,
        "nominal_frames_per_window": nominal_frames_per_window,
        "window_duration_seconds": {
            "target": window_seconds,
            "min": min(durations_s) if durations_s else None,
            "max": max(durations_s) if durations_s else None,
            "mean": sum(durations_s) / len(durations_s) if durations_s else None,
        },
        "candidate_count": len(candidate_rows),
        "frame_count": len(frame_rows),
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
        "notes": [
            "SceneFNr is used only as source-documented scene-frame indexing into synchronized QTM time.",
            "Uniform windows are candidate intake; no RGB content or model output was inspected.",
            "All event buckets remain NOT_EVALUABLE until the independent review/adjudication chain runs.",
        ],
    }
    return frame_rows, candidate_rows, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    scenario_csv = Path(args.scenario_csv).resolve()
    rgb_path = Path(args.rgb_path).resolve()
    intrinsics_path = Path(args.intrinsics_path).resolve() if args.intrinsics_path else None
    parsed = _read_synchronized_rows(scenario_csv, device=args.device)
    frame_rows, candidate_rows, manifest = _build_window_rows(
        parsed=parsed,
        scenario_csv=scenario_csv,
        root=root,
        rgb_path=rgb_path,
        intrinsics_path=intrinsics_path,
        run_id=args.run_id,
        window_seconds=args.window_seconds,
        scene_fps=args.scene_fps,
    )
    frame_path = root / "canonical" / f"thor_magni_frame_registry_{args.run_id}.jsonl"
    candidate_path = root / "candidates" / f"thor_magni_candidate_index_{args.run_id}.jsonl"
    manifest_path = root / "manifests" / f"thor_magni_window_manifest_{args.run_id}.json"
    receipt_path = root / "receipts" / f"thor_magni_window_receipt_{args.run_id}.json"
    for path in (frame_path, candidate_path, manifest_path, receipt_path):
        if path.exists():
            raise ContractError(f"output already exists; refusing overwrite: {path}")
    write_jsonl(frame_path, frame_rows)
    write_jsonl(candidate_path, candidate_rows)
    write_json(manifest_path, manifest)
    receipt = {
        "schema": "hftf_d7_public_real_thor_magni_window_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "dataset_id": "THOR-MAGNI",
        "source_session_id": manifest["source_session_id"],
        "ancestry_group": manifest["ancestry_group"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frame_registry_path": str(frame_path),
        "candidate_index_path": str(candidate_path),
        "counts": {
            "source_rows": manifest["source_row_count"],
            "unique_scene_frames": manifest["unique_scene_frame_count"],
            "frame_rows": len(frame_rows),
            "candidate_windows": len(candidate_rows),
            "parent_events_admitted": 0,
        },
        "access_status": "PUBLIC_SELECTED_MEMBERS_SYNCHRONIZED_WINDOW_INTAKE",
        "status": "PUBLIC_SELECTED_MEMBERS_SYNCHRONIZED_WINDOW_INTAKE",
        "selection": "MODEL_BLIND_UNIFORM_NON_OVERLAPPING_4S_SOURCE_TIME_WINDOWS",
        "event_truth_authority": False,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
        "notes": manifest["notes"],
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--scenario-csv", required=True)
    parser.add_argument("--rgb-path", required=True)
    parser.add_argument("--intrinsics-path")
    parser.add_argument("--device", choices=["PPL", "TB2", "TB3"], default="PPL")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--scene-fps", type=float, default=30.0)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
