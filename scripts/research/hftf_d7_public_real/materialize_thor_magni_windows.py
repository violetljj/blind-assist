#!/usr/bin/env python3
"""Materialize model-blind THOR-MAGNI synchronized frame/window intake.

THOR-MAGNI CSV files merge QTM timestamps, rigid-body tracks, and eye-tracker
scene-frame indices.  This command uses QTM ``Frame``/``Time`` at 100 Hz to
define non-overlapping four-second windows, then uses the source ``SceneFNr``
column only to bind available scene recording frames.  Repeated QTM rows,
missing scene frames, and missing centroids are retained as audit evidence. It
does not inspect RGB content, use a model, infer event buckets, or merge
candidates into the D7 top-level event registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
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
    representative_rows: dict[int, dict[str, Any]] = {}
    scene_groups: dict[int, list[dict[str, Any]]] = {}
    qtm_rows: list[dict[str, Any]] = []
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
            if scene_frame is not None and scene_frame < 0:
                scene_frame = None
            if qtm_frame is None or qtm_time_s is None:
                continue
            if qtm_times and qtm_time_s < qtm_times[-1]:
                raise ContractError(f"non-monotone QTM time in {path}: {qtm_time_s} < {qtm_times[-1]}")
            qtm_times.append(qtm_time_s)
            bodies: dict[str, dict[str, Any]] = {}
            for body, indices in centroid_columns.items():
                values = [_parse_float(raw[index]) if index < len(raw) else None for index in indices]
                bodies[body] = {
                    "role": role_by_body.get(body),
                    "centroid_mm": values,
                }
            row = {
                "scene_frame_index": scene_frame,
                "qtm_frame": qtm_frame,
                "qtm_time_s": qtm_time_s,
                "timestamp_ns": int(round(qtm_time_s * 1_000_000_000)),
                "camera_body": camera_body,
                "camera_body_role": role_by_body.get(camera_body),
                "bodies": bodies,
            }
            qtm_rows.append(row)
            if scene_frame is not None:
                scene_groups.setdefault(scene_frame, []).append(row)
                representative_rows.setdefault(scene_frame, row)
    if not qtm_rows:
        raise ContractError(f"no synchronized {device} SceneFNr rows found in {path}")
    ordered = [representative_rows[index] for index in sorted(representative_rows)]
    qtm_frame_counts = Counter(int(row["qtm_frame"]) for row in qtm_rows)
    qtm_frame_min = min(qtm_frame_counts)
    qtm_frame_max = max(qtm_frame_counts)
    expected_qtm_frames = set(range(qtm_frame_min, qtm_frame_max + 1))
    missing_qtm_frames = sorted(expected_qtm_frames - set(qtm_frame_counts))
    return {
        "metadata": metadata,
        "header": header,
        "scene_column": scene_column,
        "camera_body": camera_body,
        "body_names": body_names,
        "role_by_body": role_by_body,
        "rows": ordered,
        "qtm_rows": qtm_rows,
        "scene_groups": scene_groups,
        "source_row_count": len(qtm_rows),
        "unique_scene_frame_count": len(ordered),
        "scene_frame_missing_row_count": sum(1 for row in qtm_rows if row["scene_frame_index"] is None),
        "qtm_frame_min": qtm_frame_min,
        "qtm_frame_max": qtm_frame_max,
        "qtm_unique_frame_count": len(qtm_frame_counts),
        "qtm_duplicate_frame_rows": sum(max(0, count - 1) for count in qtm_frame_counts.values()),
        "qtm_duplicate_frames": {
            str(frame): count for frame, count in sorted(qtm_frame_counts.items()) if count > 1
        },
        "missing_qtm_frames": missing_qtm_frames,
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
    qtm_fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if window_seconds <= 0 or scene_fps <= 0 or qtm_fps <= 0:
        raise ContractError("window_seconds, scene_fps, and qtm_fps must be positive")
    qtm_frames_per_window = int(round(window_seconds * qtm_fps))
    if qtm_frames_per_window < 2:
        raise ContractError("window must contain at least two nominal QTM frames")
    rows = parsed["rows"]
    qtm_rows = parsed["qtm_rows"]
    by_frame = {int(row["scene_frame_index"]): row for row in rows}
    by_qtm_frame: dict[int, list[dict[str, Any]]] = {}
    for row in qtm_rows:
        by_qtm_frame.setdefault(int(row["qtm_frame"]), []).append(row)
    if len(by_qtm_frame) != parsed["qtm_unique_frame_count"]:
        raise ContractError("QTM frame index audit drifted while building windows")
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
        source_qtm_rows = parsed["scene_groups"][scene_frame]
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
                "qtm_source_row_count": len(source_qtm_rows),
                "qtm_source_frame_indices": [item["qtm_frame"] for item in source_qtm_rows],
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
                "qtm_source_row_count": len(source_qtm_rows),
                "qtm_duplicate_source_row_count": max(0, len(source_qtm_rows) - 1),
                "timestamp_semantics": "SOURCE_SYNCHRONIZED_QTM_TIME",
                "official_sync_contract": PAPER_URL,
                "source_truth_status": "METADATA_GEOMETRY_ONLY",
            },
            "source_license": LICENSE,
            "provider_revision": "zenodo-record-13865754",
            "source_hash": source_hash,
        })
    candidate_rows: list[dict[str, Any]] = []
    complete_scene_windows = 0
    complete_camera_centroid_windows = 0
    qtm_start = int(parsed["qtm_frame_min"])
    qtm_end = int(parsed["qtm_frame_max"])
    for qtm_window_start in range(
        qtm_start,
        qtm_end - qtm_frames_per_window + 2,
        qtm_frames_per_window,
    ):
        qtm_window_end = qtm_window_start + qtm_frames_per_window - 1
        selected_qtm = [
            row
            for row in qtm_rows
            if qtm_window_start <= int(row["qtm_frame"]) <= qtm_window_end
        ]
        selected_qtm_frames = sorted({int(row["qtm_frame"]) for row in selected_qtm})
        expected_qtm_frames = list(range(qtm_window_start, qtm_window_end + 1))
        if selected_qtm_frames != expected_qtm_frames:
            continue
        scene_frames = sorted({
            int(row["scene_frame_index"])
            for row in selected_qtm
            if row["scene_frame_index"] is not None and int(row["scene_frame_index"]) in by_frame
        })
        scene_missing_rows = sum(1 for row in selected_qtm if row["scene_frame_index"] is None)
        scene_internal_gaps = []
        if scene_frames:
            scene_internal_gaps = [
                index
                for index in range(scene_frames[0], scene_frames[-1] + 1)
                if index not in set(scene_frames)
            ]
        scene_complete = scene_missing_rows == 0 and not scene_internal_gaps
        if scene_complete:
            complete_scene_windows += 1
        camera_body = parsed["camera_body"]
        camera_centroid_complete_rows = sum(
            1
            for row in selected_qtm
            if len(row["bodies"].get(camera_body, {}).get("centroid_mm", [])) == 3
            and all(value is not None for value in row["bodies"].get(camera_body, {}).get("centroid_mm", []))
        )
        camera_centroid_complete = camera_centroid_complete_rows == len(selected_qtm)
        if camera_centroid_complete:
            complete_camera_centroid_windows += 1
        first_qtm = by_qtm_frame[qtm_window_start][0]
        last_qtm = by_qtm_frame[qtm_window_end][0]
        start_scene_frame = scene_frames[0] if scene_frames else qtm_window_start
        end_scene_frame = scene_frames[-1] if scene_frames else qtm_window_end
        frame_id_list = [frame_ids[index] for index in scene_frames]
        candidate_id = stable_id(
            "d7cand", "THOR-MAGNI", session_id, qtm_window_start, qtm_window_end, source_hash
        )
        candidate_rows.append({
            "schema": "hftf_d7_public_real_candidate_v1",
            "candidate_id": candidate_id,
            "parent_event_id": stable_id("d7parent", candidate_id),
            "dataset_id": "THOR-MAGNI",
            "source_session_id": session_id,
            "ancestry_group": ancestry_group,
            "source_id": file_id,
            "segment_index": 0,
            "start_frame_index": start_scene_frame,
            "end_frame_index": end_scene_frame,
            "start_timestamp_ns": first_qtm["timestamp_ns"],
            "end_timestamp_ns": last_qtm["timestamp_ns"],
            "timestamp_semantics": "SOURCE_SYNCHRONIZED_QTM_TIME",
            "frame_ids": frame_id_list,
            "frame_count": len(frame_id_list),
            "candidate_selection": "MODEL_BLIND_UNIFORM_QTM_100HZ_SOURCE_COVERAGE",
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
                "qtm_window_start_frame": qtm_window_start,
                "qtm_window_end_frame": qtm_window_end,
                "qtm_window_row_count": len(selected_qtm),
                "qtm_window_unique_frame_count": len(selected_qtm_frames),
                "qtm_window_duplicate_row_count": len(selected_qtm) - len(selected_qtm_frames),
                "scene_frame_count": len(scene_frames),
                "scene_frame_missing_row_count": scene_missing_rows,
                "scene_frame_internal_gaps": scene_internal_gaps,
                "scene_frame_complete": scene_complete,
                "camera_centroid_complete_rows": camera_centroid_complete_rows,
                "camera_centroid_complete": camera_centroid_complete,
                "window_duration_seconds": (last_qtm["timestamp_ns"] - first_qtm["timestamp_ns"]) / 1_000_000_000,
                "source_sync_contract": PAPER_URL,
                "candidate_truth_boundary": "INTAKE_ONLY_UNREVIEWED",
            },
        })
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
        "scene_frame_missing_row_count": parsed["scene_frame_missing_row_count"],
        "qtm_frame_min": parsed["qtm_frame_min"],
        "qtm_frame_max": parsed["qtm_frame_max"],
        "qtm_unique_frame_count": parsed["qtm_unique_frame_count"],
        "qtm_duplicate_frame_rows": parsed["qtm_duplicate_frame_rows"],
        "qtm_duplicate_frames": parsed["qtm_duplicate_frames"],
        "missing_qtm_frames": parsed["missing_qtm_frames"],
        "window_seconds": window_seconds,
        "qtm_fps": qtm_fps,
        "qtm_frames_per_window": qtm_frames_per_window,
        "nominal_scene_fps": scene_fps,
        "complete_scene_windows": complete_scene_windows,
        "complete_camera_centroid_windows": complete_camera_centroid_windows,
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
            "QTM Frame/Time defines non-overlapping four-second windows; SceneFNr only binds available source video frames.",
            "Repeated QTM Frame rows are retained in the audit and are not silently deduplicated.",
            "N/A SceneFNr rows, scene-frame gaps, and missing camera centroids are preserved; no interpolation was applied.",
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
        qtm_fps=args.qtm_fps,
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
    parser.add_argument("--qtm-fps", type=float, default=100.0)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
