#!/usr/bin/env python3
"""Discover public SANPO-Real candidates with auditable mask geometry.

This sparse pass never downloads RGB.  It is intentionally a shortlist only:
every shortlisted session still needs a downloaded, exact 50-frame geometry
gate and model review before it can enter the dense-annotation queue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
from PIL import Image

from select_sanpo_sequence_by_geometry import (
    CENTER_HAZARD_IDS,
    PROFILE_TARGETS,
    components_for_mask,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sanpo_sequence_evalset import (  # noqa: E402
    GCS_PREFIX,
    DEFAULT_CAMERA,
    DEFAULT_LENS,
    GCS_API,
    fetch_json,
    fetch_text,
    frame_number,
    get_gcs_object,
    list_gcs_objects,
    media_url,
    resample_indices,
)
from screen_sanpo_mask_windows import mask_array, source_fps_for  # noqa: E402


CAMERAS = ("camera_chest", "camera_head")
AUTO_CAMERA = "auto"
LABELS = {
    "curb": 2,
    "pedestrian": 12,
    "rider": 13,
    "stairs": 15,
    "inaccessible_surface": 18,
    "generic_obstacle": 20,
    "vehicle": 21,
    "pole": 24,
}


def mask_geometry(url: str, retries: int = 3) -> tuple[dict[int, list[dict]], dict[str, float]]:
    """Read one remote mask through the same retry contract as window screening."""
    return components_for_mask(mask_array(url, retries))


def longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def sparse_profile_evidence(components: dict[int, list[dict]], path: dict[str, float]) -> dict[str, Any]:
    """Classify one sparse mask without treating mere label presence as evidence."""
    central_hazards = [
        component for class_id in CENTER_HAZARD_IDS for component in components.get(class_id, [])
        if component["corridor_target_ratio"] >= 0.12 and component["bottom_ratio"] >= 0.45
    ]
    center_targets = [
        component for class_id in PROFILE_TARGETS["center_obstacle"] for component in components.get(class_id, [])
        if component["corridor_target_ratio"] >= 0.12 and component["bottom_ratio"] >= 0.45
    ]
    lateral_targets = [
        component for class_id in PROFILE_TARGETS["lateral_pedestrian_or_ebike"] for component in components.get(class_id, [])
        if component["corridor_target_ratio"] <= 0.01
        and (component["center_x_ratio"] <= 0.35 or component["center_x_ratio"] >= 0.65)
        and component["bottom_ratio"] >= 0.35
    ]
    center_lateral_targets = [
        component for class_id in PROFILE_TARGETS["lateral_pedestrian_or_ebike"] for component in components.get(class_id, [])
        if component["corridor_target_ratio"] >= 0.12 and component["bottom_ratio"] >= 0.45
    ]
    boundary_targets = [
        component for class_id in PROFILE_TARGETS["step_curb"] for component in components.get(class_id, [])
        if component["bottom_ratio"] >= 0.45
    ]
    path_ok = path["walkable_corridor_ratio"] >= 0.18
    return {
        "path_geometry_usable": path_ok,
        "walkable_corridor_ratio": path["walkable_corridor_ratio"],
        "center_obstacle": bool(center_targets) and path_ok,
        "lateral_pedestrian_or_ebike": bool(lateral_targets) and not center_lateral_targets and not central_hazards and path_ok,
        # This is deliberately a broad boundary candidate, not a semantic claim
        # that the sequence is a step alert.  The exact 50-frame gate and RGB
        # review later distinguish a parallel curb from a step/curb transition.
        "step_curb": bool(boundary_targets) and path_ok,
        "has_center_hazard": bool(central_hazards),
        "has_center_lateral_target": bool(center_lateral_targets),
        "best_center_target": max(center_targets, key=lambda item: item["corridor_blocking_ratio"], default=None),
        "best_lateral_target": max(lateral_targets, key=lambda item: item["bottom_ratio"], default=None),
        "best_boundary_target": max(boundary_targets, key=lambda item: item["bottom_ratio"], default=None),
    }


def session_ids(split: str) -> list[str]:
    name = f"{GCS_PREFIX}/sanpo-real/splits/{split}_session_ids.txt"
    item = get_gcs_object(name)
    return [line.strip() for line in fetch_text(media_url(name, item.get("generation"))).splitlines() if line.strip()]


def first_mask_page(prefix: str) -> list[dict]:
    """One GCS page is sufficient for discovery; full inventory is deferred to download."""
    payload = fetch_json(f"{GCS_API}?{urlencode({'prefix': prefix, 'maxResults': 1000})}")
    return [item for item in payload.get("items", []) if item["name"].endswith(".png")]


def select_mask_view(session_id: str, requested_camera: str, minimum_frame_count: int) -> tuple[str, list[dict]] | None:
    """Select one official left view by actual mask inventory, preferring chest.

    A SANPO native session remains the split atom even when it has two cameras.
    This helper chooses exactly one view and never substitutes right imagery.
    """
    cameras = CAMERAS if requested_camera == AUTO_CAMERA else (requested_camera,)
    for camera in cameras:
        prefix = f"{GCS_PREFIX}/sanpo-real/{session_id}/{camera}/{DEFAULT_LENS}/segmentation_masks/"
        items = first_mask_page(prefix)
        if len(items) >= minimum_frame_count:
            return camera, items
    return None


def summarize_local_lateral_prefilter(
    frames: list[dict[str, Any]], expected_frame_count: int,
    minimum_target_frames: int, minimum_target_run: int, minimum_path_frames: int,
) -> dict[str, Any]:
    """Screen a short, source-mask-only lateral window before the exact 50-frame gate.

    This is a cost-control rejection filter, not an acceptance gate: a pass only
    authorizes the existing 50-frame screen, whose contract remains authoritative.
    """
    if len(frames) != expected_frame_count:
        raise ValueError(f"local prefilter needs {expected_frame_count} frames, got {len(frames)}")
    target = [bool(item["profiles"]["lateral_pedestrian_or_ebike"]) for item in frames]
    path = [bool(item["profiles"]["path_geometry_usable"]) for item in frames]
    center_hazard = [bool(item["profiles"]["has_center_hazard"]) for item in frames]
    center_lateral = [bool(item["profiles"]["has_center_lateral_target"]) for item in frames]
    reasons: list[str] = []
    if sum(target) < minimum_target_frames:
        reasons.append("local_lateral_target_frames_below_minimum")
    if longest_run(target) < minimum_target_run:
        reasons.append("local_lateral_target_run_below_minimum")
    if any(center_hazard):
        reasons.append("local_center_hazard_contamination")
    if any(center_lateral):
        reasons.append("local_center_lateral_target_contamination")
    if sum(path) < minimum_path_frames:
        reasons.append("local_path_geometry_frames_below_minimum")
    return {
        "format": "blindassist_sanpo_local_lateral_prefilter_v1",
        "decision": "pass_for_exact_50_frame_gate" if not reasons else "reject",
        "rejection_reasons": reasons,
        "frame_count": len(frames),
        "summary": {
            "lateral_target_frame_count": sum(target),
            "lateral_target_longest_run": longest_run(target),
            "path_geometry_usable_frame_count": sum(path),
            "center_hazard_frame_count": sum(center_hazard),
            "center_lateral_target_frame_count": sum(center_lateral),
        },
        "thresholds": {
            "minimum_target_frames": minimum_target_frames,
            "minimum_target_run": minimum_target_run,
            "minimum_path_frames": minimum_path_frames,
        },
        "frames": frames,
        "important_limit": "pass is not an acceptance; the existing exact 50-frame geometry gate remains mandatory",
    }


def local_lateral_prefilter(
    session_id: str, camera: str, lens: str, start_frame: int,
    objects: dict[int, dict[str, Any]], target_fps: float, frame_count: int,
    retries: int, minimum_target_frames: int, minimum_target_run: int, minimum_path_frames: int,
) -> dict[str, Any]:
    source_fps = source_fps_for(session_id, camera, retries)
    selected = resample_indices(objects, source_fps, target_fps, start_frame, frame_count)
    if len(selected) != frame_count:
        return {
            "format": "blindassist_sanpo_local_lateral_prefilter_v1",
            "decision": "reject",
            "rejection_reasons": ["insufficient_aligned_source_masks_for_local_prefilter"],
            "frame_count": len(selected),
            "selected_source_frames": selected,
            "thresholds": {
                "minimum_target_frames": minimum_target_frames,
                "minimum_target_run": minimum_target_run,
                "minimum_path_frames": minimum_path_frames,
            },
        }
    frames: list[dict[str, Any]] = []
    for source_frame in selected:
        components, path = mask_geometry(media_url(objects[source_frame]["name"], objects[source_frame].get("generation")), retries)
        frames.append({
            "source_frame": source_frame,
            "profiles": sparse_profile_evidence(components, path),
        })
    result = summarize_local_lateral_prefilter(
        frames, frame_count, minimum_target_frames, minimum_target_run, minimum_path_frames,
    )
    result.update({
        "session_id": session_id,
        "camera": camera,
        "lens": lens,
        "start_frame": start_frame,
        "source_fps": source_fps,
        "target_fps": target_fps,
        "selected_source_frames": selected,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument(
        "--start-session-index", type=int, default=0,
        help="Zero-based offset in the sorted official split list; use it to resume disjoint scans.",
    )
    parser.add_argument("--max-sessions", type=int, default=0, help="0 scans every session in the selected split")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--minimum-hits", type=int, default=3)
    parser.add_argument("--local-lateral-frame-count", type=int, default=0,
                        help="0 disables; positive value prefilters each sparse lateral candidate before its exact 50-frame gate.")
    parser.add_argument("--local-lateral-min-target-frames", type=int, default=8)
    parser.add_argument("--local-lateral-min-target-run", type=int, default=8)
    parser.add_argument("--local-lateral-min-path-frames", type=int, default=13)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--camera", choices=(AUTO_CAMERA, *CAMERAS), default=AUTO_CAMERA,
        help="auto prefers chest then falls back to head using public left-mask inventory",
    )
    parser.add_argument("--labels", nargs="+", choices=sorted(LABELS), default=sorted(LABELS))
    parser.add_argument(
        "--profiles", nargs="+", choices=sorted(PROFILE_TARGETS),
        default=("center_obstacle", "lateral_pedestrian_or_ebike", "step_curb"),
        help="Restrict sparse discovery to selected profiles; all still require the exact 50-frame gate.",
    )
    args = parser.parse_args()
    if (
        args.sample_count <= 0 or args.minimum_hits <= 0 or args.start_session_index < 0
        or args.local_lateral_frame_count < 0 or args.target_fps <= 0 or args.retries <= 0
    ):
        raise SystemExit("sample/minimum/retry values must be positive and --start-session-index non-negative")
    if args.local_lateral_frame_count and any(
        value <= 0 or value > args.local_lateral_frame_count for value in (
            args.local_lateral_min_target_frames,
            args.local_lateral_min_target_run,
            args.local_lateral_min_path_frames,
        )
    ):
        raise SystemExit("local lateral thresholds must be positive and no larger than --local-lateral-frame-count")

    splits = ("train", "test") if args.split == "all" else (args.split,)
    wanted_by_name = {name: LABELS[name] for name in args.labels}
    records: list[dict] = []
    failures: list[dict] = []
    local_lateral_rejections: list[dict[str, Any]] = []
    scan_coverage: list[dict[str, Any]] = []
    for split in splits:
        all_ids = session_ids(split)
        ids = all_ids[args.start_session_index:]
        if args.max_sessions:
            ids = ids[: args.max_sessions]
        attempted = 0
        insufficient_mask_pages = 0
        selected_camera_counts = {camera: 0 for camera in CAMERAS}
        for index, session_id in enumerate(ids, start=1):
            attempted += 1
            try:
                selection = select_mask_view(session_id, args.camera, args.sample_count)
                if selection is None:
                    insufficient_mask_pages += 1
                    continue
                camera, items = selection
                selected_camera_counts[camera] += 1
                numbers = sorted(frame_number(item["name"]) for item in items)
                sampled = [numbers[round(i * (len(numbers) - 1) / (args.sample_count - 1))] for i in range(args.sample_count)]
                by_number = {frame_number(item["name"]): item for item in items}
                frame_evidence: list[dict] = []
                for frame in sampled:
                    item = by_number[frame]
                    components, path = mask_geometry(media_url(item["name"], item.get("generation")), args.retries)
                    profile = sparse_profile_evidence(components, path)
                    frame_evidence.append({"source_frame": frame, "profiles": profile})
                for profile_name in args.profiles:
                    matches = [item for item in frame_evidence if item["profiles"][profile_name]]
                    match_bools = [item["profiles"][profile_name] for item in frame_evidence]
                    if len(matches) >= args.minimum_hits and longest_run(match_bools) >= 2:
                        frames = [item["source_frame"] for item in matches]
                        record = {
                            "session_id": session_id,
                            "official_split": split,
                            "camera": camera,
                            "lens": DEFAULT_LENS,
                            "selection_profile": profile_name,
                            "sampled_source_frames": sampled,
                            "geometry_matching_source_frames": frames,
                            "sparse_longest_consecutive_sample_run": longest_run(match_bools),
                            "sparse_frame_evidence": frame_evidence,
                            "recommended_start_frame": max(0, frames[len(frames) // 2] - 15),
                            "next_gate": "run the exact remote 50-frame geometry gate before any RGB download",
                            "license": "Creative Commons Attribution 4.0 International",
                            "dataset_page": "https://google-research-datasets.github.io/sanpo_dataset/",
                        }
                        if profile_name == "lateral_pedestrian_or_ebike" and args.local_lateral_frame_count:
                            prefilter = local_lateral_prefilter(
                                session_id, camera, DEFAULT_LENS, int(record["recommended_start_frame"]),
                                by_number, args.target_fps, args.local_lateral_frame_count, args.retries,
                                args.local_lateral_min_target_frames, args.local_lateral_min_target_run,
                                args.local_lateral_min_path_frames,
                            )
                            if prefilter["decision"] != "pass_for_exact_50_frame_gate":
                                local_lateral_rejections.append({
                                    "session_id": session_id,
                                    "official_split": split,
                                    "camera": camera,
                                    "lens": DEFAULT_LENS,
                                    "start_frame": record["recommended_start_frame"],
                                    "prefilter": prefilter,
                                })
                                continue
                            record["local_lateral_prefilter"] = prefilter
                        records.append(record)
                print(f"scanned={split}:{index}/{len(ids)} records={len(records)}", flush=True)
            except Exception as error:  # network/data errors are reported, not hidden
                failures.append({"session_id": session_id, "official_split": split, "error": f"{type(error).__name__}: {error}"})
        scan_coverage.append({
            "official_split": split,
            "available_session_count": len(all_ids),
            "requested_start_session_index": args.start_session_index,
            "requested_max_sessions": args.max_sessions,
            "selected_session_count": len(ids),
            "attempted_session_count": attempted,
            "insufficient_first_mask_page_count": insufficient_mask_pages,
            "selected_camera_counts": selected_camera_counts,
            "network_or_data_failure_count": sum(1 for item in failures if item["official_split"] == split),
        })
    payload = {
        "source": "SANPO-Real v0 public GCS metadata and sparse segmentation-mask scan",
        "license": "Creative Commons Attribution 4.0 International",
        "camera_selection": args.camera,
        "lens": DEFAULT_LENS,
        "sample_count": args.sample_count,
        "minimum_hits": args.minimum_hits,
        "start_session_index": args.start_session_index,
        "scan_coverage": scan_coverage,
        "local_lateral_prefilter": {
            "enabled": bool(args.local_lateral_frame_count),
            "frame_count": args.local_lateral_frame_count,
            "target_fps": args.target_fps,
            "minimum_target_frames": args.local_lateral_min_target_frames,
            "minimum_target_run": args.local_lateral_min_target_run,
            "minimum_path_frames": args.local_lateral_min_path_frames,
            "important_limit": "only a rejection filter; an exact 50-frame screen remains mandatory after a pass",
        },
        "labels": wanted_by_name,
        "profiles": list(args.profiles),
        "selection_method": {
            "version": "corridor-path-persistence-v1",
            "center_obstacle": "sparse source-mask target intrudes into the near-field center corridor; path has walkable support; repeated in >=minimum-hits sampled frames",
            "lateral_pedestrian_or_ebike": "pedestrian/rider stays outside the corridor, path has walkable support, and no other center hazard occurs in sampled frames",
            "step_curb": "curb/stairs reaches the lower field with walkable corridor support; sparse candidate only, because RGB review assigns parallel-boundary versus step/curb semantics",
            "important_limit": "sparse results are not a 50-frame acceptance. The exact draft selector is mandatory before model review.",
        },
        "candidates": records,
        "local_lateral_prefilter_rejections": local_lateral_rejections,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"candidates={len(records)} failures={len(failures)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
