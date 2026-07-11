#!/usr/bin/env python3
"""Screen exact SANPO 50-frame windows from source masks before RGB download.

This is the second stage after sparse discovery.  It fetches only public
panoptic masks for requested source-frame starts, applies the same geometry
gate as an on-disk draft, and records the exact 10-FPS source frames.  Passing
windows are still only RGB-download candidates; they require privacy review,
model review, and dense annotation before any v3 dataset use.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sanpo_sequence_evalset import (  # noqa: E402
    DEFAULT_CAMERA,
    DEFAULT_LENS,
    GCS_PREFIX,
    fetch_json,
    frame_number,
    get_gcs_object,
    list_gcs_objects,
    media_url,
    resample_indices,
)
from select_sanpo_sequence_by_geometry import (  # noqa: E402
    CENTER_HAZARD_IDS,
    PROFILE_TARGETS,
    components_for_mask,
    summarize_frame_evidence,
)


def source_fps_for(session_id: str, camera: str) -> float:
    description = fetch_json(media_url(f"{GCS_PREFIX}/sanpo-real/{session_id}/description.json"))
    locations = list(description.get("session_camera_location", []))
    if camera not in locations:
        raise ValueError(f"{session_id}: camera {camera!r} is absent")
    return float(description["session_camera_details"][locations.index(camera)]["fps"])


def mask_array(url: str) -> np.ndarray:
    with urlopen(url, timeout=60) as response:
        with Image.open(io.BytesIO(response.read())) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)


def frame_evidence(frame_index: int, source_frame_index: int, rgb: np.ndarray, profile: str) -> dict[str, Any]:
    components, path = components_for_mask(rgb)
    target_ids = PROFILE_TARGETS[profile]
    targets = [component for class_id in target_ids for component in components.get(class_id, [])]
    hazards = [component for class_id in CENTER_HAZARD_IDS for component in components.get(class_id, [])]
    central_targets = [item for item in targets if item["corridor_target_ratio"] >= 0.12 and item["bottom_ratio"] >= 0.45]
    central_hazards = [item for item in hazards if item["corridor_target_ratio"] >= 0.12 and item["bottom_ratio"] >= 0.45]
    lateral_targets = [
        item for item in targets
        if item["corridor_target_ratio"] <= 0.01
        and (item["center_x_ratio"] <= 0.35 or item["center_x_ratio"] >= 0.65)
        and item["bottom_ratio"] >= 0.35
    ]
    return {
        "frame_index": frame_index,
        "source_frame_index": source_frame_index,
        "path_geometry": path,
        "targets": targets,
        "target_center_intrusion": bool(central_targets),
        "target_clean_lateral": bool(lateral_targets),
        "any_center_hazard": bool(central_hazards),
        "best_target": max(targets, key=lambda item: item["corridor_blocking_ratio"], default=None),
    }


def screen_window(session_id: str, camera: str, lens: str, start_frame: int, profile: str, target_fps: float, retries: int) -> dict[str, Any]:
    source_fps = source_fps_for(session_id, camera)
    prefix = f"{GCS_PREFIX}/sanpo-real/{session_id}/{camera}/{lens}/segmentation_masks/"
    objects = {frame_number(item["name"]): item for item in list_gcs_objects(prefix, retries) if item["name"].endswith(".png")}
    selected = resample_indices(objects, source_fps, target_fps, start_frame, 50)
    if len(selected) != 50:
        return {
            "session_id": session_id,
            "camera": camera,
            "lens": lens,
            "start_frame": start_frame,
            "profile": profile,
            "decision": "reject",
            "rejection_reasons": ["insufficient_aligned_source_masks_for_50_frames"],
            "selected_source_frames": selected,
        }
    frames = []
    for index, source_frame in enumerate(selected):
        item = objects[source_frame]
        frames.append(frame_evidence(index, source_frame, mask_array(media_url(item["name"], item.get("generation"))), profile))
    result = summarize_frame_evidence(frames, profile, f"sanpo_{session_id}_{camera}_{lens}_{start_frame:06d}_{int(target_fps)}fps")
    result.update({
        "screen_kind": "remote_mask_only",
        "session_id": session_id,
        "camera": camera,
        "lens": lens,
        "start_frame": start_frame,
        "source_fps": source_fps,
        "target_fps": target_fps,
        "selected_source_frames": selected,
        "next_gate": "download RGB+source masks into a fresh draft, rerun the on-disk geometry selector, then model review",
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--start-frame", type=int, action="append", required=True, help="Repeat for each exact source-frame window to screen.")
    parser.add_argument("--profile", choices=sorted(PROFILE_TARGETS), default="lateral_pedestrian_or_ebike")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, choices=("camera_chest", "camera_head"))
    parser.add_argument("--lens", default=DEFAULT_LENS, choices=("left",))
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = [screen_window(args.session_id, args.camera, args.lens, start, args.profile, args.target_fps, args.retries) for start in args.start_frame]
    payload = {
        "format": "blindassist_sanpo_remote_mask_window_screen_v1",
        "profile": args.profile,
        "session_id": args.session_id,
        "rgb_downloaded": False,
        "windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    accepted = sum(item.get("decision") == "accept_for_model_review" for item in windows)
    print(json.dumps({"windows": len(windows), "accepted": accepted, "output": str(args.output)}, ensure_ascii=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
