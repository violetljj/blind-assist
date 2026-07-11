#!/usr/bin/env python3
"""Verify a 50-frame SANPO draft using corridor, persistence, and path geometry.

This is a *selection* gate, not a safety label.  It reads only source masks in
an already-downloaded draft and records why a sequence may be sent to model
review.  It never creates semantic masks, risk labels, or training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FORMAT = "blindassist_sanpo_sequence_geometry_selection_v1"
WALKABLE_IDS = {1, 3, 5, 6, 17}  # road, sidewalk, crosswalk, paved trail, other walkable
CENTER_HAZARD_IDS = {15, 18, 20, 21, 24}  # stairs, inaccessible, obstacle, vehicle, pole
PROFILE_TARGETS = {
    "step_curb": {2, 15},
    "center_obstacle": {18, 20, 24},
    "lateral_pedestrian_or_ebike": {12, 13},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corridor_mask(height: int, width: int) -> np.ndarray:
    """A conservative near-field walking corridor trapezoid.

    The upper corridor is deliberately narrow (45--55% width) and fans out to
    25--75% at the image bottom.  It is a reproducible proxy for selection;
    model review and dense annotation remain the authority for the real path.
    """
    y = np.arange(height, dtype=np.float32)[:, None]
    top = height * 0.40
    progress = np.clip((y - top) / max(1.0, height - top), 0.0, 1.0)
    left = width * (0.45 - 0.20 * progress)
    right = width * (0.55 + 0.20 * progress)
    x = np.arange(width, dtype=np.float32)[None, :]
    return (y >= top) & (x >= left) & (x < right)


def components_for_mask(rgb: np.ndarray) -> tuple[dict[int, list[dict[str, Any]]], dict[str, float]]:
    """Return panoptic components plus corridor/path metrics for one source mask."""
    height, width = rgb.shape[:2]
    corridor = corridor_mask(height, width)
    class_ids = rgb[:, :, 0]
    corridor_pixels = int(corridor.sum())
    walkable_ratio = float(np.logical_and(corridor, np.isin(class_ids, list(WALKABLE_IDS))).sum() / max(1, corridor_pixels))
    result: dict[int, list[dict[str, Any]]] = {}
    # A source mask can contain hundreds of irrelevant panoptic colors.  This
    # gate needs only targets and possible center contaminants; enumerating all
    # colors turns an otherwise linear 50-frame check into a multi-minute scan.
    interesting_ids = set().union(*PROFILE_TARGETS.values()) | CENTER_HAZARD_IDS
    instance_ids = rgb[:, :, 1].astype(np.uint32) * 256 + rgb[:, :, 2].astype(np.uint32)
    flat_classes = class_ids.reshape(-1)
    relevant_positions = np.flatnonzero(np.isin(flat_classes, list(interesting_ids)))
    if not relevant_positions.size:
        return result, {
            "walkable_corridor_ratio": round(walkable_ratio, 4),
            "corridor_width_top_ratio": 0.10,
            "corridor_width_bottom_ratio": 0.50,
        }
    relevant_classes = flat_classes[relevant_positions].astype(np.uint32)
    relevant_instances = instance_ids.reshape(-1)[relevant_positions]
    # Group only relevant pixels by (class, instance).  Do not allocate one
    # full-resolution boolean image per instance; urban SANPO masks often have
    # hundreds of tiny obstacle instances in a single frame.
    keys = relevant_classes * np.uint32(65536) + relevant_instances
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_positions = relevant_positions[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1, len(sorted_keys)]
    corridor_flat = corridor.reshape(-1)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            key = int(sorted_keys[start])
            class_id, instance_id = divmod(key, 65536)
            positions = sorted_positions[start:end]
            ys, xs = divmod(positions, width)
            pixel_count = int(positions.size)
            corridor_pixels_for_component = int(corridor_flat[positions].sum())
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            result.setdefault(class_id, []).append({
            "instance_id": int(instance_id),
            "bbox_xyxy": [x1, y1, x2, y2],
            "pixel_count": int(pixel_count),
            "center_x_ratio": round(float(xs.mean() / width), 4),
            "bottom_ratio": round(y2 / height, 4),
            "corridor_target_ratio": round(corridor_pixels_for_component / max(1, int(pixel_count)), 4),
            "corridor_blocking_ratio": round(corridor_pixels_for_component / max(1, corridor_pixels), 5),
        })
    return result, {
        "walkable_corridor_ratio": round(walkable_ratio, 4),
        "corridor_width_top_ratio": 0.10,
        "corridor_width_bottom_ratio": 0.50,
    }


def longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def frame_evidence(row: dict[str, Any], root: Path, profile: str) -> dict[str, Any]:
    target_ids = PROFILE_TARGETS[profile]
    mask = root / "source_masks" / "test" / f"{row['id']}.png"
    if not mask.is_file():
        # Draft manifests always use this convention today, but fail loudly if
        # a future builder changes it rather than silently relaxing the gate.
        raise FileNotFoundError(f"source mask missing for {row['id']}: {mask}")
    with Image.open(mask) as image:
        components, path = components_for_mask(np.asarray(image.convert("RGB"), dtype=np.uint8))
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
        "frame_index": int(row["frame_index"]),
        "source_frame_index": int(row["source_frame_index"]),
        "path_geometry": path,
        "targets": targets,
        "target_center_intrusion": bool(central_targets),
        "target_clean_lateral": bool(lateral_targets),
        "any_center_hazard": bool(central_hazards),
        "best_target": max(targets, key=lambda item: item["corridor_blocking_ratio"], default=None),
    }


def evaluate(rows: list[dict[str, Any]], root: Path, profile: str) -> dict[str, Any]:
    if len(rows) != 50:
        raise ValueError(f"selection requires exactly 50 frames, got {len(rows)}")
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    if [int(row["frame_index"]) for row in ordered] != list(range(50)):
        raise ValueError("selection requires contiguous frame_index 0..49")
    sequence_ids = {str(row["sequence_id"]) for row in ordered}
    if len(sequence_ids) != 1:
        raise ValueError("selection requires exactly one sequence_id")
    frames = [frame_evidence(row, root, profile) for row in ordered]
    path_ok = [item["path_geometry"]["walkable_corridor_ratio"] >= 0.18 for item in frames]
    if profile == "center_obstacle":
        target_present = [item["target_center_intrusion"] for item in frames]
        reasons = []
        if sum(target_present) < 20:
            reasons.append("center_intrusion_frames_below_20")
        if longest_run(target_present) < 8:
            reasons.append("center_intrusion_longest_run_below_8")
    elif profile == "lateral_pedestrian_or_ebike":
        target_present = [item["target_clean_lateral"] for item in frames]
        disqualifying_center_hazard = [item["any_center_hazard"] for item in frames]
        disqualifying_center_target = [item["target_center_intrusion"] for item in frames]
        reasons = []
        if sum(target_present) < 20:
            reasons.append("clean_lateral_target_frames_below_20")
        if longest_run(target_present) < 8:
            reasons.append("clean_lateral_longest_run_below_8")
        if any(disqualifying_center_hazard):
            reasons.append("other_center_hazard_contaminates_negative")
        if any(disqualifying_center_target):
            reasons.append("center_target_contaminates_lateral_negative")
    else:  # step_curb: route boundary/ramp evidence, never a free-standing alert label.
        target_present = [any(item["bottom_ratio"] >= 0.45 for item in frame["targets"]) for frame in frames]
        reasons = []
        # A ramp/curb transition can legitimately pass the near field in under
        # one second at 10 FPS.  It still needs temporal support, but the
        # center-obstacle persistence threshold would incorrectly discard this
        # no-alert boundary case.
        if sum(target_present) < 5:
            reasons.append("step_or_curb_frames_below_5")
        if longest_run(target_present) < 3:
            reasons.append("step_or_curb_longest_run_below_3")
    if sum(path_ok) < 40:
        reasons.append("walkable_path_geometry_frames_below_40")
    best = [item["best_target"] for item in frames if item["best_target"]]
    return {
        "format": FORMAT,
        "draft_manifest_sha256": None,  # filled by main after the path is known
        "sequence_id": next(iter(sequence_ids)),
        "frame_count": 50,
        "profile": profile,
        "decision": "accept_for_model_review" if not reasons else "reject",
        "rejection_reasons": reasons,
        "summary": {
            "target_qualified_frame_count": sum(target_present),
            "target_longest_consecutive_run": longest_run(target_present),
            "walkable_path_geometry_frame_count": sum(path_ok),
            "median_walkable_corridor_ratio": round(float(np.median([item["path_geometry"]["walkable_corridor_ratio"] for item in frames])), 4),
            "max_target_corridor_blocking_ratio": max((item["corridor_blocking_ratio"] for item in best), default=0.0),
        },
        "frames": frames,
        "selection_contract": {
            "center_obstacle": "target must intrude into the conservative center corridor for >=20 frames with one >=8-frame run; usable walkable-path geometry >=40 frames",
            "lateral_pedestrian_or_ebike": "pedestrian/rider must stay lateral for >=20 frames with one >=8-frame run; no other center hazard and usable path geometry >=40 frames",
            "step_curb": "curb/stairs must persist for >=5 frames with one >=3-frame run and usable path geometry >=40 frames; the shorter transition allowance is only for boundary/ramp no-alert cases, not obstacles",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_TARGETS), required=True)
    parser.add_argument("--output", type=Path, help="Defaults to qa/selection_evidence.json")
    args = parser.parse_args()
    root = args.draft_root.resolve()
    manifest = root / "manifest.draft.jsonl"
    rows = load_jsonl(manifest)
    result = evaluate(rows, root, args.profile)
    result["draft_manifest_sha256"] = sha256(manifest)
    output = (args.output or root / "qa" / "selection_evidence.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": result["decision"] == "accept_for_model_review", "output": str(output), "summary": result["summary"]}, ensure_ascii=False))
    return 0 if result["decision"] == "accept_for_model_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
