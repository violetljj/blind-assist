"""Qualify the frozen ABotN V0 action graph and rendered pixels before provider use."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


PUBLIC_SCHEMA = "blindassist_abotn_v0_action_graph_public_v0"
PIXEL_TERMINALS = {
    "blindassist_abotn_webgl_action_graph_pixels_v0": "ABOTN_WEBGL_ACTION_GRAPH_PIXELS_PASS",
    "blindassist_abotn_official_action_graph_pixels_v0": "ABOTN_OFFICIAL_ACTION_GRAPH_PIXELS_PASS",
}
MIN_DIRECTION_SHIFT_PX = 40.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _orb_horizontal_shift(first: Path, second: Path) -> tuple[float, int]:
    import cv2
    import numpy as np

    detector = cv2.ORB_create(nfeatures=6000)
    first_keypoints, first_descriptors = detector.detectAndCompute(cv2.imread(str(first), 0), None)
    second_keypoints, second_descriptors = detector.detectAndCompute(cv2.imread(str(second), 0), None)
    if first_descriptors is None or second_descriptors is None:
        raise ValueError("ORB direction check has no descriptors")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(first_descriptors, second_descriptors)
    retained = sorted(matches, key=lambda item: item.distance)[: min(500, len(matches))]
    if len(retained) < 50:
        raise ValueError("ORB direction check has insufficient matches")
    shifts = [
        second_keypoints[item.trainIdx].pt[0] - first_keypoints[item.queryIdx].pt[0]
        for item in retained
    ]
    return float(np.median(shifts)), len(retained)


def qualify(
    *, public_graph_path: Path, private_truth_path: Path, freeze_path: Path,
    pixel_receipt_path: Path, output_path: Path,
) -> dict[str, Any]:
    public = json.loads(public_graph_path.read_text(encoding="utf-8"))
    private = json.loads(private_truth_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    pixels = json.loads(pixel_receipt_path.read_text(encoding="utf-8"))
    pixel_root = pixel_receipt_path.parent
    roster_path = pixel_root / "roster.json"
    manifest_path = pixel_root / "manifest.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if public.get("schema_version") != PUBLIC_SCHEMA or public.get("private_truth_access") is not False:
        raise ValueError("public action graph is not eligible")
    if freeze.get("terminal") != "ABOTN_V0_ACTION_GRAPH_FROZEN_ELIGIBLE":
        raise ValueError("action graph freeze receipt is not eligible")
    if PIXEL_TERMINALS.get(pixels.get("schema_version")) != pixels.get("terminal"):
        raise ValueError("action graph pixel receipt is not eligible")
    if pixels.get("public_graph_sha256") != _sha256(public_graph_path):
        raise ValueError("pixel/public graph binding drift")
    if pixels.get("roster_sha256") != _sha256(roster_path) or manifest.get("roster_sha256") != _sha256(roster_path):
        raise ValueError("pixel roster binding drift")
    if manifest.get("terminal_receipt_sha256") != _sha256(pixel_receipt_path):
        raise ValueError("pixel terminal receipt binding drift")

    graph_nodes = public.get("nodes", [])
    roster_rows = roster.get("observations", [])
    frames = pixels.get("frames", [])
    graph_ids = [row["node_id"] for row in graph_nodes]
    if graph_ids != [row["observation_id"] for row in roster_rows] or graph_ids != [row["observation_id"] for row in frames]:
        raise ValueError("graph/roster/frame order drift")
    graph_paths = [row["rendered_frame_path"] for row in graph_nodes]
    if graph_paths != [row["output_path"] for row in roster_rows] or graph_paths != [row["path"] for row in frames]:
        raise ValueError("graph/roster/frame path drift")
    for frame in frames:
        path = pixel_root / frame["path"]
        if not path.is_file() or _sha256(path) != frame["sha256"]:
            raise ValueError(f"pixel payload drift: {frame['observation_id']}")
    node_ids = set(graph_ids)
    if any(target not in node_ids for node in graph_nodes for target in node["actions"].values()):
        raise ValueError("action graph has a dangling edge")

    public_text = public_graph_path.read_text(encoding="utf-8") + roster_path.read_text(encoding="utf-8")
    endpoint_literal = json.dumps(private["endpoint_xy"])
    private_hits = [label for label in ("endpoint_xy", "distance_to_goal_m", endpoint_literal) if label in public_text]
    if private_hits:
        raise ValueError(f"private arrival truth leaked into public graph pixels: {private_hits}")

    by_pose_yaw = {(row["pose_index"], row["viewport_yaw_index"]): row for row in graph_nodes}
    pose_indices = sorted({int(row["pose_index"]) for row in graph_nodes})
    if len(pose_indices) < 2:
        raise ValueError("action graph needs at least two source poses")
    sample_count = min(5, len(pose_indices))
    sample_pose_indices = tuple(
        pose_indices[round(index * (len(pose_indices) - 1) / (sample_count - 1))]
        for index in range(sample_count)
    )
    direction_checks = []
    for pose_index in sample_pose_indices:
        center = by_pose_yaw[(pose_index, 0)]
        left = by_pose_yaw[(pose_index, 1)]
        right = by_pose_yaw[(pose_index, -1)]
        left_shift, left_matches = _orb_horizontal_shift(
            pixel_root / center["rendered_frame_path"], pixel_root / left["rendered_frame_path"]
        )
        right_shift, right_matches = _orb_horizontal_shift(
            pixel_root / center["rendered_frame_path"], pixel_root / right["rendered_frame_path"]
        )
        direction_checks.append({
            "pose_index": pose_index,
            "turn_left_median_feature_shift_px": left_shift,
            "turn_right_median_feature_shift_px": right_shift,
            "turn_left_retained_match_count": left_matches,
            "turn_right_retained_match_count": right_matches,
            "pass": left_shift > MIN_DIRECTION_SHIFT_PX and right_shift < -MIN_DIRECTION_SHIFT_PX,
        })
    if not all(row["pass"] for row in direction_checks):
        raise ValueError("rendered turn direction does not match the V0 action label")

    import cv2

    receipt = {
        "schema_version": "blindassist_abotn_v0_action_graph_pixel_qualification_v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": "ABOTN_V0_ACTION_GRAPH_PIXELS_QUALIFIED_FOR_ONE_CLOSED_LOOP_RUN",
        "graph_node_count": len(graph_nodes),
        "frame_count": len(frames),
        "unique_frame_sha256_count": len({frame["sha256"] for frame in frames}),
        "edge_count_by_action": freeze["edge_count_by_action"],
        "shortest_start_to_arrival_steps": freeze["shortest_start_to_arrival_steps"],
        "maximum_instructions": freeze["maximum_instructions"],
        "direction_sample_pose_indices": list(sample_pose_indices),
        "minimum_direction_shift_px": MIN_DIRECTION_SHIFT_PX,
        "direction_checks": direction_checks,
        "opencv_version": cv2.__version__,
        "private_truth_literal_hits": private_hits,
        "provider_calls_before_qualification": 0,
        "teacher_calls_before_qualification": 0,
        "baseline_calls_before_qualification": 0,
        "one_shot_provider_execution_authorized": True,
        "pixel_renderer": pixels.get("renderer", {}).get("kind", "UNOFFICIAL_WEBGL_MECHANICS_CANARY"),
        "inputs": {
            "public_graph_sha256": _sha256(public_graph_path),
            "private_truth_sha256": _sha256(private_truth_path),
            "freeze_receipt_sha256": _sha256(freeze_path),
            "pixel_receipt_sha256": _sha256(pixel_receipt_path),
            "pixel_roster_sha256": _sha256(roster_path),
        },
        "claim_ceiling": "ACTION_GRAPH_AND_PIXEL_TRANSPORT_MECHANICS_ONLY",
        "next_action": "RUN_ONE_SEALED_CURRENT_FRAME_V0_CLOSED_LOOP_EPISODE",
    }
    if output_path.exists():
        raise ValueError("qualification output already exists; refusing replay")
    _atomic_json(output_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-graph", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--pixel-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = qualify(
        public_graph_path=args.public_graph.resolve(),
        private_truth_path=args.private_truth.resolve(),
        freeze_path=args.freeze_receipt.resolve(),
        pixel_receipt_path=args.pixel_receipt.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
