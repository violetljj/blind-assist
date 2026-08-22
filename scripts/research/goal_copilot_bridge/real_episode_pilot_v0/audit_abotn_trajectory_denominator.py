"""Audit frozen ABotN trajectory pixels against evaluator-private arrival truth."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "blindassist_abotn_trajectory_arrival_denominator_v0"
PIXEL_SCHEMA = "blindassist_abotn_webgl_trajectory_pixels_v0"
ARRIVE_THRESHOLD_M = 2.0


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


def audit(annotation: dict[str, Any], pixels: dict[str, Any], pixels_dir: Path) -> dict[str, Any]:
    if pixels.get("schema_version") != PIXEL_SCHEMA or pixels.get("terminal") != "ABOTN_WEBGL_TRAJECTORY_PIXELS_PASS":
        raise ValueError("trajectory pixel receipt is not eligible")
    trajectory = annotation.get("trajectory")
    endpoint = annotation.get("label", {}).get("extend", {}).get("end_point")
    if not isinstance(trajectory, list) or not trajectory or not isinstance(endpoint, list) or len(endpoint) != 2:
        raise ValueError("annotation lacks trajectory or metric endpoint")
    frames = pixels.get("frames")
    if not isinstance(frames, list) or len(frames) != len(trajectory):
        raise ValueError("trajectory/frame denominator mismatch")
    for index, frame in enumerate(frames):
        if frame.get("observation_index") != index:
            raise ValueError("frame order drift")
        frame_path = pixels_dir / str(frame.get("path"))
        if not frame_path.is_file() or _sha256(frame_path) != frame.get("sha256"):
            raise ValueError(f"frame payload changed at index {index}")

    distances = [
        math.hypot(float(pose["x"]) - float(endpoint[0]), float(pose["y"]) - float(endpoint[1]))
        for pose in trajectory
    ]
    within = [index for index, distance in enumerate(distances) if distance < ARRIVE_THRESHOLD_M]
    first_within = within[0] if within else None
    monotonic_nonincreasing_steps = sum(
        current <= previous for previous, current in zip(distances, distances[1:])
    )
    return {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": "ABOTN_TRAJECTORY_ARRIVAL_DENOMINATOR_PRESENT_OPEN_LOOP_CONTROL_NOT_EVALUABLE",
        "observation_count": len(distances),
        "arrival_rule": "distance_to_goal_m < arrive_threshold_m",
        "arrive_threshold_m": ARRIVE_THRESHOLD_M,
        "outside_arrival_count": len(distances) - len(within),
        "within_arrival_count": len(within),
        "first_within_arrival_observation_index": first_within,
        "initial_distance_to_goal_m": distances[0],
        "final_distance_to_goal_m": distances[-1],
        "minimum_distance_to_goal_m": min(distances),
        "monotonic_nonincreasing_step_count": monotonic_nonincreasing_steps,
        "transition_count": len(distances) - 1,
        "pixel_payload_integrity": "PASS",
        "truth_authority": "EVALUATOR_PRIVATE_METRIC_ENDPOINT_ONLY",
        "open_loop_trajectory_source": "SOURCE_DEMONSTRATION_NOT_PROVIDER_CONTROLLED",
        "provider_execution_eligibility": "NOT_AUTHORIZED_NO_CLOSED_LOOP_ACTION_ADAPTER_OR_ARRIVAL_OUTPUT",
        "selection_accuracy": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
        "control_success": "NOT_EVALUABLE_OPEN_LOOP_SOURCE_TRAJECTORY",
        "provider_calls": 0,
        "teacher_calls": 0,
        "baseline_calls": 0,
        "claim_ceiling": "TRAJECTORY_PIXEL_AND_METRIC_ARRIVAL_DENOMINATOR_MECHANICS_ONLY",
        "next_action": "IMPLEMENT_EXISTING_V0_ACTION_TO_RENDERER_ADAPTER_BEFORE_ANY_PROVIDER_CALL",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--pixel-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    annotation_path = args.annotation.resolve()
    pixel_receipt_path = args.pixel_receipt.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise ValueError("output already exists; refusing replay")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    pixels = json.loads(pixel_receipt_path.read_text(encoding="utf-8"))
    receipt = audit(annotation, pixels, pixel_receipt_path.parent)
    receipt["inputs"] = {
        "annotation_path": str(annotation_path),
        "annotation_sha256": _sha256(annotation_path),
        "pixel_receipt_path": str(pixel_receipt_path),
        "pixel_receipt_sha256": _sha256(pixel_receipt_path),
    }
    _atomic_json(output_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
