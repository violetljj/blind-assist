"""Evaluate one phone-transferable known-height geometry source for DTR-R0.

The sealed JRDB RGB detector/tracker ledger supplies only current person boxes.
For each box, a fixed 1.70 m upright-person prior and the official vertical
focal length recover horizontal range; the stitched cylindrical coordinate
recovers bearing.  The resulting truth-blind metric ledger is written before
JRDB identity and future event annotations are opened.

There is one height prior and no sweep.  DTR tracking, route intersection,
horizon, lifecycle, and evaluator remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    BASE_LINK_FROM_LOGICAL_RGB360_X_M,
    BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
    FIRST_FRAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LAST_FRAME,
    SCHEMA as RGB_BRIDGE_SCHEMA,
    SEQUENCE,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)
from jrdb_sensor_geometry_bridge import (
    PERSON_RADIUS_M,
    evaluate_tracks,
    load_calibration,
    load_truth_and_associate,
    read_jsonl,
    write_json,
    write_jsonl,
)


SCHEMA = "dtr-r0-jrdb-known-height-geometry-bridge-v1"
SENSOR_LEDGER_SCHEMA = "dtr-r0-jrdb-known-height-track-geometry-v1"
CLAIM_CEILING = "CURATED_PUBLIC_REAL_RGB_TRACK_PLUS_FIXED_KNOWN_HEIGHT_GEOMETRY_ONLY"
PERSON_HEIGHT_M = 1.70


def estimate_known_height_geometry(
    bbox: Sequence[float],
    focal_y_px: float,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    height_px = y2 - y1
    if not math.isfinite(height_px) or height_px <= 0.0:
        return None
    horizontal_range_m = focal_y_px * PERSON_HEIGHT_M / height_px
    u_center = 0.5 * (x1 + x2)
    bearing_rad = (u_center / IMAGE_WIDTH) * (2.0 * math.pi) - math.pi
    logical_forward_m = horizontal_range_m * math.cos(bearing_rad)
    logical_left_m = -horizontal_range_m * math.sin(bearing_rad)
    return {
        "forward_m": logical_forward_m + BASE_LINK_FROM_LOGICAL_RGB360_X_M,
        "left_m": logical_left_m + BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
        "horizontal_range_m": horizontal_range_m,
        "bbox_height_px": height_px,
        "bearing_rad": bearing_rad,
        "touches_vertical_boundary": bool(y1 <= 0.0 or y2 >= IMAGE_HEIGHT),
    }


def materialize_sensor_ledger(
    detector_rows: Sequence[dict[str, Any]],
    timestamps: dict[int, float],
    poses: Sequence[dict[str, Any]],
    focal_y_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]]]:
    detector_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in detector_rows:
        detector_by_frame.setdefault(int(row["frame_index"]), []).append(row)

    output: list[dict[str, Any]] = []
    frame_context: dict[int, dict[str, Any]] = {}
    frames_with_geometry = 0
    boundary_occurrences = 0
    for frame in range(FIRST_FRAME, LAST_FRAME + 1):
        image_ns = round(timestamps[frame] * 1e9)
        pose = interpolate_pose(poses, image_ns)
        frame_geometry = 0
        for detector in detector_by_frame.get(frame, []):
            geometry = estimate_known_height_geometry(
                detector["bbox_xyxy"], focal_y_px
            )
            output.append(
                {
                    "schema": SENSOR_LEDGER_SCHEMA,
                    "sequence": SEQUENCE,
                    "frame_index": frame,
                    "image_time_s": timestamps[frame],
                    "track_id": detector["track_id"],
                    "bbox_xyxy": detector["bbox_xyxy"],
                    "confidence": detector["confidence"],
                    "image_sha256": detector["image_sha256"],
                    "geometry": geometry,
                }
            )
            frame_geometry += int(geometry is not None)
            boundary_occurrences += int(
                geometry is not None and geometry["touches_vertical_boundary"]
            )
        frames_with_geometry += int(frame_geometry > 0)
        frame_context[frame] = {
            "pose": pose,
            "image_time_s": timestamps[frame],
        }

    geometry_rows = sum(row["geometry"] is not None for row in output)
    return output, {
        "frames": LAST_FRAME - FIRST_FRAME + 1,
        "frames_with_any_geometry": frames_with_geometry,
        "detector_track_occurrences": len(output),
        "geometry_occurrences": geometry_rows,
        "geometry_coverage": geometry_rows / len(output) if output else None,
        "vertical_boundary_occurrences": boundary_occurrences,
    }, frame_context


def run(args: argparse.Namespace) -> dict[str, Any]:
    rgb_result_path = args.rgb_result.resolve(strict=True)
    rgb_tracks_path = args.rgb_tracks.resolve(strict=True)
    labels_path = args.labels_zip.resolve(strict=True)
    timestamps_path = args.timestamps_zip.resolve(strict=True)
    bag_path = args.bag.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    rgb_result = json.loads(rgb_result_path.read_text(encoding="utf-8"))
    require(rgb_result.get("schema_version") == RGB_BRIDGE_SCHEMA, "rgb_result_schema")
    require(
        sha256_file(rgb_tracks_path)
        == rgb_result["truth_blind_detector_tracker"]["ledger_sha256"],
        "rgb_track_ledger_hash_drift",
    )
    detector_rows = read_jsonl(rgb_tracks_path)
    timestamps = load_image_timestamps(timestamps_path)
    calibration = load_calibration(calibration_dir)
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    sensor_rows, sensor_coverage, frame_context = materialize_sensor_ledger(
        detector_rows,
        timestamps,
        poses,
        float(calibration["median_focal_y_px"]),
    )
    sensor_ledger = args.output.with_name(
        args.output.stem + ".sensor-tracks.jsonl"
    ).resolve()
    write_jsonl(sensor_ledger, sensor_rows)
    sensor_ledger_sha = sha256_file(sensor_ledger)

    # Evaluator-only annotation access begins after the metric ledger is sealed.
    tracks, geometry_quality = load_truth_and_associate(
        labels_path, sensor_rows, frame_context
    )
    evaluation = evaluate_tracks(tracks)
    return {
        "schema_version": SCHEMA,
        "status": "DTR_R0_KNOWN_HEIGHT_RGB_GEOMETRY_OBSERVATION_AVAILABLE",
        "claim_ceiling": CLAIM_CEILING,
        "source": {
            "dataset": "JRDB public train split",
            "sequence": SEQUENCE,
            "window": {"first_frame": FIRST_FRAME, "last_frame": LAST_FRAME},
            "rgb_bridge_result": str(rgb_result_path),
            "rgb_bridge_result_sha256": sha256_file(rgb_result_path),
            "rgb_track_ledger": str(rgb_tracks_path),
            "rgb_track_ledger_sha256": sha256_file(rgb_tracks_path),
            "labels_zip": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps_zip": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "calibration": calibration,
            "bag_authority": bag_authority,
        },
        "truth_blind_sensor_geometry": {
            "ledger": str(sensor_ledger),
            "ledger_sha256": sensor_ledger_sha,
            "source": "current detector box plus fixed person height and camera vertical focal length",
            "estimator": {
                "person_height_m": PERSON_HEIGHT_M,
                "person_radius_m": PERSON_RADIUS_M,
                "horizontal_range_rule": "focal_y_px * person_height_m / bbox_height_px",
                "bearing_rule": "stitched cylindrical bbox-center bearing",
                "vertical_boundary_filter": False,
                "parameter_sweep": False,
            },
            "coverage": sensor_coverage,
        },
        "privileged_evaluator": {
            "association": "current tracker bbox to native stitched 2-D label at IoU >= 0.30",
            "future_truth": "future native 3-D centers and body extent",
            "geometry_quality": geometry_quality,
        },
        "evaluation": evaluation,
        "limitations": [
            "This is the same single curated 143-frame Development window as the prior bridges.",
            "A fixed 1.70 m upright full-body prior is wrong for children, seated/crouched people, truncation, and imperfect detector boxes.",
            "Vertical image-boundary boxes are reported but not filtered, and there is no height or threshold sweep.",
            "Evaluator identity plus future center/body-extent truth still use JRDB annotations; current observations use only RGB boxes, calibration, and fixed priors.",
            "This is offline public-data evidence, not Android runtime, user benefit, natural-distribution, or safety evidence.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-result", type=Path, required=True)
    parser.add_argument("--rgb-tracks", type=Path, required=True)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    result = run(args)
    write_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "sensor_coverage": result["truth_blind_sensor_geometry"]["coverage"],
                "geometry_quality": result["privileged_evaluator"]["geometry_quality"],
                "evaluation": result["evaluation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
