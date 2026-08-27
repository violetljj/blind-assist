"""Gate causal JRDB lidar geometry with one fixed person instance-mask source.

This bridge keeps the RGB tracker, DTR matcher, horizon, and lifecycle frozen.
It changes only the current metric-geometry information source after the fixed
full-box upper-lidar estimator proved too noisy: YOLO11n-seg person masks select
motion-compensated upper and lower Velodyne points before a median ground-plane
position is computed.

The mask/lidar ledger is written before JRDB annotations are opened.  There is
one model, one association rule, and one estimator; no parameter sweep.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Any, Sequence

from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    DETECTOR_CONFIDENCE,
    DETECTOR_MAX_DET,
    DETECTOR_NMS_IOU,
    FIRST_FRAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INFERENCE_SIZE,
    LAST_FRAME,
    MINIMUM_EVALUATOR_IOU,
    SCHEMA as RGB_BRIDGE_SCHEMA,
    SEQUENCE,
    TILE_STARTS,
    TILE_WIDTH,
    associate_frame,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)
from jrdb_sensor_geometry_bridge import (
    LidarScan,
    PERSON_RADIUS_M,
    evaluate_tracks,
    load_calibration,
    load_truth_and_associate,
    project_logical_to_stitched,
    read_jsonl,
    read_lower_lidar,
    read_upper_lidar,
    transform_points_to_image_time,
    write_json,
    write_jsonl,
)


SCHEMA = "dtr-r0-jrdb-dual-lidar-mask-geometry-bridge-v1"
SENSOR_LEDGER_SCHEMA = "dtr-r0-jrdb-dual-lidar-mask-track-geometry-v1"
CLAIM_CEILING = (
    "CURATED_PUBLIC_REAL_RGB_TRACK_PLUS_INSTANCE_MASK_GATED_CAUSAL_DUAL_LIDAR_ONLY"
)
MINIMUM_MASK_POINT_SUPPORT = 3


@dataclass(frozen=True)
class MaskCandidate:
    bbox_xyxy: list[float]
    confidence: float
    tile_start: int
    polygon_xy: Any

    def association_row(self) -> dict[str, Any]:
        return {"bbox_xyxy": self.bbox_xyxy}


def mask_candidates(predictions: Sequence[Any]) -> list[MaskCandidate]:
    import numpy as np

    require(len(predictions) == len(TILE_STARTS), "mask_tile_prediction_count_drift")
    output: list[MaskCandidate] = []
    for tile_start, prediction in zip(TILE_STARTS, predictions):
        boxes = getattr(prediction, "boxes", None)
        masks = getattr(prediction, "masks", None)
        if boxes is None or not len(boxes):
            continue
        require(masks is not None, "segment_prediction_missing_masks")
        polygons = masks.xy
        require(len(polygons) == len(boxes), "segment_box_mask_count_drift")
        coordinates = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        for index, polygon in enumerate(polygons):
            values = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(values) < 3:
                continue
            x1, y1, x2, y2 = (float(value) for value in coordinates[index])
            x1 = min(IMAGE_WIDTH, max(0.0, x1 + tile_start))
            x2 = min(IMAGE_WIDTH, max(0.0, x2 + tile_start))
            y1 = min(IMAGE_HEIGHT, max(0.0, y1))
            y2 = min(IMAGE_HEIGHT, max(0.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            output.append(
                MaskCandidate(
                    bbox_xyxy=[x1, y1, x2, y2],
                    confidence=float(confidence[index]),
                    tile_start=int(tile_start),
                    polygon_xy=values,
                )
            )
    return output


def estimate_mask_geometry(
    candidate: MaskCandidate,
    projected: Any,
    base_xy: Any,
) -> dict[str, Any] | None:
    import cv2
    import numpy as np

    pixels = np.asarray(projected, dtype=np.float64)
    points = np.asarray(base_xy, dtype=np.float64)
    local_u = pixels[:, 0] - candidate.tile_start
    v = pixels[:, 1]
    finite = np.all(np.isfinite(pixels), axis=1) & np.all(
        np.isfinite(points), axis=1
    )
    in_tile = (
        finite
        & (local_u >= 0.0)
        & (local_u < TILE_WIDTH)
        & (v >= 0.0)
        & (v < IMAGE_HEIGHT)
        & (np.linalg.norm(points, axis=1) > 0.25)
    )
    point_indices = np.flatnonzero(in_tile)
    if not len(point_indices):
        return None

    mask = np.zeros((IMAGE_HEIGHT, TILE_WIDTH), dtype=np.uint8)
    polygon = np.rint(candidate.polygon_xy).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, TILE_WIDTH - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, IMAGE_HEIGHT - 1)
    cv2.fillPoly(mask, [polygon], 1)
    px = np.floor(local_u[point_indices]).astype(np.int32)
    py = np.floor(v[point_indices]).astype(np.int32)
    selected_indices = point_indices[mask[py, px] > 0]
    if len(selected_indices) < MINIMUM_MASK_POINT_SUPPORT:
        return None
    selected = points[selected_indices]
    return {
        "forward_m": float(np.median(selected[:, 0])),
        "left_m": float(np.median(selected[:, 1])),
        "mask_point_support": int(len(selected)),
    }


def materialize_mask_sensor_ledger(
    detector_rows: Sequence[dict[str, Any]],
    image_records: Sequence[dict[str, Any]],
    timestamps: dict[int, float],
    poses: Sequence[dict[str, Any]],
    upper_scans: Sequence[LidarScan],
    lower_scans: Sequence[LidarScan],
    calibration: dict[str, Any],
    model_path: Path,
    batch_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    require(batch_frames > 0, "batch_frames_must_be_positive")
    detector_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in detector_rows:
        detector_by_frame.setdefault(int(row["frame_index"]), []).append(row)
    image_by_frame = {int(row["frame_index"]): row for row in image_records}
    require(
        set(image_by_frame) == set(range(FIRST_FRAME, LAST_FRAME + 1)),
        "rgb_result_image_window_drift",
    )

    model = YOLO(str(model_path.resolve(strict=True)), task="segment")
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    upper_scan_times = [item.time_ns for item in upper_scans]
    lower_scan_times = [item.time_ns for item in lower_scans]
    output: list[dict[str, Any]] = []
    upper_lidar_ages: list[float] = []
    lower_lidar_ages: list[float] = []
    match_ious: list[float] = []
    frame_context: dict[int, dict[str, Any]] = {}
    mask_candidate_count = 0
    mask_track_matches = 0
    frames_with_geometry = 0
    frames = list(range(FIRST_FRAME, LAST_FRAME + 1))

    for batch_start in range(0, len(frames), batch_frames):
        batch = frames[batch_start : batch_start + batch_frames]
        crops = []
        for frame in batch:
            record = image_by_frame[frame]
            image_path = Path(record["path"]).resolve(strict=True)
            require(
                sha256_file(image_path) == record["sha256"],
                f"image_hash_drift:{frame}",
            )
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            require(image is not None, f"image_decode_failed:{frame}")
            require(
                image.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH),
                f"image_shape_drift:{frame}:{image.shape[:2]}",
            )
            crops.extend(image[:, start : start + TILE_WIDTH] for start in TILE_STARTS)

        predictions = model.predict(
            crops,
            imgsz=INFERENCE_SIZE,
            conf=DETECTOR_CONFIDENCE,
            iou=DETECTOR_NMS_IOU,
            classes=[0],
            max_det=DETECTOR_MAX_DET,
            augment=False,
            device=device,
            batch=len(crops),
            verbose=False,
        )
        require(len(predictions) == len(crops), "mask_detector_batch_length_drift")

        for offset, frame in enumerate(batch):
            candidates = mask_candidates(
                predictions[
                    offset * len(TILE_STARTS) : (offset + 1) * len(TILE_STARTS)
                ]
            )
            mask_candidate_count += len(candidates)
            detector_frame = detector_by_frame.get(frame, [])
            associations = associate_frame(
                detector_frame,
                [candidate.association_row() for candidate in candidates],
            )
            candidate_by_detector = {
                detector_index: (candidates[candidate_index], overlap)
                for detector_index, candidate_index, overlap in associations
            }
            mask_track_matches += len(candidate_by_detector)
            match_ious.extend(overlap for _candidate, overlap in candidate_by_detector.values())

            image_ns = round(timestamps[frame] * 1e9)
            upper_index = bisect_right(upper_scan_times, image_ns) - 1
            lower_index = bisect_right(lower_scan_times, image_ns) - 1
            require(upper_index >= 0, f"causal_upper_lidar_missing:{frame}")
            require(lower_index >= 0, f"causal_lower_lidar_missing:{frame}")
            upper_scan = upper_scans[upper_index]
            lower_scan = lower_scans[lower_index]
            upper_age_s = (image_ns - upper_scan.time_ns) / 1e9
            lower_age_s = (image_ns - lower_scan.time_ns) / 1e9
            require(upper_age_s >= 0.0, f"future_upper_lidar_selected:{frame}")
            require(lower_age_s >= 0.0, f"future_lower_lidar_selected:{frame}")
            image_pose = interpolate_pose(poses, image_ns)
            upper_base_xy, upper_logical = transform_points_to_image_time(
                upper_scan.logical_points,
                interpolate_pose(poses, upper_scan.time_ns),
                image_pose,
            )
            lower_base_xy, lower_logical = transform_points_to_image_time(
                lower_scan.logical_points,
                interpolate_pose(poses, lower_scan.time_ns),
                image_pose,
            )
            base_xy = np.concatenate((upper_base_xy, lower_base_xy), axis=0)
            logical = np.concatenate((upper_logical, lower_logical), axis=0)
            projected = project_logical_to_stitched(logical, calibration)

            frame_geometry = 0
            for detector_index, detector in enumerate(detector_frame):
                matched = candidate_by_detector.get(detector_index)
                candidate = matched[0] if matched else None
                overlap = matched[1] if matched else None
                geometry = (
                    estimate_mask_geometry(candidate, projected, base_xy)
                    if candidate is not None
                    else None
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
                        "mask_bbox_xyxy": candidate.bbox_xyxy if candidate else None,
                        "mask_confidence": candidate.confidence if candidate else None,
                        "mask_track_iou": overlap,
                        "upper_lidar_time_s": upper_scan.time_ns / 1e9,
                        "upper_lidar_age_s": upper_age_s,
                        "lower_lidar_time_s": lower_scan.time_ns / 1e9,
                        "lower_lidar_age_s": lower_age_s,
                        "geometry": geometry,
                    }
                )
                frame_geometry += int(geometry is not None)
            frames_with_geometry += int(frame_geometry > 0)
            upper_lidar_ages.append(upper_age_s)
            lower_lidar_ages.append(lower_age_s)
            frame_context[frame] = {
                "pose": image_pose,
                "image_time_s": timestamps[frame],
            }

        print(
            json.dumps(
                {
                    "mask_geometry_frames": min(batch_start + len(batch), len(frames)),
                    "total": len(frames),
                    "geometry_occurrences": sum(
                        row["geometry"] is not None for row in output
                    ),
                }
            ),
            flush=True,
        )

    geometry_rows = sum(row["geometry"] is not None for row in output)
    coverage = {
        "frames": len(frames),
        "frames_with_any_geometry": frames_with_geometry,
        "detector_track_occurrences": len(output),
        "geometry_occurrences": geometry_rows,
        "geometry_coverage": geometry_rows / len(output) if output else None,
        "mask_candidate_count": mask_candidate_count,
        "mask_track_matches": mask_track_matches,
        "mask_track_match_coverage": (
            mask_track_matches / len(output) if output else None
        ),
        "mask_track_iou": {
            "minimum": min(match_ious) if match_ious else None,
            "median": statistics.median(match_ious) if match_ious else None,
            "maximum": max(match_ious) if match_ious else None,
        },
        "causal_lidar_age_s": {
            "upper": {
                "minimum": min(upper_lidar_ages),
                "median": statistics.median(upper_lidar_ages),
                "maximum": max(upper_lidar_ages),
            },
            "lower": {
                "minimum": min(lower_lidar_ages),
                "median": statistics.median(lower_lidar_ages),
                "maximum": max(lower_lidar_ages),
            },
        },
    }
    runtime = {
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return output, coverage, frame_context, runtime


def run(args: argparse.Namespace) -> dict[str, Any]:
    rgb_result_path = args.rgb_result.resolve(strict=True)
    rgb_tracks_path = args.rgb_tracks.resolve(strict=True)
    labels_path = args.labels_zip.resolve(strict=True)
    timestamps_path = args.timestamps_zip.resolve(strict=True)
    bag_path = args.bag.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    model_path = args.segmentation_model.resolve(strict=True)
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
    image_ns = [
        round(timestamps[frame] * 1e9)
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    ]
    upper_scans, upper_scan_coverage = read_upper_lidar(
        bag_path,
        calibration,
        min(image_ns) - 200_000_000,
        max(image_ns),
    )
    lower_scans, lower_scan_coverage = read_lower_lidar(
        bag_path,
        calibration,
        min(image_ns) - 200_000_000,
        max(image_ns),
    )
    sensor_rows, sensor_coverage, frame_context, runtime = (
        materialize_mask_sensor_ledger(
            detector_rows,
            rgb_result["source"]["images"]["records"],
            timestamps,
            poses,
            upper_scans,
            lower_scans,
            calibration,
            model_path,
            args.batch_frames,
        )
    )
    sensor_ledger = args.output.with_name(
        args.output.stem + ".sensor-tracks.jsonl"
    ).resolve()
    write_jsonl(sensor_ledger, sensor_rows)
    sensor_ledger_sha = sha256_file(sensor_ledger)

    # Evaluator-only annotation access begins after the sensor ledger is sealed.
    tracks, geometry_quality = load_truth_and_associate(
        labels_path, sensor_rows, frame_context
    )
    evaluation = evaluate_tracks(tracks)
    return {
        "schema_version": SCHEMA,
        "status": "DTR_R0_CAUSAL_DUAL_LIDAR_MASK_GEOMETRY_OBSERVATION_AVAILABLE",
        "claim_ceiling": CLAIM_CEILING,
        "source": {
            "dataset": "JRDB public train split",
            "sequence": SEQUENCE,
            "window": {"first_frame": FIRST_FRAME, "last_frame": LAST_FRAME},
            "rgb_bridge_result": str(rgb_result_path),
            "rgb_bridge_result_sha256": sha256_file(rgb_result_path),
            "rgb_track_ledger": str(rgb_tracks_path),
            "rgb_track_ledger_sha256": sha256_file(rgb_tracks_path),
            "segmentation_model": str(model_path),
            "segmentation_model_sha256": sha256_file(model_path),
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
            "topics": [
                "upper_velodyne/velodyne_points",
                "lower_velodyne/velodyne_points",
            ],
            "temporal_rule": "latest per-sensor lidar header timestamp <= image timestamp",
            "motion_compensation": "each lidar-time base -> odom -> image-time base using causal bag TF",
            "projection": "official JRDB logical RGB360 cylindrical projection",
            "mask_source": {
                "model": str(model_path),
                "model_sha256": sha256_file(model_path),
                "task": "segment/person",
                "tile_width": TILE_WIDTH,
                "tile_starts": list(TILE_STARTS),
                "confidence": DETECTOR_CONFIDENCE,
                "nms_iou": DETECTOR_NMS_IOU,
                "image_size": INFERENCE_SIZE,
                "max_detections": DETECTOR_MAX_DET,
                "track_mask_association_iou": MINIMUM_EVALUATOR_IOU,
                "parameter_sweep": False,
                "runtime": runtime,
            },
            "estimator": {
                "rule": "median base-frame x/y of all fused upper/lower raw lidar points inside the matched person instance mask",
                "minimum_mask_point_support": MINIMUM_MASK_POINT_SUPPORT,
                "person_radius_m": PERSON_RADIUS_M,
                "parameter_sweep": False,
            },
            "scan_coverage": {
                "upper": upper_scan_coverage,
                "lower": lower_scan_coverage,
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
            "This is the same single curated 143-frame Development window as the RGB bridge.",
            "The fixed mask source was introduced after the full-box lidar source missed the desired false-alert effect; it is not an independent confirmation.",
            "The detector tracker and segmenter are separate fixed YOLO11n model passes joined by current-frame IoU.",
            "Evaluator identity plus future center/body-extent truth still use JRDB annotations; current DTR metric observations use raw lidar centers and a fixed 0.30 m person radius.",
            "Both JRDB Velodynes are fused once; there is no model, threshold, tracker, route-matcher, or lifecycle sweep.",
            "This is offline public-data evidence, not phone/Android runtime, user benefit, natural-distribution, or safety evidence.",
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
    parser.add_argument("--segmentation-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-frames", type=int, default=4)
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
