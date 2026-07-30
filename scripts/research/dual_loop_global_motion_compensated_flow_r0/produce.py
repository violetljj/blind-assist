"""Truth-blind global-motion-compensated target-flow producer."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


PROTOCOL_ID = "DUAL_LOOP_GLOBAL_MOTION_COMPENSATED_TARGET_FLOW_R0"
IMPLEMENTATION_ID = "DUAL_LOOP_GMC_TARGET_FLOW_IMPL_R0"
ARM_ID = "BACKGROUND_HOMOGRAPHY_RESIDUAL_TARGET_SIMILARITY"
EXPECTED_REPLAY_SHA256 = "14f1f7f0f330d8b01146e37c31505240f3f0e8d301846ebcad44a628948e6440"
EXPECTED_ROWS = 13_014
TTL_NS = 100_000_000
PARAMETERS: dict[str, Any] = {
    "causal_lookback_frames": 1,
    "ttl_ns": TTL_NS,
    "background_roi_expansion_fraction": 0.10,
    "background_max_corners": 400,
    "target_max_corners": 80,
    "quality_level": 0.01,
    "min_distance_px": 5,
    "block_size_px": 5,
    "lk_window_px": [21, 21],
    "lk_max_level": 2,
    "lk_termination_count": 20,
    "lk_termination_epsilon": 0.03,
    "fb_error_max_px": 1.5,
    "background_homography_ransac_px": 2.0,
    "background_minimum_inliers": 20,
    "background_minimum_inlier_fraction": 0.50,
    "target_current_roi_expansion_fraction": 0.15,
    "target_minimum_tracks": 8,
    "target_minimum_quadrants": 2,
    "target_similarity_ransac_px": 1.5,
    "target_minimum_similarity_inliers": 6,
    "quality_track_reference": 24,
    "quality_floor": 0.50,
}
PARAMETER_SHA256 = hashlib.sha256(
    json.dumps(PARAMETERS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def roi_pixels(row: dict[str, Any], shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = shape
    cx, cy, box_width, box_height = [float(value) for value in row["roi_xywh_normalized"]]
    values = (cx * width, cy * height, box_width * width, box_height * height)
    if not all(math.isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
        raise ValueError("invalid replay ROI")
    return values


def rectangle_mask(
    shape: tuple[int, int],
    roi: tuple[float, float, float, float],
    expansion: float = 0.0,
) -> np.ndarray:
    height, width = shape
    cx, cy, box_width, box_height = roi
    half_width = box_width * (1.0 + expansion) / 2.0
    half_height = box_height * (1.0 + expansion) / 2.0
    left = max(0, int(math.floor(cx - half_width)))
    top = max(0, int(math.floor(cy - half_height)))
    right = min(width, int(math.ceil(cx + half_width)))
    bottom = min(height, int(math.ceil(cy + half_height)))
    mask = np.zeros(shape, dtype=np.uint8)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 255
    return mask


def track_points(
    previous: np.ndarray,
    current: np.ndarray,
    points: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points is None or len(points) == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty, np.empty((0,), dtype=np.float64)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(PARAMETERS["lk_termination_count"]),
        float(PARAMETERS["lk_termination_epsilon"]),
    )
    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=tuple(PARAMETERS["lk_window_px"]),
        maxLevel=int(PARAMETERS["lk_max_level"]),
        criteria=criteria,
    )
    if current_points is None or forward_status is None:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty, np.empty((0,), dtype=np.float64)
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        current_points,
        None,
        winSize=tuple(PARAMETERS["lk_window_px"]),
        maxLevel=int(PARAMETERS["lk_max_level"]),
        criteria=criteria,
    )
    if backward is None or backward_status is None:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty, np.empty((0,), dtype=np.float64)
    first = points.reshape(-1, 2).astype(np.float64)
    second = current_points.reshape(-1, 2).astype(np.float64)
    back = backward.reshape(-1, 2).astype(np.float64)
    error = np.linalg.norm(back - first, axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(first).all(axis=1)
        & np.isfinite(second).all(axis=1)
        & np.isfinite(error)
        & (error <= float(PARAMETERS["fb_error_max_px"]))
    )
    return first[valid], second[valid], error[valid]


def inside_roi(
    points: np.ndarray,
    roi: tuple[float, float, float, float],
    shape: tuple[int, int],
    expansion: float,
) -> np.ndarray:
    height, width = shape
    cx, cy, box_width, box_height = roi
    half_width = box_width * (1.0 + expansion) / 2.0
    half_height = box_height * (1.0 + expansion) / 2.0
    return (
        (points[:, 0] >= max(0.0, cx - half_width))
        & (points[:, 0] < min(float(width), cx + half_width))
        & (points[:, 1] >= max(0.0, cy - half_height))
        & (points[:, 1] < min(float(height), cy + half_height))
    )


def estimate_background_homography(
    previous: np.ndarray,
    current: np.ndarray,
    previous_rois: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    mask = np.full(previous.shape, 255, dtype=np.uint8)
    for roi in previous_rois:
        mask[rectangle_mask(previous.shape, roi, float(PARAMETERS["background_roi_expansion_fraction"])) > 0] = 0
    features = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=int(PARAMETERS["background_max_corners"]),
        qualityLevel=float(PARAMETERS["quality_level"]),
        minDistance=float(PARAMETERS["min_distance_px"]),
        mask=mask,
        blockSize=int(PARAMETERS["block_size_px"]),
        useHarrisDetector=False,
    )
    first, second, errors = track_points(previous, current, features)
    if len(first) < int(PARAMETERS["background_minimum_inliers"]):
        return None, {"reason": "BACKGROUND_TRACKS_LT_20", "tracks": len(first)}
    homography, inliers = cv2.findHomography(
        first.astype(np.float32),
        second.astype(np.float32),
        cv2.RANSAC,
        float(PARAMETERS["background_homography_ransac_px"]),
    )
    if homography is None or inliers is None or not np.isfinite(homography).all():
        return None, {"reason": "BACKGROUND_HOMOGRAPHY_FAILED", "tracks": len(first)}
    inlier_count = int(inliers.reshape(-1).astype(bool).sum())
    fraction = inlier_count / len(first)
    if (
        inlier_count < int(PARAMETERS["background_minimum_inliers"])
        or fraction < float(PARAMETERS["background_minimum_inlier_fraction"])
    ):
        return None, {
            "reason": "BACKGROUND_HOMOGRAPHY_SUPPORT_LOW",
            "tracks": len(first),
            "inliers": inlier_count,
            "inlier_fraction": fraction,
        }
    return homography.astype(np.float64), {
        "reason": None,
        "tracks": len(first),
        "inliers": inlier_count,
        "inlier_fraction": fraction,
        "median_fb_error_px": float(np.median(errors)),
    }


def residual_target_rate(
    previous: np.ndarray,
    current: np.ndarray,
    previous_roi: tuple[float, float, float, float],
    current_roi: tuple[float, float, float, float],
    homography: np.ndarray,
    dt_seconds: float,
    background_quality: dict[str, Any],
) -> tuple[float | None, float, dict[str, Any], str | None]:
    mask = rectangle_mask(previous.shape, previous_roi)
    features = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=int(PARAMETERS["target_max_corners"]),
        qualityLevel=float(PARAMETERS["quality_level"]),
        minDistance=float(PARAMETERS["min_distance_px"]),
        mask=mask,
        blockSize=int(PARAMETERS["block_size_px"]),
        useHarrisDetector=False,
    )
    first, second, errors = track_points(previous, current, features)
    if len(first):
        keep = inside_roi(
            second,
            current_roi,
            current.shape,
            float(PARAMETERS["target_current_roi_expansion_fraction"]),
        )
        first, second, errors = first[keep], second[keep], errors[keep]
    if len(first) < int(PARAMETERS["target_minimum_tracks"]):
        return None, 0.0, {"target_tracks": len(first), **background_quality}, "TARGET_TRACKS_LT_8"
    center = np.asarray(previous_roi[:2], dtype=np.float64)
    quadrants = {
        (bool(point[0] >= center[0]), bool(point[1] >= center[1]))
        for point in first
    }
    if len(quadrants) < int(PARAMETERS["target_minimum_quadrants"]):
        return None, 0.0, {"target_tracks": len(first), "quadrants": len(quadrants), **background_quality}, "TARGET_QUADRANTS_LT_2"
    predicted = cv2.perspectiveTransform(
        first.reshape(-1, 1, 2).astype(np.float32),
        homography.astype(np.float64),
    ).reshape(-1, 2)
    affine, inliers = cv2.estimateAffinePartial2D(
        predicted.astype(np.float32),
        second.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(PARAMETERS["target_similarity_ransac_px"]),
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or inliers is None or not np.isfinite(affine).all():
        return None, 0.0, {"target_tracks": len(first), **background_quality}, "TARGET_SIMILARITY_FAILED"
    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < int(PARAMETERS["target_minimum_similarity_inliers"]):
        return None, 0.0, {"target_tracks": len(first), "target_inliers": inlier_count, **background_quality}, "TARGET_SIMILARITY_INLIERS_LT_6"
    scale = math.sqrt(float(affine[0, 0]) ** 2 + float(affine[0, 1]) ** 2)
    if not math.isfinite(scale) or scale <= 0:
        return None, 0.0, {"target_tracks": len(first), **background_quality}, "TARGET_SCALE_INVALID"
    predicted_target = cv2.transform(predicted.reshape(-1, 1, 2), affine).reshape(-1, 2)
    residual = np.linalg.norm(predicted_target - second, axis=1)
    median_residual = float(np.median(residual[inlier_mask]))
    rate = math.log(scale) / dt_seconds
    quality = (
        min(1.0, inlier_count / float(PARAMETERS["quality_track_reference"]))
        * float(background_quality["inlier_fraction"])
        * max(0.0, 1.0 - median_residual / float(PARAMETERS["target_similarity_ransac_px"]))
    )
    components = {
        **background_quality,
        "target_tracks": len(first),
        "target_inliers": inlier_count,
        "target_inlier_fraction": inlier_count / len(first),
        "target_median_residual_px": median_residual,
        "target_median_fb_error_px": float(np.median(errors)),
        "residual_similarity_scale": scale,
    }
    if quality < float(PARAMETERS["quality_floor"]):
        return None, quality, components, "QUALITY_BELOW_0_50"
    return rate, quality, components, None


class GrayCache:
    def __init__(self, image_root: Path, capacity: int = 6) -> None:
        self.image_root = image_root
        self.capacity = capacity
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, relative: str) -> np.ndarray:
        cached = self.cache.pop(relative, None)
        if cached is not None:
            self.cache[relative] = cached
            return cached
        image = cv2.imread(str(self.image_root / relative), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.image_root / relative)
        self.cache[relative] = image
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return image


def base_output(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "parameter_sha256": PARAMETER_SHA256,
        "arm_id": ARM_ID,
        "capture_id": row["capture_id"],
        "source_frame_id": row["source_frame_id"],
        "captured_at_ns": int(row["captured_at_ns"]),
        "available_at_ns": int(row["captured_at_ns"]),
        "target_id": row["target_id"],
        "track_epoch": row["track_epoch"],
        "region": row["region"],
        "signed_approach_rate_per_s": None,
        "quality": {"score": 0.0, "components": {}},
        "ttl_ns": TTL_NS,
        "valid_until_ns": int(row["captured_at_ns"]) + TTL_NS,
        "abstention_reason": None,
    }


def run(replay_path: Path, image_root: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("output namespace already exists")
    if sha256_file(replay_path) != EXPECTED_REPLAY_SHA256:
        raise ValueError("replay input hash drift")
    rows = read_jsonl(replay_path)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("replay row count drift")
    by_target_index = {
        (str(row["target_id"]), int(row["source_frame_index"])): row for row in rows
    }
    rois_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rois_by_index[int(row["source_frame_index"])].append(row)
    cache = GrayCache(image_root)
    homography_cache: dict[tuple[str, str], tuple[np.ndarray | None, dict[str, Any]]] = {}
    output_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    cv2.setNumThreads(1)
    for row in rows:
        output = base_output(row)
        current_index = int(row["source_frame_index"])
        previous_row = by_target_index.get((str(row["target_id"]), current_index - 1))
        if (
            bool(row.get("history_reset"))
            or previous_row is None
            or previous_row["track_epoch"] != row["track_epoch"]
        ):
            output["abstention_reason"] = "INSUFFICIENT_HISTORY"
            output_rows.append(output)
            continue
        dt_ns = int(row["captured_at_ns"]) - int(previous_row["captured_at_ns"])
        if dt_ns <= 0 or dt_ns > TTL_NS:
            output["abstention_reason"] = "HISTORY_GAP"
            output_rows.append(output)
            continue
        previous = cache.get(str(previous_row["image_relative_path"]))
        current = cache.get(str(row["image_relative_path"]))
        if previous.shape != current.shape:
            output["abstention_reason"] = "FRAME_SHAPE_CHANGE"
            output_rows.append(output)
            continue
        pair = (str(previous_row["image_relative_path"]), str(row["image_relative_path"]))
        homography_result = homography_cache.get(pair)
        if homography_result is None:
            previous_rois = [
                roi_pixels(candidate, previous.shape)
                for candidate in rois_by_index[current_index - 1]
            ]
            homography_result = estimate_background_homography(
                previous, current, previous_rois
            )
            homography_cache[pair] = homography_result
        homography, background_quality = homography_result
        if homography is None:
            output["abstention_reason"] = str(background_quality["reason"])
            output["quality"]["components"] = background_quality
            output_rows.append(output)
            continue
        rate, quality, components, reason = residual_target_rate(
            previous,
            current,
            roi_pixels(previous_row, previous.shape),
            roi_pixels(row, current.shape),
            homography,
            dt_ns / 1_000_000_000.0,
            background_quality,
        )
        output["quality"] = {"score": quality, "components": components}
        output["abstention_reason"] = reason
        output["signed_approach_rate_per_s"] = rate
        output_rows.append(output)
    atomic_jsonl(output_path, output_rows)
    reasons: dict[str, int] = defaultdict(int)
    for row in output_rows:
        reasons[str(row["abstention_reason"] or "AVAILABLE")] += 1
    receipt = {
        "schema": "blindassist.dual_loop_gmc_target_flow_producer_receipt.v1",
        "status": "COMPLETE",
        "truth_opened": False,
        "protected_events_opened": False,
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "parameter_sha256": PARAMETER_SHA256,
        "replay_input_sha256": EXPECTED_REPLAY_SHA256,
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "opencv_threads": 1,
        "wall_seconds": time.perf_counter() - started,
        "dispositions": dict(sorted(reasons.items())),
        "output_path": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.replay_input, args.image_root, args.output, args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
