"""Audit the frozen ARKitScenes visit-484248 scale-free counterexample."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from evaluate_bonn_rgbd_consumed import (
    DIRECTIONS,
    advance,
    sha256,
    summarize_sequence,
    write_new,
)
from produce_external_rgb_metric_depth_observations import DepthAnythingV2MetricSource
from scipy import ndimage, stats

from core import BANDS, ROI_Y, score_relative_intrusion

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "docs/research/hftf/SCALE_FREE_TRAVERSABILITY_R2_ARKITSCENES_COUNTEREXAMPLE_AUDIT_R0_PROTOCOL_2026-08-04.json"
EXPECTED_VISIT = "484248"
EXPECTED_VIDEO = "48458617"
ORIENTATION_NAMES = ("upright", "left", "upside_down", "right")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bounds(length: int, start: float, end: float) -> tuple[int, int]:
    return round(length * start), round(length * end)


def trajectory_pose(line: str) -> tuple[float, np.ndarray]:
    """Independent equivalent of official ARKitScenes TrajStringToMatrix."""
    values = np.asarray([float(token) for token in line.split()], dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("invalid ARKitScenes trajectory row")
    rotation, _ = cv2.Rodrigues(values[1:4])
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = values[4:7]
    return float(values[0]), np.linalg.inv(extrinsics)


def official_orientation_index(camera_to_world: np.ndarray) -> int:
    z_vector = np.asarray(camera_to_world, dtype=np.float64)[2, :3]
    orientations = np.asarray(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    return int(np.argmax(orientations @ z_vector))


def rotate_by_orientation(values: np.ndarray, orientation_index: int) -> np.ndarray:
    if orientation_index == 0:
        return values.copy()
    if orientation_index == 1:
        return cv2.rotate(values, cv2.ROTATE_90_CLOCKWISE)
    if orientation_index == 2:
        return cv2.rotate(values, cv2.ROTATE_180)
    if orientation_index == 3:
        return cv2.rotate(values, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("invalid orientation index")


def pose_diagnostics(camera_to_world: np.ndarray, intrinsics: np.ndarray) -> tuple[float, float]:
    """Return stored-image up angle (clockwise from top) and optical tilt."""
    up_camera = camera_to_world[:3, :3].T @ np.asarray([0.0, 1.0, 0.0])
    up_camera /= np.linalg.norm(up_camera)
    image_dx = float(intrinsics[0, 0] * up_camera[0])
    image_dy = float(intrinsics[1, 1] * up_camera[1])
    angle = math.degrees(math.atan2(image_dx, -image_dy))
    tilt = abs(math.degrees(math.asin(float(np.clip(up_camera[2], -1.0, 1.0)))))
    return angle, tilt


def read_intrinsics(path: Path, shape: tuple[int, int]) -> np.ndarray:
    width, height, fx, fy, cx, cy = np.loadtxt(path, dtype=np.float64).tolist()
    if (int(width), int(height)) != (shape[1], shape[0]):
        raise ValueError(f"intrinsics/image mismatch: {path}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def dense_reconstructions(
    depth_raw: np.ndarray, confidence: np.ndarray, contract: dict[str, Any]
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, float, float]:
    depth = depth_raw.astype(np.float64) / 1000.0
    low, high = map(float, contract["valid_depth_m"])
    valid = (confidence == int(contract["confidence_value"])) & np.isfinite(depth) & (depth >= low) & (depth <= high)
    fraction = float(np.mean(valid))
    if fraction < float(contract["minimum_source_valid_fraction"]):
        return None, None, valid, fraction, float("nan")
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    global_dense = depth[tuple(nearest)]
    _height, width = depth.shape
    local_dense = np.full_like(depth, np.nan)
    cross, filled = 0, 0
    for start, end in BANDS.values():
        x0, x1 = bounds(width, start, end)
        local_valid = valid[:, x0:x1]
        target_invalid = ~local_valid
        source_x = nearest[1][:, x0:x1]
        filled += int(np.sum(target_invalid))
        cross += int(np.sum(target_invalid & ((source_x < x0) | (source_x >= x1))))
        if not np.any(local_valid):
            continue
        local_nearest = ndimage.distance_transform_edt(~local_valid, return_distances=False, return_indices=True)
        local_dense[:, x0:x1] = depth[:, x0:x1][tuple(local_nearest)]
    return global_dense, local_dense, valid, fraction, cross / max(1, filled)


def fit_dominant_plane(
    depth: np.ndarray, source_valid: np.ndarray, intrinsics: np.ndarray,
    up_camera: np.ndarray, config: dict[str, Any], seed: int
) -> dict[str, Any]:
    height, width = depth.shape
    y0, y1 = bounds(height, *config["roi_y"])
    x0, x1 = bounds(width, *config["roi_x"])
    stride = int(config["sample_stride_px"])
    yy, xx = np.mgrid[y0:y1:stride, x0:x1:stride]
    zz = depth[yy, xx]
    keep = source_valid[yy, xx] & np.isfinite(zz) & (zz > 0.0)
    yy, xx, zz = yy[keep], xx[keep], zz[keep]
    if len(zz) < int(config["minimum_points"]):
        return {"status": "UNKNOWN", "reason": "INSUFFICIENT_SOURCE_POINTS", "point_count": len(zz)}
    points = np.column_stack(((xx - intrinsics[0, 2]) * zz / intrinsics[0, 0], (yy - intrinsics[1, 2]) * zz / intrinsics[1, 1], zz))
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), dtype=bool)
    threshold = float(config["inlier_threshold_m"])
    for _ in range(int(config["ransac_iterations"])):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        inliers = np.abs(points @ normal - sample[0] @ normal) <= threshold
        if int(np.sum(inliers)) > int(np.sum(best)):
            best = inliers
    if int(np.sum(best)) < 3:
        return {"status": "UNKNOWN", "reason": "PLANE_FIT_FAILED", "point_count": len(points)}
    center = np.mean(points[best], axis=0)
    _, _, vh = np.linalg.svd(points[best] - center, full_matrices=False)
    normal = vh[-1]
    gravity_alignment = abs(float(normal @ up_camera))
    residual = np.abs((points - center) @ normal)
    best = residual <= threshold
    band_fractions = {}
    for name, (start, end) in BANDS.items():
        bx0, bx1 = bounds(width, start, end)
        membership = (xx >= bx0) & (xx < bx1)
        band_fractions[name] = float(np.mean(best[membership])) if np.any(membership) else 0.0
    return {
        "status": "VALID",
        "point_count": len(points),
        "inlier_count": int(np.sum(best)),
        "roi_inlier_fraction": float(np.mean(best)),
        "band_inlier_fractions": band_fractions,
        "same_plane_all_bands": all(value >= float(config["same_plane_minimum_inlier_fraction_each_band"]) for value in band_fractions.values()),
        "large_plane": float(np.mean(best)) >= float(config["large_plane_roi_inlier_fraction"]),
        "median_inlier_depth_m": float(np.median(zz[best])),
        "near_plane": float(np.median(zz[best])) <= float(config["near_plane_median_depth_m_maximum"]),
        "normal_gravity_alignment": gravity_alignment,
        "plane_orientation": (
            "HORIZONTAL"
            if gravity_alignment >= float(config["horizontal_normal_gravity_alignment_minimum"])
            else "VERTICAL"
            if gravity_alignment <= float(config["vertical_normal_gravity_alignment_maximum"])
            else "SLANTED"
        ),
        "normal_camera": normal.tolist(),
    }


def score_equal(left: dict[str, Any], right: dict[str, Any], tolerance: float = 2e-5) -> bool:
    if left.get("status") != right.get("status") or left.get("reason") != right.get("reason"):
        return False
    if left.get("status") != "VALID":
        return True
    return all(abs(float(left["scores"][name]) - float(right["scores"][name])) <= tolerance for name in BANDS)


def label(decision: dict[str, Any]) -> str:
    return str(decision.get("label", decision.get("reason", decision.get("status"))))


def ambiguity_class(candidate: dict[str, Any], truth: dict[str, Any]) -> str:
    c, t = label(candidate), label(truth)
    if c == "AMBIGUOUS" and t == "AMBIGUOUS":
        return "REASONABLE_PROXY"
    if c == "AMBIGUOUS" and t in DIRECTIONS:
        return "WRONG_REFUSAL_PROXY"
    if c in DIRECTIONS and t == "AMBIGUOUS":
        return "OVER_ANSWER_PROXY"
    return "OTHER"


def route_implication(orientation_counts: Counter[str], frame_count: int) -> str:
    if orientation_counts.get("upright", 0) < frame_count:
        return "DO_NOT_USE_UNRECTIFIED_OUTPUT_AS_AUXILIARY_OR_FALLBACK"
    return "NO_ROUTE_ROLE_DECISION_FROM_THIS_CONSUMED_VISIT_ALONE"


def depth_fit(candidate: np.ndarray, sensor: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    paired = valid & np.isfinite(candidate) & (candidate > 0.0) & np.isfinite(sensor) & (sensor > 0.0)
    if int(np.sum(paired)) < 3:
        return {"status": "UNKNOWN", "paired_pixel_count": int(np.sum(paired))}
    log_scale = float(np.median(np.log(sensor[paired]) - np.log(candidate[paired])))
    scaled = candidate[paired] * math.exp(log_scale)
    relative = np.abs(scaled - sensor[paired]) / sensor[paired]
    correlation = float(stats.spearmanr(candidate[paired], sensor[paired]).statistic)
    return {
        "status": "VALID", "paired_pixel_count": int(np.sum(paired)), "log_scale": log_scale,
        "scale": math.exp(log_scale), "post_scale_abs_rel_median": float(np.median(relative)),
        "post_scale_abs_rel_p90": float(np.quantile(relative, 0.9)), "spearman_spatial_rank": correlation,
    }


def colored_depth(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    use = np.isfinite(depth) & (depth > 0.0)
    if valid is not None:
        use &= valid
    out = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(use):
        lo, hi = np.quantile(depth[use], [0.02, 0.98])
        out[use] = np.clip((depth[use] - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(255 - out, cv2.COLORMAP_TURBO)


def band_overlay(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    y0, y1 = bounds(height, *ROI_Y)
    cv2.rectangle(result, (bounds(width, .05, .95)[0], y0), (bounds(width, .05, .95)[1] - 1, y1 - 1), (255, 255, 255), 1)
    for color, (name, (start, end)) in zip(((255, 80, 80), (80, 255, 80), (80, 80, 255)), BANDS.items()):
        x0, x1 = bounds(width, start, end)
        cv2.rectangle(result, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        cv2.putText(result, name[0].upper(), (x0 + 3, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return result


def make_figure(records: list[dict[str, Any]], images: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    categories = ("WRONG_REFUSAL_PROXY", "OVER_ANSWER_PROXY", "OTHER")
    chosen = []
    for category in categories:
        candidates = [row for row in records if row["ambiguity_class"] == category]
        if category == "OTHER":
            candidates = [row for row in candidates if row["candidate_label"] in DIRECTIONS and row["truth_label"] in DIRECTIONS and row["candidate_label"] != row["truth_label"]]
        if candidates:
            chosen.append(max(candidates, key=lambda row: row.get("decision_separation", 0.0)))
    if not chosen:
        chosen = [records[len(records) // 2]]
    panels = []
    for record in chosen:
        rgb, da, truth = images[record["frame_index"]]
        upright = rotate_by_orientation(rgb, record["orientation_index"])
        row_panels = [band_overlay(rgb), band_overlay(upright), colored_depth(da), colored_depth(truth)]
        resized = [cv2.resize(panel, (320, 240), interpolation=cv2.INTER_AREA) for panel in row_panels]
        caption = f"f{record['frame_index']} {record['candidate_label']} / {record['truth_label']}"
        for index, panel in enumerate(resized):
            title = ("stored bands", "official upright", "DA depth", "sensor nearest")[index]
            cv2.rectangle(panel, (0, 0), (319, 35), (0, 0, 0), -1)
            cv2.putText(panel, title, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, caption, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.hstack(resized))
    if not cv2.imwrite(str(path), np.vstack(panels)):
        raise OSError(f"failed to write {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dav2-repo", required=True, type=Path)
    parser.add_argument("--dav2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    protocol = load_json(PROTOCOL_PATH)
    if protocol["status"] != "FROZEN_BEFORE_COUNTEREXAMPLE_MECHANISM_OUTPUT_EXECUTION":
        raise ValueError("counterexample audit protocol is not frozen")
    source_result_path = REPO_ROOT / protocol["source_result"]["path"]
    source_frames_path = source_result_path.parent / "frames.jsonl"
    if sha256(source_result_path) != protocol["source_result"]["sha256"] or sha256(source_frames_path) != protocol["source_result"]["frame_ledger_sha256"]:
        raise ValueError("immutable R2 evidence identity failure")
    source_result = load_json(source_result_path)
    if source_result["terminal"] != protocol["source_result"]["immutable_terminal"]:
        raise ValueError("immutable R2 terminal mismatch")
    if sha256(args.dav2_checkpoint) != protocol["candidate"]["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint identity failure")
    all_source_rows = [json.loads(line) for line in source_frames_path.read_text(encoding="utf-8").splitlines()]
    source_rows = [row for row in all_source_rows if str(row["sequence_id"]) == EXPECTED_VISIT]
    if len(source_rows) != 150 or any(str(row["video_id"]) != EXPECTED_VIDEO for row in source_rows):
        raise ValueError("visit 484248 ledger identity failure")
    if [int(row["frame_index"]) for row in source_rows] != list(range(150)):
        raise ValueError("visit ledger is not contiguous")

    video_root = args.dataset_root / EXPECTED_VIDEO
    trajectory = [trajectory_pose(line) for line in (video_root / "lowres_wide.traj").read_text(encoding="utf-8").splitlines()]
    intrinsics_files = sorted((video_root / "lowres_wide_intrinsics").glob("*.pincam"))
    intrinsics_by_time = [(float(path.stem.rsplit("_", 1)[-1]), path) for path in intrinsics_files]
    source = DepthAnythingV2MetricSource(args.dav2_repo, args.dav2_checkpoint, args.device, input_size=518, precision="fp16" if args.device.startswith("cuda") else "fp32")
    candidate_history: list[dict[str, float]] = []
    rotated_candidate_history: list[dict[str, float]] = []
    rotated_truth_history: list[dict[str, float]] = []
    ledger, image_cache = [], {}
    for source_row in source_rows:
        frame_index = int(source_row["frame_index"])
        rgb_path = args.dataset_root / source_row["rgb_relative_path"]
        depth_path = args.dataset_root / source_row["depth_relative_path"]
        confidence_path = depth_path.parent.parent / "confidence" / depth_path.name
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth_raw is None or confidence is None:
            raise OSError(f"failed to decode frame {frame_index}")
        timestamp = float(source_row["timestamp_s"])
        pose_time, pose = min(trajectory, key=lambda row: (abs(row[0] - timestamp), row[0]))
        pose_delta = abs(pose_time - timestamp)
        orientation_index = official_orientation_index(pose)
        intrinsics_time, intrinsics_path = min(intrinsics_by_time, key=lambda row: (abs(row[0] - timestamp), row[0]))
        if abs(intrinsics_time - timestamp) > .0015:
            raise ValueError("intrinsics timestamp mismatch")
        intrinsics = read_intrinsics(intrinsics_path, depth_raw.shape)
        gravity_angle, optical_tilt = pose_diagnostics(pose, intrinsics)
        up_camera = pose[:3, :3].T @ np.asarray([0.0, 1.0, 0.0])
        up_camera /= np.linalg.norm(up_camera)
        truth_global, truth_local, source_valid, source_fraction, cross_band_fill = dense_reconstructions(depth_raw, confidence, protocol["truth_reconstruction"])
        da_depth, _ = source.infer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {})
        candidate_score = score_relative_intrusion(da_depth)
        candidate_decision, candidate_history = advance(candidate_score, candidate_history)
        if not score_equal(candidate_score, source_row["candidate_score"]) or label(candidate_decision) != label(source_row["candidate_decision"]):
            raise ValueError(f"frozen candidate replay mismatch at frame {frame_index}")
        if truth_global is None:
            local_score = {"status": "UNKNOWN", "reason": "INSUFFICIENT_CONFIDENCE2_SOURCE_FRACTION"}
            plane = {"status": "UNKNOWN", "reason": "INSUFFICIENT_CONFIDENCE2_SOURCE_FRACTION"}
            fit = {"status": "UNKNOWN", "paired_pixel_count": 0}
            rotated_candidate_score = score_relative_intrusion(rotate_by_orientation(da_depth, orientation_index))
            rotated_truth_score = local_score
        else:
            local_score = score_relative_intrusion(truth_local)
            plane = fit_dominant_plane(depth_raw.astype(np.float64) / 1000.0, source_valid, intrinsics, up_camera, protocol["mechanism_diagnostics"]["dominant_plane"], int(protocol["mechanism_diagnostics"]["dominant_plane"]["ransac_seed"]) + frame_index)
            height, width = source_valid.shape
            y0, y1 = bounds(height, *ROI_Y); x0, x1 = bounds(width, .05, .95)
            roi_valid = np.zeros_like(source_valid); roi_valid[y0:y1, x0:x1] = source_valid[y0:y1, x0:x1]
            fit = depth_fit(da_depth, depth_raw.astype(np.float64) / 1000.0, roi_valid)
            rotated_candidate_score = score_relative_intrusion(rotate_by_orientation(da_depth, orientation_index))
            rotated_truth_score = score_relative_intrusion(rotate_by_orientation(truth_global, orientation_index))
        rotated_candidate_decision, rotated_candidate_history = advance(rotated_candidate_score, rotated_candidate_history)
        rotated_truth_decision, rotated_truth_history = advance(rotated_truth_score, rotated_truth_history)
        candidate_scores = source_row["candidate_score"].get("scores", {})
        truth_scores = source_row["truth_score"].get("scores", {})
        order_agreement = None
        opposite_order = None
        if set(candidate_scores) == set(BANDS) and set(truth_scores) == set(BANDS):
            names = list(BANDS)
            cwin = names[int(np.argmin([candidate_scores[name] for name in names]))]
            twin = names[int(np.argmin([truth_scores[name] for name in names]))]
            order_agreement = cwin == twin
            opposite_order = {cwin, twin} == {"left", "right"}
        record = {
            "schema": "blindassist_scale_free_traversability_r2_counterexample_frame_v1", "visit_id": EXPECTED_VISIT,
            "video_id": EXPECTED_VIDEO, "frame_index": frame_index, "timestamp_s": timestamp,
            "pose_timestamp_difference_s": pose_delta, "pose_supported": pose_delta <= float(protocol["mechanism_diagnostics"]["pose"]["maximum_timestamp_difference_s"]),
            "orientation_index": orientation_index, "orientation_name": ORIENTATION_NAMES[orientation_index],
            "gravity_angle_stored_image_degrees": gravity_angle, "absolute_optical_axis_tilt_degrees": optical_tilt,
            "truth_source_valid_fraction": source_fraction, "cross_band_nearest_fill_fraction": cross_band_fill,
            "global_vs_band_local_score_order_agreement": (
                None if truth_global is None or local_score.get("status") != "VALID"
                else int(np.argmin([source_row["truth_score"]["scores"][name] for name in BANDS]))
                == int(np.argmin([local_score["scores"][name] for name in BANDS]))
            ),
            "candidate_label": label(source_row["candidate_decision"]), "truth_label": label(source_row["truth_decision"]),
            "ambiguity_class": ambiguity_class(source_row["candidate_decision"], source_row["truth_decision"]),
            "candidate_truth_band_order_agreement": order_agreement, "candidate_truth_opposite_band_order": opposite_order,
            "band_local_truth_score": local_score, "dominant_plane": plane, "da_vs_sensor": fit,
            "orientation_counterfactual": {"candidate_score": rotated_candidate_score, "candidate_decision": rotated_candidate_decision, "truth_score": rotated_truth_score, "truth_decision": rotated_truth_decision, "authority": "EXPLANATORY_ONLY"},
            "decision_separation": max(float(source_row["candidate_decision"].get("margin_log_units", 0.0)), float(source_row["truth_decision"].get("margin_log_units", 0.0))),
        }
        ledger.append(record)
        image_cache[frame_index] = (bgr, da_depth, truth_global if truth_global is not None else np.full(depth_raw.shape, np.nan))

    orientation_counts = Counter(row["orientation_name"] for row in ledger)
    ambiguity_counts = Counter(row["ambiguity_class"] for row in ledger)
    pose_supported = [row for row in ledger if row["pose_supported"]]
    rotated_rows = [{"frame_index": row["frame_index"], "candidate_score": row["orientation_counterfactual"]["candidate_score"], "candidate_decision": row["orientation_counterfactual"]["candidate_decision"], "truth_score": row["orientation_counterfactual"]["truth_score"], "truth_decision": row["orientation_counterfactual"]["truth_decision"]} for row in ledger]
    rotated_summary = summarize_sequence(EXPECTED_VISIT, rotated_rows)
    fits = [row["da_vs_sensor"] for row in ledger if row["da_vs_sensor"].get("status") == "VALID"]
    planes = [row["dominant_plane"] for row in ledger if row["dominant_plane"].get("status") == "VALID"]
    result = {
        "schema": "blindassist_scale_free_traversability_r2_counterexample_audit_result_v1",
        "status": "COUNTEREXAMPLE_MECHANISM_AUDIT_COMPLETE_DEVELOPMENT_ONLY", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH), "source_result_sha256": sha256(source_result_path), "source_frame_ledger_sha256": sha256(source_frames_path),
        "immutable_source_terminal": source_result["terminal"], "visit_id": EXPECTED_VISIT, "video_id": EXPECTED_VIDEO, "frame_count": len(ledger),
        "orientation": {"official_orientation_counts": dict(orientation_counts), "pose_supported_frames": len(pose_supported),
            "gravity_angle_stored_image_degrees_median": float(np.median([row["gravity_angle_stored_image_degrees"] for row in pose_supported])),
            "absolute_optical_axis_tilt_degrees_median": float(np.median([row["absolute_optical_axis_tilt_degrees"] for row in pose_supported])),
            "finding": "STORED_COLUMNS_DO_NOT_REPRESENT_PHYSICAL_LEFT_CENTER_RIGHT" if orientation_counts.get("upright", 0) < 75 else "MOSTLY_UPRIGHT"},
        "ambiguity_proxy_counts": dict(ambiguity_counts),
        "source_and_fill": {"source_valid_fraction_quantiles": np.quantile([row["truth_source_valid_fraction"] for row in ledger], [0, .05, .5, .95, 1]).tolist(),
            "cross_band_nearest_fill_fraction_median": float(np.nanmedian([row["cross_band_nearest_fill_fraction"] for row in ledger])),
            "global_vs_band_local_order_agreement_fraction": float(np.mean([row["global_vs_band_local_score_order_agreement"] for row in ledger if row["global_vs_band_local_score_order_agreement"] is not None]))},
        "dominant_plane": {"valid_frames": len(planes), "large_plane_frames": sum(bool(row["large_plane"]) for row in planes), "same_plane_all_bands_frames": sum(bool(row["same_plane_all_bands"]) for row in planes), "near_plane_frames": sum(bool(row["near_plane"]) for row in planes), "orientation_counts": dict(Counter(row["plane_orientation"] for row in planes))},
        "da_vs_sensor": {"valid_frames": len(fits), "global_scale_median": float(np.median([row["scale"] for row in fits])),
            "post_scale_abs_rel_median_frame_macro": float(np.median([row["post_scale_abs_rel_median"] for row in fits])),
            "post_scale_abs_rel_p90_frame_macro": float(np.median([row["post_scale_abs_rel_p90"] for row in fits])),
            "spearman_spatial_rank_median": float(np.nanmedian([row["spearman_spatial_rank"] for row in fits])),
            "per_frame_band_order_agreement_fraction": float(np.mean([row["candidate_truth_band_order_agreement"] for row in ledger if row["candidate_truth_band_order_agreement"] is not None])),
            "per_frame_opposite_band_order_fraction": float(np.mean([row["candidate_truth_opposite_band_order"] for row in ledger if row["candidate_truth_opposite_band_order"] is not None]))},
        "orientation_normalized_counterfactual": {**rotated_summary, "authority": "EXPLANATORY_ONLY_NEVER_REPLACES_R2"},
        "mechanism_conclusion": "HANDHELD_ORIENTATION_PLUS_CLOSE_PLANAR_GEOMETRY_AND_SPATIAL_RANK_ERROR; NOT_GLOBAL_SCALE_ONLY",
        "route_implication": route_implication(orientation_counts, len(ledger)),
        "allowed_route_decisions": protocol["allowed_route_decisions"],
        "final_candidate_role": "REQUIRES_CROSS_EVIDENCE_SYNTHESIS",
        "claim_ceiling": protocol["claim_ceiling"], "production_or_safety_claim_allowed": False,
    }
    args.output_root.mkdir(parents=True)
    ledger_path = args.output_root / "frame_mechanism_ledger.jsonl"
    write_new(ledger_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger).encode())
    figure_path = args.output_root / "representative_counterexamples.png"
    make_figure(ledger, image_cache, figure_path)
    result["frame_mechanism_ledger_sha256"] = sha256(ledger_path)
    result["representative_counterexamples_sha256"] = sha256(figure_path)
    write_new(args.output_root / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
