"""X50 footprint-supported depth release for object permanence.

X50 preserves X46 object permanence, but strengthens its RGB-D negative
evidence.  A propagated belief is released only when the current depth image
observes free space beyond the belief at both its footprint centroid and every
footprint vertex.  A single occluded, invalid, or out-of-view support ray keeps
the belief alive.

The angular grid and depth slab remain inherited from the RGB-D adapter.  C22
is consumed posthoc synthetic Development and cannot confirm X50.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_rgbd_model_adapter as adapter  # noqa: E402
import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x46_evidence_terminated_object_permanence as x46  # noqa: E402
import dtr_carla_x47_depth_free_space_permanence_release as x47  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X50_FOOTPRINT_SUPPORTED_DEPTH_RELEASE"
ARM_X50 = "X50_ISSUED_PLAN_FOOTPRINT_SUPPORTED_DEPTH_RELEASE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x47.fixed_constants(),
        "representation": "OBJECT_PERMANENCE_WITH_FOOTPRINT_SUPPORTED_DEPTH_RELEASE",
        "depth_release_support": "PROPAGATED_FOOTPRINT_CENTROID_AND_ALL_VERTICES",
        "depth_release_rule": (
            "ALL_SUPPORT_RAYS_IN_FOV_AND_OBSERVED_BEYOND_PREDICTED_DEPTH_PLUS_"
            "INHERITED_NEAR_DEPTH_SLAB"
        ),
        "partial_free_space_policy": "RETAIN_BELIEF",
        "support_ray_count_threshold": None,
        "new_numeric_threshold_added": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def footprint_support_points(track: Mapping[str, Any]) -> np.ndarray:
    footprint = np.asarray(track["footprint_xy"], dtype=np.float64).reshape(-1, 2)
    x24.require(len(footprint) >= 3, "x50_footprint_support")
    centroid = np.mean(footprint, axis=0, keepdims=True)
    return np.concatenate((centroid, footprint), axis=0)


def support_ray(
    observation: adapter.FrameObservation,
    point_xy: Sequence[float],
    calibration: adapter.CameraCalibration,
    route_frame: adapter.AnchorFrame,
) -> tuple[int, int, float] | None:
    forward, right = (float(value) for value in point_xy)
    center = np.asarray(route_frame.center_xy_m, dtype=np.float64)
    world_xy = (
        center
        + forward * np.asarray(route_frame.forward_xy, dtype=np.float64)
        + right * np.asarray(route_frame.right_xy, dtype=np.float64)
    )
    world_z = (
        float(route_frame.z_origin_m) + float(observation.camera_transform["z"])
    ) / 2.0
    camera = x47.world_to_camera_flu(
        np.asarray([world_xy[0], world_xy[1], world_z], dtype=np.float64),
        observation.camera_transform,
    )
    depth, left, up = (float(value) for value in camera)
    if depth <= adapter.MINIMUM_DEPTH_M:
        return None
    intrinsic = calibration.intrinsic
    u = float(intrinsic[0, 2]) - left * float(intrinsic[0, 0]) / depth
    v = float(intrinsic[1, 2]) - up * float(intrinsic[1, 1]) / depth
    if not (0.0 <= u < calibration.width and 0.0 <= v < calibration.height):
        return None
    pixel_u, pixel_v = x47.nearest_angular_grid_pixel(u, v, calibration)
    return pixel_u, pixel_v, depth


def footprint_visible_free_space(
    observation: adapter.FrameObservation,
    tracks: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
    route_frame: adapter.AnchorFrame,
) -> bool:
    rays = [
        support_ray(observation, point, calibration, route_frame)
        for track in tracks
        for point in footprint_support_points(track)
    ]
    if not rays or any(ray is None for ray in rays):
        return False
    depth_m = adapter.load_depth_m(observation, calibration)
    for ray in rays:
        assert ray is not None
        pixel_u, pixel_v, predicted_depth = ray
        observed_depth = float(depth_m[pixel_v, pixel_u])
        if (
            not math.isfinite(observed_depth)
            or observed_depth <= adapter.MINIMUM_DEPTH_M
            or observed_depth >= calibration.depth_max_m
            or observed_depth
            <= predicted_depth + x47.inherited_depth_slab_m(predicted_depth)
        ):
            return False
    return True


def suppress_permanence_arm(arm: dict[str, Any]) -> None:
    suppressed_ids = sorted(str(value) for value in arm["confirmed_risk_track_ids"])
    arm.update(
        {
            "route_risk": False,
            "minimum_entry_s": None,
            "candidate_risk_track_ids": [],
            "confirmed_risk_track_ids": [],
            "candidate_risk_parent_track_ids": [],
            "confirmed_risk_parent_track_ids": [],
            "x50_footprint_free_space_released": True,
            "x50_footprint_free_space_released_track_ids": suppressed_ids,
        }
    )


def apply_footprint_depth_release_episode(
    episode: Any,
    value: dict[str, Any],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    value = copy.deepcopy(value)
    terminated = False
    release_frames = 0
    downstream_suppression_frames = 0
    retained_occlusion_frames = 0

    for observation, frame in zip(
        episode.observations, value["frames"], strict=True
    ):
        arm = frame["arms"][x46.ARM_X46]
        permanence_used = bool(arm.get("x46_object_permanence_used", False))
        arm["x50_footprint_free_space_released"] = False
        arm["x50_footprint_free_space_released_track_ids"] = []
        if not permanence_used:
            terminated = False
            continue
        tracks_by_id = {str(row["track_id"]): row for row in frame["tracks"]}
        carriers = [
            tracks_by_id[str(track_id)]
            for track_id in arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in tracks_by_id
        ]
        if terminated:
            suppress_permanence_arm(arm)
            downstream_suppression_frames += 1
        elif footprint_visible_free_space(
            observation, carriers, calibration, episode.route_frame
        ):
            suppress_permanence_arm(arm)
            terminated = True
            release_frames += 1
        else:
            retained_occlusion_frames += 1

    value["arms"][ARM_X50] = value["arms"].pop(x46.ARM_X46)
    value["diagnostics"]["x50_route_mode_counts"] = value["diagnostics"].pop(
        "x46_route_mode_counts"
    )
    value["diagnostics"]["x50_footprint_free_space_release_frames"] = release_frames
    value["diagnostics"]["x50_downstream_suppression_frames"] = (
        downstream_suppression_frames
    )
    value["diagnostics"]["x50_retained_occlusion_frames"] = retained_occlusion_frames
    for frame in value["frames"]:
        frame["arms"][ARM_X50] = frame["arms"].pop(x46.ARM_X46)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    return apply_footprint_depth_release_episode(
        episode,
        x46.predict_episode(episode, candidate_values, calibration),
        calibration,
    )


def self_check() -> dict[str, Any]:
    inherited = x47.self_check()
    points = footprint_support_points(
        {"footprint_xy": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]}
    )
    x24.require(
        points.shape == (5, 2) and np.allclose(points[0], [1.0, 1.0]),
        "x50_centroid_and_vertices",
    )
    return {
        "status": "X50_FOOTPRINT_SUPPORTED_DEPTH_RELEASE_FALSIFIER_MET",
        "x47_structural_status": inherited["status"],
        "centroid_and_all_vertices_required": True,
        "partial_free_space_retains_belief": True,
        "depth_thresholds_inherited": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
