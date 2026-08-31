"""X47 depth-free-space release for X46 object-permanence beliefs.

X47 preserves X46, but projects each continued belief back to the current
truth-blind RGB-D observation.  A belief terminates when its anchor-frame
center ray is inside the camera and the observed depth remains free beyond the
predicted target depth.  Occluded or out-of-view beliefs remain conservative.

The angular grid and depth slab are inherited from the RGB-D adapter.  C22 is
consumed posthoc synthetic Development and cannot confirm X47.
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


EXPERIMENT_ID = "DTR_CARLA_X47_DEPTH_FREE_SPACE_PERMANENCE_RELEASE"
ARM_X47 = "X47_ISSUED_PLAN_DEPTH_FREE_SPACE_PERMANENCE_RELEASE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x46.fixed_constants(),
        "representation": "EVIDENCE_TERMINATED_PERMANENCE_WITH_DEPTH_FREE_SPACE_RELEASE",
        "depth_release_rule": (
            "IN_FOV_BELIEF_CENTER_RAY_OBSERVED_BEYOND_PREDICTED_DEPTH_PLUS_"
            "INHERITED_NEAR_DEPTH_SLAB_TERMINATES_BELIEF"
        ),
        "depth_release_angular_grid": [
            adapter.ANGULAR_GRID_WIDTH,
            adapter.ANGULAR_GRID_HEIGHT,
        ],
        "depth_release_minimum_slab_m": adapter.MINIMUM_NEAR_SLAB_M,
        "depth_release_maximum_slab_m": adapter.MAXIMUM_NEAR_SLAB_M,
        "depth_release_slab_ratio": adapter.NEAR_SLAB_DEPTH_RATIO,
        "depth_release_threshold_source": "INHERITED_RGBD_NEAR_DEPTH_SLAB",
        "depth_release_out_of_fov_policy": "RETAIN_BELIEF",
        "depth_release_invalid_depth_policy": "RETAIN_BELIEF",
        "depth_release_class_rule": "CLASS_INDEPENDENT",
        "new_numeric_threshold_added": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def anchor_center_world(
    track: Mapping[str, Any],
    route_frame: adapter.AnchorFrame,
    camera_transform: Mapping[str, float],
) -> np.ndarray:
    forward = float(track["position_forward_m"])
    right = float(track["position_right_m"])
    center = np.asarray(route_frame.center_xy_m, dtype=np.float64)
    world_xy = (
        center
        + forward * np.asarray(route_frame.forward_xy, dtype=np.float64)
        + right * np.asarray(route_frame.right_xy, dtype=np.float64)
    )
    # The midpoint between ground origin and camera optical height is a
    # geometry-derived center ray, not a class-size or score threshold.
    world_z = (float(route_frame.z_origin_m) + float(camera_transform["z"])) / 2.0
    return np.asarray([world_xy[0], world_xy[1], world_z], dtype=np.float64)


def world_to_camera_flu(
    point_world: np.ndarray, camera_transform: Mapping[str, float]
) -> np.ndarray:
    origin = adapter.camera_flu_to_world(
        np.zeros((1, 3), dtype=np.float64), camera_transform
    )[0]
    axes = adapter.camera_flu_to_world(
        np.eye(3, dtype=np.float64), camera_transform
    ) - origin[None, :]
    return (np.asarray(point_world, dtype=np.float64) - origin) @ axes.T


def nearest_angular_grid_pixel(
    u: float, v: float, calibration: adapter.CameraCalibration
) -> tuple[int, int]:
    grid_column = min(
        adapter.ANGULAR_GRID_WIDTH - 1,
        max(0, int(math.floor(u * adapter.ANGULAR_GRID_WIDTH / calibration.width))),
    )
    grid_row = min(
        adapter.ANGULAR_GRID_HEIGHT - 1,
        max(0, int(math.floor(v * adapter.ANGULAR_GRID_HEIGHT / calibration.height))),
    )
    pixel_u = int(
        math.floor(
            (grid_column + 0.5) * calibration.width / adapter.ANGULAR_GRID_WIDTH
        )
    )
    pixel_v = int(
        math.floor(
            (grid_row + 0.5) * calibration.height / adapter.ANGULAR_GRID_HEIGHT
        )
    )
    return (
        min(calibration.width - 1, max(0, pixel_u)),
        min(calibration.height - 1, max(0, pixel_v)),
    )


def belief_center_ray(
    observation: adapter.FrameObservation,
    track: Mapping[str, Any],
    calibration: adapter.CameraCalibration,
    route_frame: adapter.AnchorFrame,
) -> tuple[int, int, float] | None:
    point_world = anchor_center_world(track, route_frame, observation.camera_transform)
    camera = world_to_camera_flu(point_world, observation.camera_transform)
    forward, left, up = (float(value) for value in camera)
    if forward <= adapter.MINIMUM_DEPTH_M:
        return None
    intrinsic = calibration.intrinsic
    u = float(intrinsic[0, 2]) - left * float(intrinsic[0, 0]) / forward
    v = float(intrinsic[1, 2]) - up * float(intrinsic[1, 1]) / forward
    if not (0.0 <= u < calibration.width and 0.0 <= v < calibration.height):
        return None
    pixel_u, pixel_v = nearest_angular_grid_pixel(u, v, calibration)
    return pixel_u, pixel_v, forward


def inherited_depth_slab_m(predicted_depth_m: float) -> float:
    return max(
        adapter.MINIMUM_NEAR_SLAB_M,
        min(
            adapter.MAXIMUM_NEAR_SLAB_M,
            adapter.NEAR_SLAB_DEPTH_RATIO * float(predicted_depth_m),
        ),
    )


def visible_free_space(
    observation: adapter.FrameObservation,
    tracks: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
    route_frame: adapter.AnchorFrame,
) -> bool:
    rays = [
        belief_center_ray(observation, track, calibration, route_frame)
        for track in tracks
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
            <= predicted_depth + inherited_depth_slab_m(predicted_depth)
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
            "x47_depth_free_space_released": True,
            "x47_depth_free_space_released_track_ids": suppressed_ids,
        }
    )


def apply_depth_release_episode(
    episode: Any,
    value: dict[str, Any],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    terminated = False
    release_frames = 0
    downstream_suppression_frames = 0
    retained_occlusion_frames = 0

    for observation, frame in zip(
        episode.observations, value["frames"], strict=True
    ):
        arm = frame["arms"][x46.ARM_X46]
        permanence_used = bool(arm.get("x46_object_permanence_used", False))
        arm["x47_depth_free_space_released"] = False
        arm["x47_depth_free_space_released_track_ids"] = []
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
        elif visible_free_space(
            observation, carriers, calibration, episode.route_frame
        ):
            suppress_permanence_arm(arm)
            terminated = True
            release_frames += 1
        else:
            retained_occlusion_frames += 1

    value["arms"][ARM_X47] = value["arms"].pop(x46.ARM_X46)
    value["diagnostics"]["x47_route_mode_counts"] = value["diagnostics"].pop(
        "x46_route_mode_counts"
    )
    value["diagnostics"]["x47_depth_free_space_release_frames"] = release_frames
    value["diagnostics"]["x47_downstream_suppression_frames"] = (
        downstream_suppression_frames
    )
    value["diagnostics"]["x47_retained_occlusion_frames"] = (
        retained_occlusion_frames
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X47] = frame["arms"].pop(x46.ARM_X46)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    return apply_depth_release_episode(
        episode,
        x46.predict_episode(episode, candidate_values, calibration),
        calibration,
    )


def self_check() -> dict[str, Any]:
    inherited = x46.self_check()
    calibration = adapter.CameraCalibration(1280, 720, 90.0, 1000.0)
    x24.require(
        nearest_angular_grid_pixel(640.0, 360.0, calibration) == (644, 364),
        "x47_angular_grid_projection",
    )
    x24.require(
        abs(inherited_depth_slab_m(5.0) - adapter.MINIMUM_NEAR_SLAB_M)
        <= x31.EPSILON,
        "x47_inherited_depth_slab",
    )
    return {
        "status": "X47_DEPTH_FREE_SPACE_PERMANENCE_RELEASE_FALSIFIER_MET",
        "x46_structural_status": inherited["status"],
        "truth_blind_rgbd_negative_evidence": True,
        "out_of_fov_retains_belief": True,
        "depth_thresholds_inherited": True,
        "class_independent": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
