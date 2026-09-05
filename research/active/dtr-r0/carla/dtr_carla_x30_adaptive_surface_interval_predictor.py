"""Freeze and predict X30 adaptive-surface contact-interval risk.

X30 changes the observation representation rather than detector thresholds.  A
fixed 32 x 32 sampling budget is allocated inside every detector mask bounding
box, so a small distant actor is not discarded by the global 160 x 90 angular
grid before it reaches metric tracking.  The samples are unprojected through
the frozen RGB-D calibration into the route frame and then consumed by X29's
temporal occupancy-lineage authority.

Risk is an interval, not only an inward boundary crossing: a rigid/static
authorized footprint that is already inside the route tube remains a candidate
even after relative closing speed changes sign.  X29's frozen contradiction
latch is also enforced literally; a conflicted track cannot fall back to the
older occupancy authority.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_rgbd_model_adapter as adapter  # noqa: E402
import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x24_plan_route_core as route  # noqa: E402
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x29_temporal_occupancy_lineage_predictor as x29  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X30_ADAPTIVE_SURFACE_CONTACT_INTERVAL"
FREEZE_SCHEMA = "blindassist-dtr-carla-x30-adaptive-surface-interval-freeze-v1"
PREDICTION_SCHEMA = (
    "blindassist-dtr-carla-x30-adaptive-surface-interval-predictions-v1"
)
ARM_X30 = "X30_ISSUED_PLAN_ADAPTIVE_SURFACE_CONTACT_INTERVAL"

ADAPTIVE_GRID_SIDE = 32
ADAPTIVE_GRID_MAXIMUM_SAMPLES = ADAPTIVE_GRID_SIDE * ADAPTIVE_GRID_SIDE
WORLD_COMPONENT_CLASS_ID = -1
WORLD_COMPONENT_CLASS_NAME = "WORLD_OCCUPANCY_COMPONENT"
EPSILON = 1e-9


@dataclass(frozen=True)
class ComponentMeasurement:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    position_xy: np.ndarray
    footprint_xy: np.ndarray
    surface_points_xy: np.ndarray
    depth_support: int
    semantic_labels: tuple[str, ...]
    branch_count: int


def fixed_constants() -> dict[str, Any]:
    return {
        **x29.fixed_constants(),
        "representation": "SET_VALUED_WORLD_COMPONENT_SURFACE_LINEAGE",
        "candidate_surface_sampling": "FIXED_BBOX_NORMALIZED_PIXEL_LATTICE",
        "adaptive_grid_side": ADAPTIVE_GRID_SIDE,
        "adaptive_grid_maximum_samples": ADAPTIVE_GRID_MAXIMUM_SAMPLES,
        "minimum_adaptive_foreground_samples": adapter.MINIMUM_MASK_DEPTH_POINTS,
        "surface_coordinate_frame": "ANCHOR_FORWARD_RIGHT",
        "surface_depth_rule": "MASK_Q15_NEAR_SLAB",
        "within_frame_component_rule": (
            "TRANSITIVE_NONEMPTY_WORLD_LATTICE_CELL_OVERLAP"
        ),
        "semantic_label_policy": "PROVENANCE_LABEL_SET_NOT_IDENTITY_GATE",
        "temporal_association_class": WORLD_COMPONENT_CLASS_NAME,
        "insufficient_transport_rule": (
            "FEWER_THAN_INHERITED_MINIMUM_PAIRS_DOES_NOT_BLOCK_OCCUPANCY_BRANCH"
        ),
        "risk_geometry": "AUTHORIZED_FOOTPRINT_ROUTE_CONTACT_INTERVAL",
        "already_inside_route_tube_rule": (
            "CURRENT_OR_FUTURE_INTERVAL_OVERLAP_IS_RISK_WITHOUT_CLOSING_SIGN_GATE"
        ),
        "future_outside_to_inside_rule": (
            "FIRST_CAUSAL_TUBE_ENTRY_WITH_INHERITED_MINIMUM_CLOSING_SPEED"
        ),
        "transport_reset_enforcement": (
            "CONFLICTED_TRACK_CANNOT_FALL_BACK_TO_OCCUPANCY_AUTHORITY"
        ),
    }


def paths(run_root: Path) -> dict[str, Path]:
    return {
        "x24_freeze": run_root / "freeze-x24.json",
        "x24_predictions": run_root / "predictions-x24.json",
        "freeze": run_root / "freeze-x30.json",
        "predictions": run_root / "predictions-x30.json",
    }


def _sample_axis(low: float, high: float, size: int) -> np.ndarray:
    x24.require(size > 0, "x30_axis_size")
    left = max(0, min(size - 1, int(math.floor(float(low) * (size - 1)))))
    right = max(left, min(size - 1, int(math.ceil(float(high) * (size - 1)))))
    count = right - left + 1
    if count <= ADAPTIVE_GRID_SIDE:
        return np.arange(left, right + 1, dtype=np.int32)
    return np.unique(
        np.rint(np.linspace(left, right, ADAPTIVE_GRID_SIDE)).astype(np.int32)
    )


def candidate_conditioned_pixels(
    candidate: Mapping[str, Any], width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    bbox = tuple(float(value) for value in candidate["bbox_xyxy_normalized"])
    x24.require(len(bbox) == 4, "x30_bbox_length")
    x1, y1, x2, y2 = bbox
    x24.require(
        all(math.isfinite(value) for value in bbox)
        and 0.0 <= x1 <= x2 <= 1.0
        and 0.0 <= y1 <= y2 <= 1.0,
        "x30_bbox_range",
    )
    columns = _sample_axis(x1, x2, width)
    rows = _sample_axis(y1, y2, height)
    vv, uu = np.meshgrid(rows, columns, indexing="ij")
    x24.require(
        0 < uu.size <= ADAPTIVE_GRID_MAXIMUM_SAMPLES,
        "x30_adaptive_sample_count",
    )
    return uu, vv


def adaptive_mask_footprint_measurement(
    candidate: Mapping[str, Any],
    depth_m: np.ndarray,
    calibration: adapter.CameraCalibration,
    camera_transform: Mapping[str, Any],
    route_frame: adapter.AnchorFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    mask = x24.normalized_polygon_mask(
        candidate, calibration.width, calibration.height
    )
    depth = np.asarray(depth_m, dtype=np.float64)
    expected = (calibration.height, calibration.width)
    x24.require(mask.shape == expected and depth.shape == expected, "x30_rgbd_shape")
    uu, vv = candidate_conditioned_pixels(
        candidate, calibration.width, calibration.height
    )
    selected_depth = depth[vv, uu]
    valid = (
        mask[vv, uu]
        & np.isfinite(selected_depth)
        & (selected_depth > adapter.MINIMUM_DEPTH_M)
        & (selected_depth < calibration.depth_max_m)
    )
    if int(np.count_nonzero(valid)) < adapter.MINIMUM_MASK_DEPTH_POINTS:
        return None

    candidate_depth = selected_depth[valid]
    depth_q15 = float(np.quantile(candidate_depth, adapter.NEAR_DEPTH_QUANTILE))
    slab_m = max(
        adapter.MINIMUM_NEAR_SLAB_M,
        min(
            adapter.MAXIMUM_NEAR_SLAB_M,
            adapter.NEAR_SLAB_DEPTH_RATIO * depth_q15,
        ),
    )
    foreground = valid & (selected_depth <= depth_q15 + slab_m)
    support = int(np.count_nonzero(foreground))
    if support < adapter.MINIMUM_MASK_DEPTH_POINTS:
        return None

    pixels = np.column_stack((uu[foreground], vv[foreground])).astype(np.float64)
    camera_points = adapter.unproject_pixels_camera_flu(
        pixels,
        selected_depth[foreground],
        calibration.intrinsic,
    )
    world_points = adapter.camera_flu_to_world(camera_points, camera_transform)
    anchor_points = adapter.world_to_anchor_fru(world_points, route_frame)[:, :2]
    center, footprint = x25.robust_obb(anchor_points)
    return center, footprint, x25.deterministic_subsample(anchor_points), support


def candidate_components(
    observation: adapter.FrameObservation,
    candidate_value: Mapping[str, Any],
    calibration: adapter.CameraCalibration,
    route_frame: adapter.AnchorFrame,
) -> tuple[list[ComponentMeasurement], list[dict[str, Any]]]:
    if not candidate_value["candidates"]:
        return [], []
    depth_m = adapter.load_depth_m(observation, calibration)
    branches: list[x25.FootprintMeasurement] = []
    for candidate in candidate_value["candidates"]:
        measured = adaptive_mask_footprint_measurement(
            candidate,
            depth_m,
            calibration,
            observation.camera_transform,
            route_frame,
        )
        if measured is None:
            continue
        center, footprint, points, support = measured
        branches.append(
            x25.FootprintMeasurement(
                class_id=int(candidate["class_id"]),
                class_name=str(candidate["class_name"]),
                confidence=float(candidate["confidence"]),
                bbox=tuple(
                    float(value) for value in candidate["bbox_xyxy_normalized"]
                ),
                position_xy=center,
                footprint_xy=footprint,
                surface_points_xy=points,
                depth_support=support,
            )
        )
    if not branches:
        return [], []

    cell_sets = [x27.lattice_cells(value.surface_points_xy) for value in branches]
    parents = list(range(len(branches)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(branches)):
        for right in range(left + 1, len(branches)):
            if not cell_sets[left].isdisjoint(cell_sets[right]):
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(branches)):
        groups.setdefault(find(index), []).append(index)

    components: list[ComponentMeasurement] = []
    provenance: list[dict[str, Any]] = []
    for ordinal, indices in enumerate(
        sorted(groups.values(), key=lambda values: min(values)), start=1
    ):
        values = [branches[index] for index in indices]
        points = np.vstack([value.surface_points_xy for value in values])
        points = np.unique(np.round(points, decimals=6), axis=0)
        # Vertical samples may collapse to fewer distinct ground-plane points.
        # Recheck the inherited OBB support requirement after deduplication.
        if len(points) < adapter.MINIMUM_MASK_DEPTH_POINTS:
            continue
        center, footprint = x25.robust_obb(points)
        labels = tuple(sorted({value.class_name for value in values}))
        bbox = (
            min(value.bbox[0] for value in values),
            min(value.bbox[1] for value in values),
            max(value.bbox[2] for value in values),
            max(value.bbox[3] for value in values),
        )
        component = ComponentMeasurement(
            class_id=WORLD_COMPONENT_CLASS_ID,
            class_name=WORLD_COMPONENT_CLASS_NAME,
            confidence=max(value.confidence for value in values),
            bbox=bbox,
            position_xy=center,
            footprint_xy=footprint,
            surface_points_xy=x25.deterministic_subsample(points),
            depth_support=sum(value.depth_support for value in values),
            semantic_labels=labels,
            branch_count=len(values),
        )
        components.append(component)
        provenance.append(
            {
                "component_ordinal": ordinal,
                "semantic_labels": list(labels),
                "branch_count": len(values),
                "adaptive_surface_samples": component.depth_support,
                "world_lattice_cells": len(x27.lattice_cells(points)),
            }
        )
    return components, provenance


def candidate_measurements(
    observation: adapter.FrameObservation,
    candidate_value: Mapping[str, Any],
    calibration: adapter.CameraCalibration,
    route_frame: adapter.AnchorFrame,
) -> list[ComponentMeasurement]:
    return candidate_components(
        observation, candidate_value, calibration, route_frame
    )[0]


class AdaptiveSurfaceLineageTracker(x29.TemporalLineageTracker):
    def update(
        self,
        measurements: Sequence[ComponentMeasurement],
        now_s: float,
        wearer_position_xy: Sequence[float],
    ) -> set[str]:
        measured_ids = super().update(measurements, now_s, wearer_position_xy)
        for track in self.tracks.values():
            previous_authority = track.authority
            if track.transport_conflicted:
                track.authority = x27.UNAUTHORIZED_MOTION
                track.authorized_velocity_xy = np.zeros(2, dtype=np.float64)
            elif len(track.transport_evidence) < x27.MINIMUM_AUTHORITY_PAIRS:
                occupancy_authority, occupancy_velocity_xy = x27.resolve_authority(
                    track.evidence, now_s
                )
                if occupancy_authority != x27.EGO_CARRIED:
                    track.authority = occupancy_authority
                    track.authorized_velocity_xy = occupancy_velocity_xy
            if track.authority != previous_authority:
                track.authority_transitions += 1
        return measured_ids


def first_contact_interval_entry_s(
    footprint_xy: Sequence[Sequence[float]],
    target_velocity_xy: Sequence[float],
    route_segments: Sequence[route.RouteSegment],
) -> float | None:
    footprint = np.asarray(footprint_xy, dtype=np.float64).reshape(-1, 2)
    velocity = np.asarray(target_velocity_xy, dtype=np.float64).reshape(2)
    radius = route.DEFAULT_TUBE_RADIUS_M
    for segment in route_segments:
        start = float(segment.start_offset_s)
        end = float(segment.end_offset_s)
        duration = max(0.0, end - start)
        samples = max(1, int(math.ceil(duration / x25.ROUTE_TIME_STEP_S)))
        times = np.linspace(start, end, samples + 1)
        previous_t = float(times[0])
        previous_d = x25.segment_distance_at(
            footprint, velocity, segment, previous_t
        )
        if previous_d <= radius + EPSILON:
            return previous_t
        for raw_time in times[1:]:
            current_t = float(raw_time)
            current_d = x25.segment_distance_at(
                footprint, velocity, segment, current_t
            )
            if previous_d > radius + EPSILON and current_d <= radius + EPSILON:
                low, high = previous_t, current_t
                for _ in range(x25.BISECTION_ITERATIONS):
                    middle = (low + high) * 0.5
                    if (
                        x25.segment_distance_at(
                            footprint, velocity, segment, middle
                        )
                        <= radius
                    ):
                        high = middle
                    else:
                        low = middle
                if (
                    x25.closing_speed_at(footprint, velocity, segment, high)
                    + EPSILON
                    >= route.DEFAULT_MIN_CLOSING_SPEED_MPS
                ):
                    return high
            previous_t, previous_d = current_t, current_d
    return None


def arm_frame(
    selection: route.RouteSelection,
    *,
    receipt: Mapping[str, Any] | None,
    observation: adapter.FrameObservation,
    wearer_position: tuple[float, float],
    wearer_velocity: tuple[float, float],
    tracks: Sequence[Mapping[str, Any]],
    confirmation: x24.RiskConfirmation,
    sample_period_s: float,
) -> dict[str, Any]:
    if selection.mode_changed:
        confirmation.reset()
    segments = route.build_route_segments(
        selection,
        receipt=receipt,
        now_s=observation.time_s,
        wearer_position_xy=wearer_position,
        wearer_velocity_xy=wearer_velocity,
    )
    entries = {
        str(track["track_id"]): first_contact_interval_entry_s(
            track["footprint_xy"],
            (track["velocity_forward_mps"], track["velocity_right_mps"]),
            segments,
        )
        for track in tracks
    }
    confirmed = confirmation.update(
        entries,
        now_s=observation.time_s,
        sample_period_s=sample_period_s,
    )
    confirmed_entries = [
        float(entries[key]) for key in confirmed if entries[key] is not None
    ]
    return {
        "route_mode": selection.mode,
        "authority": selection.authority,
        "plan_receipt_sha256": selection.receipt_sha256,
        "plan_position_residual_m": selection.plan_position_residual_m,
        "plan_velocity_direction_error_degrees": (
            selection.plan_velocity_direction_error_deg
        ),
        "fallback_reason": selection.fallback_reason,
        "route_mode_changed": selection.mode_changed,
        "route_risk": bool(confirmed_entries),
        "minimum_entry_s": min(confirmed_entries) if confirmed_entries else None,
        "candidate_risk_track_ids": sorted(
            key for key, value in entries.items() if value is not None
        ),
        "confirmed_risk_track_ids": sorted(confirmed),
    }


def predict_episode(
    episode: adapter.Episode,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    x24.require(
        len(candidate_values) == len(episode.observations),
        f"x30_candidate_count:{episode.episode_id}",
    )
    tracker = AdaptiveSurfaceLineageTracker()
    receipt_cache: dict[Path, dict[str, Any]] = {}
    confirmation = x24.RiskConfirmation()
    previous_mode: str | None = None
    frames: list[dict[str, Any]] = []
    route_mode_counts: Counter[str] = Counter()
    authority_track_frames: Counter[str] = Counter()
    measured_frames = emitted_frames = hold_frames = 0
    for ordinal, (observation, candidate_value) in enumerate(
        zip(episode.observations, candidate_values, strict=True)
    ):
        neighbour = (
            episode.observations[ordinal - 1]
            if ordinal
            else episode.observations[ordinal + 1]
        )
        sample_period_s = abs(observation.time_s - neighbour.time_s)
        x24.require(
            sample_period_s > 0.0,
            f"x30_sample_period:{episode.episode_id}:{ordinal}",
        )
        wearer_position, wearer_velocity = x24.wearer_anchor_state(
            observation, episode.route_frame
        )
        measurements, measurement_components = candidate_components(
            observation,
            candidate_value,
            calibration,
            episode.route_frame,
        )
        measured_frames += int(bool(measurements))
        measured_ids = tracker.update(
            measurements, observation.time_s, wearer_position
        )
        tracks = tracker.emitted(observation.time_s, measured_ids)
        risk_tracks = [value for value in tracks if bool(value["risk_eligible"])]
        authority_track_frames.update(
            str(value["motion_authority"]) for value in tracks
        )
        emitted_frames += int(bool(tracks))
        hold_frames += int(
            any(track["disposition"] == "HOLD" for track in tracks)
        )
        receipt = x24.load_receipt(observation, receipt_cache)
        selection = route.select_route(
            receipt,
            session_id=observation.navigation_session_id,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
            previous_mode=previous_mode,
        )
        previous_mode = selection.mode
        route_mode_counts[selection.mode] += 1
        frames.append(
            {
                "sample_index": observation.sample_index,
                "time_s": observation.time_s,
                "world_frame": observation.world_frame,
                "raw_candidates": len(candidate_value["candidates"]),
                "metric_footprint_measurements": len(measurements),
                "measurement_components": measurement_components,
                "tracks": tracks,
                "risk_eligible_tracks": len(risk_tracks),
                "arms": {
                    ARM_X30: arm_frame(
                        selection,
                        receipt=receipt,
                        observation=observation,
                        wearer_position=wearer_position,
                        wearer_velocity=wearer_velocity,
                        tracks=risk_tracks,
                        confirmation=confirmation,
                        sample_period_s=sample_period_s,
                    )
                },
            }
        )
    return {
        "episode_id": episode.episode_id,
        "frames": frames,
        "diagnostics": {
            "frame_count": len(frames),
            "metric_footprint_measurement_frames": measured_frames,
            "emitted_track_frames": emitted_frames,
            "track_coverage": emitted_frames / max(1, len(frames)),
            "hold_frames": hold_frames,
            "authority_track_frames": dict(sorted(authority_track_frames.items())),
            "authority_transitions": sum(
                value.authority_transitions for value in tracker.tracks.values()
            ),
            "x30_route_mode_counts": dict(sorted(route_mode_counts.items())),
        },
        "arms": {
            ARM_X30: {
                "route_risk_frames": sum(
                    bool(frame["arms"][ARM_X30]["route_risk"])
                    for frame in frames
                ),
                "first_route_risk_time_s": next(
                    (
                        frame["time_s"]
                        for frame in frames
                        if frame["arms"][ARM_X30]["route_risk"]
                    ),
                    None,
                ),
            }
        },
    }


def _algorithm_files() -> dict[str, Path]:
    return {
        "x25_geometry": Path(x25.__file__).resolve(),
        "x27_authority": Path(x27.__file__).resolve(),
        "x29_lineage": Path(x29.__file__).resolve(),
        "x30_predictor": Path(__file__).resolve(),
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(
        not output["freeze"].exists(), f"x30_freeze_exists:{output['freeze']}"
    )
    frozen_x24, contract, _candidates = x24.require_freeze(run_root)
    x24.require(output["x24_predictions"].is_file(), "x30_x24_baseline_missing")
    value = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN_TRUTH_BLIND_PENDING_PREDICTION",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "source": {
            "x24_freeze_sha256": x24.sha256_file(output["x24_freeze"]),
            "x24_predictions_sha256": x24.sha256_file(
                output["x24_predictions"]
            ),
            "model_manifest_sha256": contract.manifest_sha256,
            "candidate_aggregate_sha256": frozen_x24["candidates"][
                "aggregate_sha256"
            ],
        },
        "algorithm_files": {
            name: {"path": str(path), "sha256": x24.sha256_file(path)}
            for name, path in _algorithm_files().items()
        },
        "episodes": len(contract.episodes),
        "frames": len(x24.flatten_observations(contract)),
        "fixed_constants": fixed_constants(),
        "arm": ARM_X30,
    }
    x24.write_json_exclusive(output["freeze"], value)
    return {**value, "freeze_sha256": x24.sha256_file(output["freeze"])}


def require_freeze(
    run_root: Path,
) -> tuple[dict[str, Any], adapter.SanitizedModelContract, list[dict[str, Any]]]:
    output = paths(run_root)
    frozen = x24.read_json(output["freeze"].resolve(strict=True), "x30_freeze")
    x24.require(frozen.get("schema") == FREEZE_SCHEMA, "x30_freeze_schema")
    x24.require(
        frozen.get("fixed_constants") == fixed_constants(),
        "x30_constants_drift",
    )
    for name, path in _algorithm_files().items():
        x24.require(
            frozen["algorithm_files"][name]["sha256"] == x24.sha256_file(path),
            f"x30_algorithm_drift:{name}",
        )
    frozen_x24, contract, candidates = x24.require_freeze(run_root)
    x24.require(
        frozen["source"]["x24_freeze_sha256"]
        == x24.sha256_file(output["x24_freeze"]),
        "x30_x24_freeze_drift",
    )
    x24.require(
        frozen["source"]["x24_predictions_sha256"]
        == x24.sha256_file(output["x24_predictions"]),
        "x30_x24_prediction_drift",
    )
    x24.require(
        frozen["source"]["candidate_aggregate_sha256"]
        == frozen_x24["candidates"]["aggregate_sha256"],
        "x30_candidate_drift",
    )
    return frozen, contract, candidates


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(
        not output["predictions"].exists(),
        f"x30_predictions_exist:{output['predictions']}",
    )
    frozen, contract, candidate_values = require_freeze(run_root)
    cursor = 0
    episodes: dict[str, Any] = {}
    for episode in contract.episodes:
        count = len(episode.observations)
        episodes[episode.episode_id] = predict_episode(
            episode,
            candidate_values[cursor : cursor + count],
            contract.calibration,
        )
        cursor += count
    value = {
        "schema": PREDICTION_SCHEMA,
        "status": "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "arms": [ARM_X30],
        "episodes": episodes,
        "fixed_constants": fixed_constants(),
        "source": {
            "freeze_sha256": x24.sha256_file(output["freeze"]),
            **frozen["source"],
        },
        "claim_boundary": {
            "synthetic_development": True,
            "evaluator_opened": False,
            "current_actor_oracle_used": False,
        },
    }
    x24.write_json_exclusive(output["predictions"], value)
    return {
        **value,
        "predictions_sha256": x24.sha256_file(output["predictions"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "predict"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    value = freeze(args) if args.command == "freeze" else predict(args)
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
