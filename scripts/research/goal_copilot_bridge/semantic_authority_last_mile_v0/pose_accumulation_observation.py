"""Pose-conditioned two-view accumulation over DeepLSD boundary fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dense_boundary_observation import (
    DISTANCE_THRESHOLD_PX,
    FINAL_SUPPORT_MINIMUM_PX,
    VERTICAL_ORIENTATION_TOLERANCE_DEG,
    DeepLsdDenseFieldExtractor,
    _angular_distance_mod_pi,
)
from .observation import ApertureObservation, RgbEpisodeInput, RgbEpisodeTruth
from .two_view_observation import (
    ImageLine,
    SourceCameraPose,
    SourcePoseTwoViewBoundaryProvider,
    _intrinsic_matrix,
    _line_distance,
    _normalised_line,
    _project_world_line,
    oracle_pixel_lines,
    triangulate_aperture,
)


DEPTH_MINIMUM_M = 0.6
DEPTH_MAXIMUM_M = 6.0
DEPTH_SAMPLE_COUNT = 55
IMAGE_X_STRIDE_PX = 2
MAXIMUM_HYPOTHESES = 96


@dataclass(frozen=True)
class BoundaryHypothesis:
    line_a: ImageLine
    line_b: ImageLine
    depth_m: float
    joint_score: float
    x_a_px: float
    x_b_mid_px: float


def _line_orientation(line: ImageLine) -> float:
    a, b, _ = line.coefficients
    return float(np.mod(math.atan2(a, -b), np.pi))


def _field_support(field: dict, line: ImageLine) -> tuple[float, float]:
    distance = field["distance"]
    orientation = field["orientation"]
    ys = np.arange(distance.shape[0], dtype=np.int32)
    a, b, c = line.coefficients
    xs = np.rint(-(b * ys + c) / a).astype(np.int32)
    valid = (xs >= 0) & (xs < distance.shape[1])
    if not np.any(valid):
        return 0.0, 0.0
    sample_y = ys[valid]
    sample_x = xs[valid]
    sample_distance = distance[sample_y, sample_x]
    angular_distance = _angular_distance_mod_pi(orientation[sample_y, sample_x], _line_orientation(line))
    support = (sample_distance <= DISTANCE_THRESHOLD_PX) & (
        angular_distance <= math.radians(VERTICAL_ORIENTATION_TOLERANCE_DEG)
    )
    support_length = float(np.count_nonzero(support))
    distance_score = np.exp(-sample_distance / DISTANCE_THRESHOLD_PX)
    orientation_score = np.clip(
        1.0 - angular_distance / math.radians(VERTICAL_ORIENTATION_TOLERANCE_DEG), 0.0, 1.0
    )
    return support_length, float(np.mean(distance_score * orientation_score))


def accumulate_boundary_hypotheses(
    episode_input: RgbEpisodeInput,
    pose_a: SourceCameraPose,
    pose_b: SourceCameraPose,
    field_a: dict,
    field_b: dict,
) -> tuple[list[BoundaryHypothesis], dict]:
    intrinsic = _intrinsic_matrix(episode_input)
    height = episode_input.intrinsics.height
    width = episode_input.intrinsics.width
    depths = np.linspace(DEPTH_MINIMUM_M, DEPTH_MAXIMUM_M, DEPTH_SAMPLE_COUNT)
    hypotheses = []
    evaluated = 0
    support_rejected = 0
    for x_a in range(int(width * 0.03), int(width * 0.97) + 1, IMAGE_X_STRIDE_PX):
        raw_line_a = ImageLine(tuple(_normalised_line(x_a, 0.0, x_a, height - 1)), 1.0, 1)
        support_a, score_a = _field_support(field_a, raw_line_a)
        if support_a < FINAL_SUPPORT_MINIMUM_PX:
            continue
        line_a = ImageLine(raw_line_a.coefficients, support_a, 1)
        for depth in depths:
            evaluated += 1
            x_m = (x_a - episode_input.intrinsics.cx) * depth / episode_input.intrinsics.fx
            point_camera = np.asarray([x_m, 0.0, depth], dtype=np.float64)
            point_world = pose_a.rotation @ point_camera + pose_a.position
            direction_world = pose_a.rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            raw_line_b = _project_world_line(point_world, direction_world, pose_b, intrinsic)
            x_b_mid = raw_line_b.x_at((height - 1) * 0.5)
            if not 0.0 <= x_b_mid < width:
                continue
            support_b, score_b = _field_support(field_b, raw_line_b)
            if support_b < FINAL_SUPPORT_MINIMUM_PX:
                support_rejected += 1
                continue
            line_b = ImageLine(raw_line_b.coefficients, support_b, 1)
            hypotheses.append(
                BoundaryHypothesis(
                    line_a,
                    line_b,
                    float(depth),
                    math.sqrt(max(score_a, 0.0) * max(score_b, 0.0)),
                    float(x_a),
                    float(x_b_mid),
                )
            )
    hypotheses.sort(key=lambda row: row.joint_score, reverse=True)
    retained = []
    for hypothesis in hypotheses:
        if any(
            abs(hypothesis.x_a_px - existing.x_a_px) < 3.0
            and abs(hypothesis.x_b_mid_px - existing.x_b_mid_px) < 3.0
            for existing in retained
        ):
            continue
        retained.append(hypothesis)
        if len(retained) == MAXIMUM_HYPOTHESES:
            break
    return retained, {
        "depth_range_m": [DEPTH_MINIMUM_M, DEPTH_MAXIMUM_M],
        "depth_sample_count": DEPTH_SAMPLE_COUNT,
        "image_x_stride_px": IMAGE_X_STRIDE_PX,
        "maximum_hypotheses": MAXIMUM_HYPOTHESES,
        "evaluated_hypothesis_count": evaluated,
        "support_rejected_count": support_rejected,
        "pre_nms_hypothesis_count": len(hypotheses),
        "retained_hypothesis_count": len(retained),
        "top_joint_scores": [row.joint_score for row in retained[:10]],
    }


class PoseAccumulatedOracleBoundaryProvider(SourcePoseTwoViewBoundaryProvider):
    """Use oracle association only after pose-conditioned joint field accumulation."""

    def __init__(
        self,
        episode_input: RgbEpisodeInput,
        truth: RgbEpisodeTruth,
        pose_a: SourceCameraPose,
        pose_b: SourceCameraPose,
        extractor: DeepLsdDenseFieldExtractor,
    ) -> None:
        super().__init__(episode_input, truth, pose_a, pose_b, "b1")
        self.extractor = extractor
        self.arm_name = "SAGE_LM_V1B_R4_POSE_CONDITIONED_MULTI_VIEW_ACCUMULATION_B1"

    def observe(self) -> ApertureObservation:
        assert self.truth is not None
        visible = [row for row in self.input.exact_anchor_observations if row.visible]
        first = next((row for row in visible if row.frame_index == 0), None)
        second = next((row for row in visible if row.frame_index == self.input.active_parallax_frame_index), None)
        if first is None or second is None or first.referent_id != second.referent_id:
            raise ValueError("active source-pose pair requires the same visible exact anchor")
        frame_a = self._load(self.input.rgb_frames[first.frame_index])
        frame_b = self._load(self.input.rgb_frames[second.frame_index])
        field_a = self.extractor.predict_field(frame_a, first.bbox_xyxy)
        field_b = self.extractor.predict_field(frame_b, second.bbox_xyxy)
        hypotheses, accumulation_diagnostics = accumulate_boundary_hypotheses(
            self.input, self.pose_a, self.pose_b, field_a, field_b
        )
        oracle_a, oracle_b = oracle_pixel_lines(self.input, self.truth, self.pose_a, self.pose_b)
        selected = []
        distances = []
        height = self.input.intrinsics.height
        for truth_a, truth_b in zip(oracle_a, oracle_b):
            if not hypotheses:
                break
            hypothesis = min(
                hypotheses,
                key=lambda row: _line_distance(row.line_a, truth_a, height)
                + _line_distance(row.line_b, truth_b, height),
            )
            selected.append(hypothesis)
            distances.extend(
                [
                    _line_distance(hypothesis.line_a, truth_a, height),
                    _line_distance(hypothesis.line_b, truth_b, height),
                ]
            )
        self.diagnostics.update(
            {
                "accumulation": accumulation_diagnostics,
                "oracle_lines_a": [line.as_dict(height) for line in oracle_a],
                "oracle_lines_b": [line.as_dict(height) for line in oracle_b],
                "oracle_association_distances_px": distances,
                "selected_hypotheses": [
                    {
                        "line_a": row.line_a.as_dict(height),
                        "line_b": row.line_b.as_dict(height),
                        "depth_m": row.depth_m,
                        "joint_score": row.joint_score,
                    }
                    for row in selected
                ],
            }
        )
        if len(selected) != 2 or max(distances, default=math.inf) > 9.0:
            self.diagnostics["failure"] = "BOUNDARY_HYPOTHESIS_MISSING"
            return self._observation(None)
        intrinsic = _intrinsic_matrix(self.input)
        geometry = triangulate_aperture(
            selected[0].line_a,
            selected[1].line_a,
            selected[0].line_b,
            selected[1].line_b,
            self.pose_a,
            self.pose_b,
            intrinsic,
            height * 0.55,
        )
        distance_score = math.exp(-float(np.mean(distances)) / 7.0)
        self.diagnostics["geometry"] = geometry.__dict__ if geometry else None
        return self._observation(geometry, distance_score)
