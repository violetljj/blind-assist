"""Anchor-conditioned aperture-pair proposals for SAGE-LM V1-B-R5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dense_boundary_observation import (
    DISTANCE_THRESHOLD_PX,
    FINAL_SUPPORT_MINIMUM_PX,
    VERTICAL_ORIENTATION_TOLERANCE_DEG,
    DeepLsdDenseFieldExtractor,
    _angular_distance_mod_pi,
)
from .observation import ApertureObservation, RgbEpisodeInput, RgbEpisodeTruth
from .pose_accumulation_observation import (
    DEPTH_MAXIMUM_M,
    DEPTH_MINIMUM_M,
    DEPTH_SAMPLE_COUNT,
    IMAGE_X_STRIDE_PX,
    _line_orientation,
)
from .two_view_observation import (
    BoundaryGeometry,
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


MAXIMUM_APERTURE_PAIRS = 96
BOUNDARY_X_BIN_PX = 24.0
BOUNDARY_DEPTH_BIN_M = 0.60
BOUNDARIES_PER_X_DEPTH_CELL = 2


@dataclass(frozen=True)
class AnchorBoundaryHypothesis:
    line_a: ImageLine
    line_b: ImageLine
    depth_m: float
    x_a_px: float
    x_b_mid_px: float
    coverage: float
    continuity: float
    reprojection_support: float


@dataclass(frozen=True)
class AperturePairHypothesis:
    left: AnchorBoundaryHypothesis
    right: AnchorBoundaryHypothesis
    geometry: BoundaryGeometry
    score: float
    anchor_relation: str
    score_components: dict[str, float]


def _field_profile(field: dict, line: ImageLine) -> tuple[float, float, float]:
    distance = field["distance"]
    orientation = field["orientation"]
    ys = np.arange(distance.shape[0], dtype=np.int32)
    a, b, c = line.coefficients
    xs = np.rint(-(b * ys + c) / a).astype(np.int32)
    valid = (xs >= 0) & (xs < distance.shape[1])
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    sample_y = ys[valid]
    sample_x = xs[valid]
    sample_distance = distance[sample_y, sample_x]
    angular_distance = _angular_distance_mod_pi(orientation[sample_y, sample_x], _line_orientation(line))
    support = (sample_distance <= DISTANCE_THRESHOLD_PX) & (
        angular_distance <= math.radians(VERTICAL_ORIENTATION_TOLERANCE_DEG)
    )
    support_count = float(np.count_nonzero(support))
    longest_run = 0
    current_run = 0
    for supported in support:
        current_run = current_run + 1 if supported else 0
        longest_run = max(longest_run, current_run)
    distance_score = np.exp(-sample_distance / DISTANCE_THRESHOLD_PX)
    orientation_score = np.clip(
        1.0 - angular_distance / math.radians(VERTICAL_ORIENTATION_TOLERANCE_DEG), 0.0, 1.0
    )
    joint_residual = float(np.mean((distance_score * orientation_score)[support])) if np.any(support) else 0.0
    return support_count, float(longest_run), joint_residual


def _enumerate_boundaries(
    episode_input: RgbEpisodeInput,
    pose_a: SourceCameraPose,
    pose_b: SourceCameraPose,
    field_a: dict,
    field_b: dict,
) -> tuple[list[AnchorBoundaryHypothesis], dict]:
    intrinsic = _intrinsic_matrix(episode_input)
    height = episode_input.intrinsics.height
    width = episode_input.intrinsics.width
    depths = np.linspace(DEPTH_MINIMUM_M, DEPTH_MAXIMUM_M, DEPTH_SAMPLE_COUNT)
    cells: dict[tuple[int, int], list[AnchorBoundaryHypothesis]] = {}
    evaluated = 0
    supported = 0
    for x_a in range(int(width * 0.03), int(width * 0.97) + 1, IMAGE_X_STRIDE_PX):
        raw_line_a = ImageLine(tuple(_normalised_line(x_a, 0.0, x_a, height - 1)), 1.0, 1)
        support_a, continuity_a, residual_a = _field_profile(field_a, raw_line_a)
        if support_a < FINAL_SUPPORT_MINIMUM_PX:
            continue
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
            support_b, continuity_b, residual_b = _field_profile(field_b, raw_line_b)
            if support_b < FINAL_SUPPORT_MINIMUM_PX:
                continue
            supported += 1
            hypothesis = AnchorBoundaryHypothesis(
                ImageLine(raw_line_a.coefficients, support_a, 1),
                ImageLine(raw_line_b.coefficients, support_b, 1),
                float(depth),
                float(x_a),
                float(x_b_mid),
                min(support_a, support_b) / height,
                min(continuity_a, continuity_b) / height,
                math.sqrt(max(residual_a, 0.0) * max(residual_b, 0.0)),
            )
            cell = (int(x_a // BOUNDARY_X_BIN_PX), int((depth - DEPTH_MINIMUM_M) // BOUNDARY_DEPTH_BIN_M))
            cells.setdefault(cell, []).append(hypothesis)
    retained = []
    for rows in cells.values():
        rows.sort(key=lambda row: (row.coverage, row.continuity, row.reprojection_support), reverse=True)
        retained.extend(rows[:BOUNDARIES_PER_X_DEPTH_CELL])
    return retained, {
        "evaluated_boundary_count": evaluated,
        "supported_boundary_count": supported,
        "diversity_cell_count": len(cells),
        "retained_boundary_count": len(retained),
        "boundary_x_bin_px": BOUNDARY_X_BIN_PX,
        "boundary_depth_bin_m": BOUNDARY_DEPTH_BIN_M,
        "boundaries_per_x_depth_cell": BOUNDARIES_PER_X_DEPTH_CELL,
    }


def _anchor_relation(left_x: float, right_x: float, anchor_x: float) -> tuple[str, float]:
    if left_x <= anchor_x <= right_x:
        return "BRACKETS_ANCHOR", 1.0
    edge_distance = min(abs(anchor_x - left_x), abs(anchor_x - right_x))
    relation = "APERTURE_LEFT_OF_ANCHOR" if right_x < anchor_x else "APERTURE_RIGHT_OF_ANCHOR"
    return relation, edge_distance


def _pair_score(
    left: AnchorBoundaryHypothesis,
    right: AnchorBoundaryHypothesis,
    geometry: BoundaryGeometry,
    anchor_x: float,
    image_width: int,
    task_conditioned_score: float | None = None,
) -> tuple[float, str, dict[str, float]]:
    relation, anchor_value = _anchor_relation(left.x_a_px, right.x_a_px, anchor_x)
    anchor_score = 1.0 if relation == "BRACKETS_ANCHOR" else math.exp(-anchor_value / (image_width * 0.18))
    components = {
        "balanced_coverage": min(left.coverage, right.coverage),
        "two_view_reprojection": math.sqrt(left.reprojection_support * right.reprojection_support),
        "anchor_bracketing": anchor_score,
        "plausible_span": geometry.width_score,
        "support_continuity": min(left.continuity, right.continuity),
    }
    secondary = (
        0.25 * components["two_view_reprojection"]
        + 0.20 * components["anchor_bracketing"]
        + 0.25 * components["plausible_span"]
        + 0.30 * components["support_continuity"]
    )
    # Balanced two-sided coverage is the primary objective: secondary quality can
    # rank complete pairs, but cannot compensate for one unsupported boundary.
    if task_conditioned_score is None:
        score = components["balanced_coverage"] * (0.25 + 0.75 * secondary)
    else:
        components["task_conditioned_boundary_probability"] = task_conditioned_score
        score = components["balanced_coverage"] * (0.15 + 0.55 * task_conditioned_score + 0.30 * secondary)
    return score, relation, components


def _select_diverse_pairs(pairs: list[AperturePairHypothesis], width: int) -> list[AperturePairHypothesis]:
    if len(pairs) <= MAXIMUM_APERTURE_PAIRS:
        return sorted(pairs, key=lambda row: row.score, reverse=True)
    # Diversity is measured in the observable pair itself, not in arbitrary
    # depth cells: the four projected boundary x coordinates are exactly the
    # coverage surface that the 9 px localization gate later consumes.
    cells: dict[tuple[int, int, int, int], AperturePairHypothesis] = {}
    cell_px = 3.0
    for pair in pairs:
        cell = tuple(
            int(value // cell_px)
            for value in (pair.left.x_a_px, pair.right.x_a_px, pair.left.x_b_mid_px, pair.right.x_b_mid_px)
        )
        if cell not in cells or pair.score > cells[cell].score:
            cells[cell] = pair
    candidates = list(cells.values())
    maximum_score = max(row.score for row in candidates)
    retained: list[AperturePairHypothesis] = []
    # Seed each observable anchor relation, then use weighted farthest-point
    # retention so repeated salient clutter cannot consume the whole budget.
    for relation in ("BRACKETS_ANCHOR", "APERTURE_LEFT_OF_ANCHOR", "APERTURE_RIGHT_OF_ANCHOR"):
        relation_rows = [row for row in candidates if row.anchor_relation == relation]
        if relation_rows:
            retained.append(max(relation_rows, key=lambda row: row.score))
    retained_ids = {id(row) for row in retained}

    def feature(pair: AperturePairHypothesis) -> np.ndarray:
        return np.asarray(
            [pair.left.x_a_px, pair.right.x_a_px, pair.left.x_b_mid_px, pair.right.x_b_mid_px],
            dtype=np.float64,
        ) / width

    features = np.stack([feature(row) for row in candidates])
    selected_indices = [index for index, row in enumerate(candidates) if id(row) in retained_ids]
    minimum_novelty = np.full(len(candidates), np.inf, dtype=np.float64)
    for index in selected_indices:
        minimum_novelty = np.minimum(minimum_novelty, np.max(np.abs(features - features[index]), axis=1))
    selected_mask = np.zeros(len(candidates), dtype=bool)
    selected_mask[selected_indices] = True
    score_quality = np.asarray([row.score / max(maximum_score, 1e-9) for row in candidates])
    while len(retained) < MAXIMUM_APERTURE_PAIRS and len(retained_ids) < len(candidates):
        priority = minimum_novelty * (0.55 + 0.45 * score_quality)
        priority[selected_mask] = -math.inf
        best_index = int(np.argmax(priority))
        best = candidates[best_index]
        retained.append(best)
        retained_ids.add(id(best))
        selected_mask[best_index] = True
        minimum_novelty = np.minimum(
            minimum_novelty, np.max(np.abs(features - features[best_index]), axis=1)
        )
    return sorted(retained, key=lambda row: row.score, reverse=True)


def propose_aperture_pairs(
    episode_input: RgbEpisodeInput,
    pose_a: SourceCameraPose,
    pose_b: SourceCameraPose,
    field_a: dict,
    field_b: dict,
    anchor_bbox: tuple[int, int, int, int],
    role_probabilities: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[AperturePairHypothesis], dict]:
    boundaries, boundary_diagnostics = _enumerate_boundaries(episode_input, pose_a, pose_b, field_a, field_b)
    width = episode_input.intrinsics.width
    height = episode_input.intrinsics.height
    anchor_x = (anchor_bbox[0] + anchor_bbox[2]) * 0.5
    left_pool = [row for row in boundaries if row.x_a_px <= anchor_x + width * 0.20]
    right_pool = [row for row in boundaries if row.x_a_px >= anchor_x - width * 0.20]
    intrinsic = _intrinsic_matrix(episode_input)
    pairs = []
    for left in left_pool:
        for right in right_pool:
            span = right.x_a_px - left.x_a_px
            if span < width * 0.13 or span > width * 0.66:
                continue
            edge_distance = min(abs(anchor_x - left.x_a_px), abs(anchor_x - right.x_a_px))
            if edge_distance > width * 0.34:
                continue
            geometry = triangulate_aperture(
                left.line_a,
                right.line_a,
                left.line_b,
                right.line_b,
                pose_a,
                pose_b,
                intrinsic,
                height * 0.55,
            )
            if geometry is None:
                continue
            task_score = None
            if role_probabilities is not None:
                left_a_probability, right_a_probability, left_b_probability, right_b_probability = role_probabilities
                indices = (
                    int(np.clip(round(left.x_a_px), 0, width - 1)),
                    int(np.clip(round(right.x_a_px), 0, width - 1)),
                    int(np.clip(round(left.x_b_mid_px), 0, width - 1)),
                    int(np.clip(round(right.x_b_mid_px), 0, width - 1)),
                )
                task_score = float(
                    (
                        left_a_probability[indices[0]]
                        * right_a_probability[indices[1]]
                        * left_b_probability[indices[2]]
                        * right_b_probability[indices[3]]
                    )
                    ** 0.25
                )
            score, relation, components = _pair_score(
                left, right, geometry, anchor_x, width, task_conditioned_score=task_score
            )
            pairs.append(AperturePairHypothesis(left, right, geometry, score, relation, components))
    retained = _select_diverse_pairs(pairs, width)
    diagnostics = {
        **boundary_diagnostics,
        "left_pool_count": len(left_pool),
        "right_pool_count": len(right_pool),
        "valid_pair_count": len(pairs),
        "retained_pair_count": len(retained),
        "maximum_aperture_pairs": MAXIMUM_APERTURE_PAIRS,
        "retained_relation_counts": {
            relation: sum(row.anchor_relation == relation for row in retained)
            for relation in ("BRACKETS_ANCHOR", "APERTURE_LEFT_OF_ANCHOR", "APERTURE_RIGHT_OF_ANCHOR")
        },
        "top_pair_scores": [row.score for row in retained[:10]],
    }
    return retained, diagnostics


class AnchorPairOracleBoundaryProvider(SourcePoseTwoViewBoundaryProvider):
    """Use evaluator association only after R5 has proposed complete pairs."""

    def __init__(
        self,
        episode_input: RgbEpisodeInput,
        truth: RgbEpisodeTruth,
        pose_a: SourceCameraPose,
        pose_b: SourceCameraPose,
        extractor: DeepLsdDenseFieldExtractor,
        role_predictor: Callable[[dict, tuple[int, int, int, int]], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        super().__init__(episode_input, truth, pose_a, pose_b, "b1")
        self.extractor = extractor
        self.role_predictor = role_predictor
        self.arm_name = "SAGE_LM_V1B_R5_ANCHOR_CONDITIONED_APERTURE_PAIR_COVERAGE_B1"

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
        role_probabilities = None
        if self.role_predictor is not None:
            left_a, right_a = self.role_predictor(field_a, first.bbox_xyxy)
            left_b, right_b = self.role_predictor(field_b, second.bbox_xyxy)
            role_probabilities = left_a, right_a, left_b, right_b
        pairs, proposal_diagnostics = propose_aperture_pairs(
            self.input,
            self.pose_a,
            self.pose_b,
            field_a,
            field_b,
            first.bbox_xyxy,
            role_probabilities=role_probabilities,
        )
        oracle_a, oracle_b = oracle_pixel_lines(self.input, self.truth, self.pose_a, self.pose_b)
        height = self.input.intrinsics.height
        selected = min(
            pairs,
            key=lambda row: _line_distance(row.left.line_a, oracle_a[0], height)
            + _line_distance(row.right.line_a, oracle_a[1], height)
            + _line_distance(row.left.line_b, oracle_b[0], height)
            + _line_distance(row.right.line_b, oracle_b[1], height),
            default=None,
        )
        distances = [] if selected is None else [
            _line_distance(selected.left.line_a, oracle_a[0], height),
            _line_distance(selected.right.line_a, oracle_a[1], height),
            _line_distance(selected.left.line_b, oracle_b[0], height),
            _line_distance(selected.right.line_b, oracle_b[1], height),
        ]
        self.diagnostics.update(
            {
                "pair_proposal": proposal_diagnostics,
                "oracle_lines_a": [line.as_dict(height) for line in oracle_a],
                "oracle_lines_b": [line.as_dict(height) for line in oracle_b],
                "oracle_association_distances_px": distances,
                "selected_pair": None if selected is None else {
                    "left_a": selected.left.line_a.as_dict(height),
                    "right_a": selected.right.line_a.as_dict(height),
                    "left_b": selected.left.line_b.as_dict(height),
                    "right_b": selected.right.line_b.as_dict(height),
                    "left_depth_m": selected.left.depth_m,
                    "right_depth_m": selected.right.depth_m,
                    "pair_score": selected.score,
                    "anchor_relation": selected.anchor_relation,
                    "score_components": selected.score_components,
                },
            }
        )
        if selected is None or max(distances, default=math.inf) > 9.0:
            self.diagnostics["failure"] = "APERTURE_PAIR_HYPOTHESIS_MISSING"
            return self._observation(None)
        distance_score = math.exp(-float(np.mean(distances)) / 7.0)
        self.diagnostics["geometry"] = selected.geometry.__dict__
        return self._observation(selected.geometry, distance_score)
