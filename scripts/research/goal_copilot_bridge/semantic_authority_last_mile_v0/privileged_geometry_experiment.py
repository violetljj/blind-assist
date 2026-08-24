"""Run the SAGE-LM V1-E0 privileged-geometry teacher ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .active_parallax_experiment import _available_episode_ids
from .experiment import _aggregate
from .privileged_geometry_teacher import (
    MeshDepthRenderer,
    PrivilegedGeometryFrame,
    link_boundary_lines,
)
from .rgb_experiment import _baseline, _sage_lm
from .task_boundary_field_experiment import LOCALIZATION_GATE_PX, TOP_K_PER_ROLE
from .two_view_experiment import _arm_diagnostics, _evaluator_episode, _source_poses
from .two_view_observation import (
    SourcePoseTwoViewBoundaryProvider,
    _intrinsic_matrix,
    _line_distance,
    oracle_pixel_lines,
    triangulate_aperture,
)

SCHEMA_VERSION = "sage_lm_v1e0_privileged_geometry_teacher_ceiling"
ARM = "v1e0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PrivilegedGeometryProvider(SourcePoseTwoViewBoundaryProvider):
    def __init__(self, episode_input, truth, pose_a, pose_b, frames: tuple[PrivilegedGeometryFrame, PrivilegedGeometryFrame]) -> None:
        super().__init__(episode_input, truth, pose_a, pose_b, "b1")
        self.frames = frames
        self.arm_name = "SAGE_LM_V1E0_PRIVILEGED_GEOMETRY_TEACHER"

    def observe(self):
        height, width = self.input.intrinsics.height, self.input.intrinsics.width
        roles = [frame.lines for frame in self.frames]
        oracle_a, oracle_b = oracle_pixel_lines(self.input, self.truth, self.pose_a, self.pose_b)
        intrinsic = _intrinsic_matrix(self.input)
        candidates = []
        for left_a in roles[0][0]:
            for right_a in roles[0][1]:
                if not width * 0.10 <= right_a.x_at(height * 0.5) - left_a.x_at(height * 0.5) <= width * 0.70:
                    continue
                for left_b in roles[1][0]:
                    for right_b in roles[1][1]:
                        if not width * 0.10 <= right_b.x_at(height * 0.5) - left_b.x_at(height * 0.5) <= width * 0.70:
                            continue
                        geometry = triangulate_aperture(left_a, right_a, left_b, right_b, self.pose_a, self.pose_b, intrinsic, height * 0.55)
                        if geometry is None:
                            continue
                        distances = [
                            _line_distance(left_a, oracle_a[0], height),
                            _line_distance(right_a, oracle_a[1], height),
                            _line_distance(left_b, oracle_b[0], height),
                            _line_distance(right_b, oracle_b[1], height),
                        ]
                        candidates.append((sum(distances), distances, geometry))
        selected = min(candidates, key=lambda row: row[0], default=None)
        distances = [] if selected is None else selected[1]
        pools = (roles[0][0], roles[0][1], roles[1][0], roles[1][1])
        direct_hits = [
            bool(pool) and min(_line_distance(line, oracle, height) for line in pool) <= LOCALIZATION_GATE_PX
            for pool, oracle in zip(pools, (*oracle_a, *oracle_b))
        ]
        self.diagnostics.update(
            {
                "teacher_frame_diagnostics": [frame.diagnostics for frame in self.frames],
                "top_k_per_role": TOP_K_PER_ROLE,
                "role_candidate_counts": [len(pool) for pool in pools],
                "direct_four_boundary_hits": direct_hits,
                "oracle_association_distances_px": distances,
                "valid_geometry_combination_count": len(candidates),
                "uses_privileged_mesh_depth": True,
                "uses_rgb": False,
            }
        )
        if selected is None or max(distances, default=math.inf) > LOCALIZATION_GATE_PX:
            self.diagnostics["failure"] = "PRIVILEGED_BOUNDARY_PAIR_MISSING"
            return self._observation(None)
        self.diagnostics["geometry"] = selected[2].__dict__
        return self._observation(selected[2], math.exp(-float(np.mean(distances)) / 7.0))


def _draw_lines(image: np.ndarray, frame: PrivilegedGeometryFrame, oracle, hits: list[bool], label: str) -> np.ndarray:
    output = image.copy()
    h = output.shape[0]
    colors = ((0, 80, 255), (255, 180, 0))
    for role, pool in enumerate(frame.lines):
        for line in pool:
            cv2.line(output, (round(line.x_at(0)), 0), (round(line.x_at(h - 1)), h - 1), colors[role], 1)
    for role, line in enumerate(oracle):
        color = (40, 220, 40) if hits[role] else (255, 0, 255)
        cv2.line(output, (round(line.x_at(0)), 0), (round(line.x_at(h - 1)), h - 1), color, 2)
    cv2.putText(output, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def _save_overlay(output_dir: Path, tiles: list[np.ndarray]) -> Path:
    rows = []
    for start in range(0, len(tiles), 4):
        rows.append(np.hstack(tiles[start : start + 4]))
    canvas = np.vstack(rows)
    path = output_dir / "teacher-ceiling-24-case-overlay.png"
    cv2.imwrite(str(path), canvas)
    return path


def run(cohort_path: Path, r3_report_path: Path, mesh_root: Path, output_dir: Path) -> dict:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if len(cohort["episodes"]) != 24:
        raise ValueError("V1-E0 requires the frozen 24-episode R2 cohort")
    r3_report = json.loads(r3_report_path.read_text(encoding="utf-8"))
    r3_available = _available_episode_ids(r3_report, "b1")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields_dir = output_dir / "teacher-fields"
    fields_dir.mkdir(exist_ok=True)
    renderer = MeshDepthRenderer()
    rows = []
    tiles = []
    mesh_receipts: dict[str, dict] = {}
    for materialized in cohort["episodes"]:
        evaluator, episode_input, truth = _evaluator_episode(materialized)
        pose_a, pose_b, pose_audit = _source_poses(materialized)
        sequence = str(materialized["source"]["sequence"])
        mesh_path = mesh_root / sequence / f"{sequence}_3dod_mesh.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"missing official ARKitScenes mesh: {mesh_path}")
        if sequence not in mesh_receipts:
            mesh_receipts[sequence] = {"path": str(mesh_path.resolve()), "sha256": _sha256(mesh_path)}
        visible_anchors = [row for row in episode_input.exact_anchor_observations if row.visible]
        first_anchor = next(row for row in visible_anchors if row.frame_index == 0)
        second_anchor = next(row for row in visible_anchors if row.frame_index == episode_input.active_parallax_frame_index)
        local_frames = (
            renderer.teacher_frame(mesh_path, pose_a, episode_input.intrinsics),
            renderer.teacher_frame(mesh_path, pose_b, episode_input.intrinsics),
        )
        linked_left, linked_right, link_diagnostics = link_boundary_lines(local_frames[0], pose_a, pose_b, episode_input.intrinsics)
        frames = (
            local_frames[0],
            PrivilegedGeometryFrame(
                local_frames[1].depth_m,
                local_frames[1].normals_camera,
                local_frames[1].signed_depth_jump_m,
                local_frames[1].valid_mask,
                local_frames[1].boundary_heatmap,
                (linked_left, linked_right),
                {**local_frames[1].diagnostics, **link_diagnostics, "candidate_source": "VIEW_A_MESH_BOUNDARIES_PROJECTED_THROUGH_3D"},
            ),
        )
        provider = PrivilegedGeometryProvider(episode_input, truth, pose_a, pose_b, frames)
        result = _sage_lm(evaluator, provider)
        field_path = fields_dir / f"{evaluator.episode_id}.npz"
        np.savez_compressed(
            field_path,
            depth_m=np.stack([frame.depth_m for frame in frames]),
            normals_camera=np.stack([frame.normals_camera for frame in frames]),
            signed_depth_jump_m=np.stack([frame.signed_depth_jump_m for frame in frames]),
            valid_mask=np.stack([frame.valid_mask for frame in frames]),
            boundary_heatmap=np.stack([frame.boundary_heatmap for frame in frames]),
        )
        result["diagnostics"]["teacher_field_path"] = str(field_path.resolve())
        result["diagnostics"]["teacher_field_sha256"] = _sha256(field_path)
        oracle_a, oracle_b = oracle_pixel_lines(episode_input, truth, pose_a, pose_b)
        direct = result["diagnostics"]["direct_four_boundary_hits"]
        images = [cv2.imread(str(episode_input.rgb_frames[first_anchor.frame_index])), cv2.imread(str(episode_input.rgb_frames[second_anchor.frame_index]))]
        pair_ok = len(result["diagnostics"]["oracle_association_distances_px"]) == 4 and max(result["diagnostics"]["oracle_association_distances_px"]) <= LOCALIZATION_GATE_PX
        tile = np.hstack(
            [
                _draw_lines(images[0], frames[0], oracle_a, direct[:2], f"{evaluator.episode_id} A"),
                _draw_lines(images[1], frames[1], oracle_b, direct[2:], "PAIR" if pair_ok else "MISS"),
            ]
        )
        tiles.append(tile)
        rows.append(
            {
                "episode_id": evaluator.episode_id,
                "kind": evaluator.kind,
                "control": materialized["control"],
                "source": materialized["source"],
                "truth": materialized["truth"],
                "source_pose_audit": pose_audit,
                "baseline": _baseline(evaluator),
                ARM: result,
            }
        )
    overlay_path = _save_overlay(output_dir, tiles)
    teacher_available = _available_episode_ids({"rows": rows}, ARM)
    four_boundary = {
        row["episode_id"] for row in rows if all(row[ARM]["diagnostics"]["direct_four_boundary_hits"])
    }
    rescued = sorted(teacher_available - r3_available)
    retained = sorted(teacher_available & r3_available)
    lost = sorted(r3_available - teacher_available)
    metrics = _aggregate(row[ARM] for row in rows)
    diagnostics = _arm_diagnostics(rows, ARM)
    diagnostics.update(
        {
            "uses_metric_depth": True,
            "metric_depth_authority": "OFFICIAL_ARKITSCENES_3DOD_MESH_RAYCAST",
            "four_boundary_recall_at_8_count": len(four_boundary),
            "true_boundary_pair_available_count": len(teacher_available),
            "r3_true_boundary_pair_available_count": len(r3_available),
            "r3_missing_rescued_count": len(rescued),
            "r3_missing_rescued_episode_ids": rescued,
            "r3_pair_retained_count": len(retained),
            "r3_pair_retained_episode_ids": retained,
            "r3_available_lost_count": len(lost),
            "r3_available_lost_episode_ids": lost,
        }
    )
    geometry_count = diagnostics["geometry_output_count"]
    continue_gate = {
        "r3_missing_rescue_at_least_6_of_9": len(rescued) >= 6,
        "true_pair_strictly_exceeds_r3_15_of_24": len(teacher_available) > 15,
        "geometry_strictly_exceeds_r3_13_of_24": geometry_count > 13,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "OPENED_CURATED_DEVELOPMENT_TEACHER_CEILING",
        "experiment_label": "V1_E0_PRIVILEGED_GEOMETRY_TEACHER_CEILING",
        "question": "Can official ARKitScenes mesh geometry recover true aperture boundaries that R3 misses?",
        "cohort": {"path": str(cohort_path.resolve()), "episode_count": 24, "usage": "FIXED_EVALUATION_ONLY"},
        "teacher_contract": {
            "geometry_source": "OFFICIAL_ARKITSCENES_3DOD_MESH_REGISTERED_TO_OFFICIAL_CAMERA_POSE",
            "rgb_in_teacher": "NOT_USED",
            "deeplsd_canny_rgb_gradient_v1c_proxy_r3": "NOT_USED",
            "outputs": ["boundary_heatmap", "signed_depth_jump_m", "valid_mask"],
            "student_training": "NOT_RUN",
        },
        "mesh_receipts": mesh_receipts,
        "frozen_surfaces": {
            "top_k_candidates_per_role": TOP_K_PER_ROLE,
            "oracle_association_localization_gate_px": LOCALIZATION_GATE_PX,
            "triangulation": "UNCHANGED_R2",
            "r3_fusion": "NOT_RUN",
            "r6": "NOT_RUN",
            "b2": "NOT_RUN",
        },
        "metrics": {ARM: metrics},
        "observation_diagnostics": diagnostics,
        "continue_to_e1_gate": continue_gate,
        "decision": "ADVANCE_TO_V1_E1_HEATMAP_STUDENT" if all(continue_gate.values()) else "STOP_BEFORE_STUDENT",
        "overlay": {"path": str(overlay_path.resolve()), "sha256": _sha256(overlay_path)},
        "rows": rows,
        "claim_ceiling": "CURATED_CONSUMED_R2_DEVELOPMENT_COHORT_ARKITSCENES_MESH_TEACHER_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--r3-report", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cohort, args.r3_report, args.mesh_root, args.output_dir)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "diagnostics": report["observation_diagnostics"], "gate": report["continue_to_e1_gate"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
