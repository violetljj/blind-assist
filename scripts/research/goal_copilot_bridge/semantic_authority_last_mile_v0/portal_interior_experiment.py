"""Run SAGE-LM V1-F anchor-conditioned portal-interior teacher ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .active_parallax_experiment import _available_episode_ids
from .experiment import _aggregate
from .observation import ApertureObservation
from .portal_interior_teacher import PortalInteriorPrediction, infer_portal_interior
from .privileged_geometry_teacher import MeshDepthRenderer
from .rgb_experiment import _baseline, _sage_lm
from .task_boundary_field_experiment import LOCALIZATION_GATE_PX
from .two_view_experiment import _arm_diagnostics, _evaluator_episode, _source_poses
from .two_view_observation import _line_distance, oracle_pixel_lines

SCHEMA_VERSION = "sage_lm_v1f_anchor_conditioned_portal_interior_field"
ARM = "v1f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PortalInteriorProvider:
    arm_name = "SAGE_LM_V1F_ANCHOR_CONDITIONED_PORTAL_INTERIOR_FIELD"

    def __init__(self, episode_input, prediction: PortalInteriorPrediction, oracle_lines) -> None:
        self.input = episode_input
        self.prediction = prediction
        self.diagnostics = dict(prediction.diagnostics)
        distances: list[float] = []
        hits = [False, False, False, False]
        if prediction.derived_boundary_lines is not None:
            pools = (
                (prediction.derived_boundary_lines[0][0],),
                (prediction.derived_boundary_lines[0][1],),
                (prediction.derived_boundary_lines[1][0],),
                (prediction.derived_boundary_lines[1][1],),
            )
            for index, (pool, oracle) in enumerate(zip(pools, (*oracle_lines[0], *oracle_lines[1]))):
                distance = _line_distance(pool[0], oracle, episode_input.intrinsics.height)
                distances.append(distance)
                hits[index] = distance <= LOCALIZATION_GATE_PX
        self.diagnostics.update(
            {
                "uses_rgb": False,
                "uses_privileged_mesh_depth": True,
                "ray_classes": ["SUPPORT_PLANE_HIT", "BEHIND_VALID_SPACE", "MESH_UNKNOWN"],
                "portal_outputs": {
                    "center_bearing_rad": prediction.center_bearing_rad,
                    "range_m": prediction.range_m,
                    "width_m": prediction.width_m,
                    "target_front_waypoint_world_m": prediction.target_front_waypoint_world_m,
                },
                "legacy_oracle_association_distances_px": distances,
                "direct_four_boundary_hits": hits,
                "teacher_view_diagnostics": [field.diagnostics for field in prediction.views],
            }
        )
        if prediction.center_x_m is not None:
            self.diagnostics["geometry"] = {
                "center_x_m": prediction.center_x_m,
                "width_m": prediction.width_m,
                "range_m": prediction.range_m,
                "confidence": prediction.confidence,
            }

    def observe(self) -> ApertureObservation:
        p = self.prediction
        if p.center_x_m is None:
            return ApertureObservation(True, None, None, None, 0.0, 0.0, 0.0, 0.0)
        return ApertureObservation(
            True,
            p.center_x_m,
            p.width_m,
            p.range_m,
            p.confidence,
            p.confidence,
            p.confidence,
            p.confidence,
        )


def run(cohort_path: Path, r3_report_path: Path, mesh_root: Path, output_dir: Path) -> dict:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if len(cohort["episodes"]) != 24:
        raise ValueError("V1-F requires exactly 24 fresh Development episodes")
    r3_report = json.loads(r3_report_path.read_text(encoding="utf-8"))
    r3_available = _available_episode_ids(r3_report, "b1")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields_dir = output_dir / "portal-fields"
    fields_dir.mkdir(exist_ok=True)
    renderer = MeshDepthRenderer()
    rows = []
    mesh_receipts: dict[str, dict] = {}
    for materialized in cohort["episodes"]:
        evaluator, episode_input, truth = _evaluator_episode(materialized)
        pose_a, pose_b, pose_audit = _source_poses(materialized)
        sequence = str(materialized["source"]["sequence"])
        mesh_path = mesh_root / sequence / f"{sequence}_3dod_mesh.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"missing official ARKitScenes mesh: {mesh_path}")
        mesh_receipts.setdefault(sequence, {"path": str(mesh_path.resolve()), "sha256": _sha256(mesh_path)})
        anchors = [anchor for anchor in episode_input.exact_anchor_observations if anchor.visible]
        anchor_a = next(anchor for anchor in anchors if anchor.frame_index == 0)
        depths = (
            renderer.render(mesh_path, pose_a, episode_input.intrinsics)[0],
            renderer.render(mesh_path, pose_b, episode_input.intrinsics)[0],
        )
        prediction = infer_portal_interior(depths, (pose_a, pose_b), episode_input.intrinsics, anchor_a.bbox_xyxy)
        oracle = oracle_pixel_lines(episode_input, truth, pose_a, pose_b)
        provider = PortalInteriorProvider(episode_input, prediction, oracle)
        result = _sage_lm(evaluator, provider)
        field_path = fields_dir / f"{evaluator.episode_id}.npz"
        if prediction.views:
            np.savez_compressed(
                field_path,
                soft_mask=np.stack([field.soft_mask for field in prediction.views]),
                plane_hit_mask=np.stack([field.plane_hit_mask for field in prediction.views]),
                behind_free_mask=np.stack([field.behind_free_mask for field in prediction.views]),
                mesh_unknown_mask=np.stack([field.unknown_mask for field in prediction.views]),
                selected_component_mask=np.stack([field.component_mask for field in prediction.views]),
            )
            result["diagnostics"]["portal_field_path"] = str(field_path.resolve())
            result["diagnostics"]["portal_field_sha256"] = _sha256(field_path)
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
    teacher_available = {
        row["episode_id"]
        for row in rows
        if len(row[ARM]["diagnostics"]["legacy_oracle_association_distances_px"]) == 4
        and max(row[ARM]["diagnostics"]["legacy_oracle_association_distances_px"]) <= LOCALIZATION_GATE_PX
    }
    geometry_ids = {row["episode_id"] for row in rows if row[ARM]["observation"]["center_x_m"] is not None}
    rescued = sorted(teacher_available - r3_available)
    retained = sorted(teacher_available & r3_available)
    lost = sorted(r3_available - teacher_available)
    diagnostics = _arm_diagnostics(rows, ARM)
    diagnostics.update(
        {
            "true_boundary_pair_available_count": len(teacher_available),
            "r3_true_boundary_pair_available_count": len(r3_available),
            "r3_missing_rescued_count": len(rescued),
            "r3_missing_rescued_episode_ids": rescued,
            "r3_pair_retained_count": len(retained),
            "r3_pair_retention_rate": len(retained) / len(r3_available) if r3_available else None,
            "r3_available_lost_count": len(lost),
            "r3_available_lost_episode_ids": lost,
            "geometry_episode_ids": sorted(geometry_ids),
        }
    )
    required_retained = math.ceil(0.8 * len(r3_available))
    gate = {
        "true_pair_exceeds_r3_by_at_least_3": len(teacher_available) >= len(r3_available) + 3,
        "geometry_exceeds_r3_by_at_least_3": len(geometry_ids) >= int(r3_report["observation_diagnostics"]["geometry_output_count"]) + 3,
        "r3_success_retention_at_least_80pct": len(retained) >= required_retained,
        "r3_missing_rescue_at_least_3": len(rescued) >= 3,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "REVERSIBLE_EXPLORATION_FRESH_FRAME_DEVELOPMENT",
        "experiment_label": "V1_F_ANCHOR_CONDITIONED_PORTAL_INTERIOR_FIELD_TEACHER_CEILING",
        "question": "Does an anchor-conditioned connected behind-plane free-space field materially exceed frozen R3 on fresh ARKitScenes frames?",
        "cohort": {"path": str(cohort_path.resolve()), "sha256": _sha256(cohort_path), "episode_count": 24, "usage": "FRESH_FRAME_DEVELOPMENT"},
        "r3_same_cohort": {"path": str(r3_report_path.resolve()), "sha256": _sha256(r3_report_path)},
        "teacher_contract": {
            "rgb_in_teacher": "NOT_USED",
            "geometry_source": "OFFICIAL_ARKITSCENES_3DOD_MESH_RAYCAST_PLUS_OFFICIAL_POSE_INTRINSICS",
            "conditioning": "SEMANTIC_ANCHOR_SUPPORT_PLANE",
            "ray_classes": ["SUPPORT_PLANE_HIT", "BEHIND_VALID_SPACE", "MESH_UNKNOWN"],
            "selection": "ANCHOR_NEAREST_CROSS_VIEW_CONSISTENT_CONNECTED_BEHIND_SPACE",
            "primary_outputs": ["portal_interior_soft_mask", "center_bearing_rad", "range_m", "width_m", "target_front_waypoint_world_m"],
            "legacy_boundaries": "DERIVED_FOR_FROZEN_EVALUATOR_ONLY",
            "student_training": "NOT_RUN",
        },
        "mesh_receipts": mesh_receipts,
        "metrics": {"bbox_center_scale": _aggregate(row["baseline"] for row in rows), ARM: _aggregate(row[ARM] for row in rows)},
        "observation_diagnostics": diagnostics,
        "advance_gate": gate,
        "decision": "ADVANCE_TO_STUDENT_DESIGN" if all(gate.values()) else "STOP_BEFORE_STUDENT",
        "forbidden_work": {"faro": "NOT_RUN", "e0_student": "NOT_RUN", "r3_fusion": "NOT_RUN", "r6": "NOT_RUN", "b2": "NOT_RUN"},
        "rows": rows,
        "claim_ceiling": "FRESH_FRAME_CURATED_ARKITSCENES_DEVELOPMENT_MESH_TEACHER_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--r3-report", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cohort, args.r3_report, args.mesh_root, args.output_dir)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostics": report["observation_diagnostics"], "gate": report["advance_gate"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
