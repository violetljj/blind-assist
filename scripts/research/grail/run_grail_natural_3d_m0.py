#!/usr/bin/env python3
"""Run GRAIL M0 derived-teacher transfer on frozen ARKitScenes scenes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from grail_natural_3d_m0 import (
    interaction_pose_teacher,
    load_binary_ply_vertices,
    load_objects,
    pose_matches,
    shortest_path,
    build_scene_grid,
)


SCENE_IDS = (
    "40777060", "40777069", "40777073", "40958737",
    "40958764", "41007603", "40776203", "41045408",
)
DEVELOPMENT_SCENES: set[str] = set()
HELD_OUT_SCENES = set(SCENE_IDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(mesh_root: Path, annotation_root: Path, scene_ids: tuple[str, ...] = SCENE_IDS) -> dict:
    rows = []
    for scene_id in scene_ids:
        mesh = mesh_root / scene_id / f"{scene_id}_3dod_mesh.ply"
        annotation = annotation_root / f"{scene_id}_3dod_annotation.json"
        if not mesh.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"missing natural-3D source for {scene_id}")
        vertices = load_binary_ply_vertices(mesh)
        grid = build_scene_grid(scene_id, vertices)
        objects = load_objects(annotation)
        for obj in objects:
            output = interaction_pose_teacher(grid, obj)
            selected = output.poses[0] if output.poses else None
            rows.append({
                "scene_id": scene_id,
                "split": "DEVELOPMENT" if scene_id in DEVELOPMENT_SCENES else "HELD_OUT",
                "target_uid": obj.uid,
                "label": obj.label,
                "floor_z_m": grid.floor_z,
                "largest_free_component_cells": (
                    int((grid.component == grid.largest_component).sum())
                    if grid.largest_component != 0 else 0
                ),
                "teacher_state": output.state,
                "chosen_face": output.chosen_face,
                "face_pose_counts": list(output.face_counts),
                "valid_pose_count": len(output.poses),
                "oracle_pose_success": pose_matches(selected, output.poses),
                "oracle_closed_loop": bool(selected and shortest_path(grid, (selected.x, selected.y))),
            })
    split_reports = {}
    for split in ("DEVELOPMENT", "HELD_OUT"):
        selected = [row for row in rows if row["split"] == split]
        valid = [row for row in selected if row["teacher_state"] == "VALID_SET"]
        split_reports[split] = {
            "scenes": len({row["scene_id"] for row in selected}),
            "target_instances": len(selected),
            "labels": dict(Counter(row["label"] for row in selected)),
            "teacher_states": dict(Counter(row["teacher_state"] for row in selected)),
            "valid_pose_coverage": len(valid) / len(selected) if selected else 0.0,
            "oracle_pose_success": sum(row["oracle_pose_success"] for row in valid),
            "oracle_pose_denominator": len(valid),
            "oracle_closed_loop": sum(row["oracle_closed_loop"] for row in valid),
            "oracle_closed_loop_denominator": len(valid),
        }
    held = split_reports["HELD_OUT"]
    gates = {
        "scene_split_disjoint": not (DEVELOPMENT_SCENES & HELD_OUT_SCENES),
        "at_least_three_held_out_scenes": held["scenes"] >= 3,
        "at_least_eight_held_out_targets": held["target_instances"] >= 8,
        "held_out_valid_pose_coverage_at_least_0_50": held["valid_pose_coverage"] >= 0.50,
        "held_out_oracle_pose_success_1_0": held["oracle_pose_success"] == held["oracle_pose_denominator"] and held["oracle_pose_denominator"] > 0,
        "held_out_oracle_closed_loop_1_0": held["oracle_closed_loop"] == held["oracle_closed_loop_denominator"] and held["oracle_closed_loop_denominator"] > 0,
    }
    return {
        "schema": "blindassist_grail_natural_3d_m0_v0",
        "profile": "REVERSIBLE_EXPLORATION/DEVELOPMENT_STANDARD",
        "source": {
            "dataset": "ARKitScenes Training 3DOD",
            "official_repository": "https://github.com/apple/ARKitScenes",
            "scene_ids": list(scene_ids),
            "v1f_scene_overlap": [],
            "mesh_sha256": {scene_id: sha256(mesh_root / scene_id / f"{scene_id}_3dod_mesh.ply") for scene_id in scene_ids},
            "annotation_sha256": {scene_id: sha256(annotation_root / f"{scene_id}_3dod_annotation.json") for scene_id in scene_ids},
        },
        "teacher_contract": {
            "source_native": ["metric mesh", "semantic instance label", "oriented 3D bounding box"],
            "derived_proxy": ["floor plane", "walkable component", "collision clearance", "line of sight", "functional face from asymmetric local free-space support"],
            "functional_side_source_truth": False,
            "navmesh_source_truth": False,
            "ambiguous_face_handling": "AMBIGUOUS/no pose",
        },
        "splits": split_reports,
        "gates": gates,
        "terminal": "GRAIL_M0_NATURAL_3D_DERIVED_TEACHER_UPPER_BOUND_ESTABLISHED" if all(gates.values()) else "GRAIL_M0_NATURAL_3D_DERIVED_TEACHER_NOT_ESTABLISHED",
        "claim_ceiling": "ARKitScenes natural metric mesh plus source OBB and derived functional-side/navmesh proxy; no source-native affordance truth, RGB student, camera transfer, user, product, or safety evidence",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", action="append")
    args = parser.parse_args()
    report = run(args.mesh_root, args.annotation_root, tuple(args.scene_id) if args.scene_id else SCENE_IDS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"terminal": report["terminal"], "splits": report["splits"], "gates": report["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
