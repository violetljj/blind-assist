#!/usr/bin/env python3
"""Execute GRAIL M0 oracle task/teacher validation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from grail_m0 import (
    Pose,
    Scene,
    choose_bbox_fixed_distance,
    choose_nearest_free,
    choose_oracle,
    counterfactual_judgements,
    interaction_pose_set,
    make_cohort,
    perturb_scene,
    pose_matches,
    shortest_path,
)


def evaluate(scenes: tuple[Scene, ...]) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for scene in scenes:
        truth = interaction_pose_set(scene)
        perturbed_truth = interaction_pose_set(perturb_scene(scene))
        oracle = choose_oracle(scene, truth)
        candidates = {
            "B0_BBOX_FIXED_DISTANCE": choose_bbox_fixed_distance(scene),
            "B1_MASK_DEPTH_NEAREST_FREE": choose_nearest_free(scene),
            "B3_ORACLE_SET_FIELD": oracle,
        }
        stability = False
        if truth and perturbed_truth:
            stability = any(pose_matches(pose, perturbed_truth) for pose in truth)
        counterfactuals = counterfactual_judgements(scene, truth)
        rows.append({
            "scene_id": scene.scene_id,
            "split": scene.split,
            "target_instance_id": scene.target.instance_id,
            "distractor_instance_id": scene.distractor.instance_id,
            "category": scene.target.category,
            "expected_pose_state": scene.expected_pose_state,
            "no_pose_reason": scene.no_pose_reason,
            "valid_pose_count": len(truth),
            "stable_under_geometry_perturbation": stability if truth else None,
            "counterfactuals": counterfactuals,
            "methods": {
                name: {
                    "committed": pose is not None,
                    "interaction_pose_success": pose_matches(pose, truth),
                    "closed_loop_completion": bool(
                        pose is not None
                        and pose_matches(pose, truth)
                        and shortest_path(scene, (pose.x, pose.y))
                    ),
                    "pose": None if pose is None else {"x": pose.x, "y": pose.y, "yaw_rad": pose.yaw_rad},
                }
                for name, pose in candidates.items()
            },
        })

    splits: dict[str, dict] = {}
    for split in ("DEVELOPMENT", "HELD_OUT"):
        selected = [row for row in rows if row["split"] == split]
        positives = [row for row in selected if row["expected_pose_state"] == "VALID_SET"]
        negatives = [row for row in selected if row["expected_pose_state"] == "NONE"]
        methods = {}
        for name in ("B0_BBOX_FIXED_DISTANCE", "B1_MASK_DEPTH_NEAREST_FREE", "B3_ORACLE_SET_FIELD"):
            methods[name] = {
                "interaction_pose_success": sum(row["methods"][name]["interaction_pose_success"] for row in positives),
                "interaction_pose_denominator": len(positives),
                "closed_loop_completion": sum(row["methods"][name]["closed_loop_completion"] for row in positives),
                "no_valid_pose_false_commit": sum(row["methods"][name]["committed"] for row in negatives),
                "no_valid_pose_denominator": len(negatives),
            }
        splits[split] = {
            "scenes": len(selected),
            "unique_scene_ids": len({row["scene_id"] for row in selected}),
            "unique_target_instances": len({row["target_instance_id"] for row in selected}),
            "valid_pose_scenes": len(positives),
            "none_scenes": len(negatives),
            "median_valid_pose_count": sorted(row["valid_pose_count"] for row in positives)[len(positives) // 2],
            "teacher_stability": sum(row["stable_under_geometry_perturbation"] is True for row in positives),
            "teacher_stability_denominator": len(positives),
            "counterfactual_rejections": dict(Counter(
                key for row in selected for key, value in row["counterfactuals"].items() if value
            )),
            "counterfactual_denominator_each": len(selected),
            "methods": methods,
        }
    held = splits["HELD_OUT"]
    oracle = held["methods"]["B3_ORACLE_SET_FIELD"]
    gates = {
        "all_scene_and_instance_ids_split_disjoint": not (
            {row["scene_id"] for row in rows if row["split"] == "DEVELOPMENT"}
            & {row["scene_id"] for row in rows if row["split"] == "HELD_OUT"}
        ) and not (
            {row["target_instance_id"] for row in rows if row["split"] == "DEVELOPMENT"}
            & {row["target_instance_id"] for row in rows if row["split"] == "HELD_OUT"}
        ),
        "held_out_positive_teacher_coverage_1_0": held["valid_pose_scenes"] == 24,
        "held_out_none_teacher_correct_1_0": oracle["no_valid_pose_false_commit"] == 0,
        "held_out_oracle_pose_success_1_0": oracle["interaction_pose_success"] == oracle["interaction_pose_denominator"],
        "held_out_oracle_closed_loop_1_0": oracle["closed_loop_completion"] == oracle["interaction_pose_denominator"],
        "held_out_teacher_stability_1_0": held["teacher_stability"] == held["teacher_stability_denominator"],
        "all_four_counterfactual_families_rejected": all(
            held["counterfactual_rejections"].get(key, 0) == held["counterfactual_denominator_each"]
            for key in (
                "same_class_wrong_instance_rejected", "correct_target_back_side_rejected",
                "free_but_goal_irrelevant_rejected", "face_target_but_unreachable_rejected",
            )
        ),
    }
    report = {
        "schema": "blindassist_grail_m0_oracle_interaction_pose_v0",
        "profile": "REVERSIBLE_EXPLORATION/DEVELOPMENT_STANDARD",
        "question": "Does a set-valued goal-relative interaction-pose task have a stable oracle geometry upper bound before student training?",
        "task_contract": {
            "factorization": ["referent", "affordance", "reachability", "visibility", "arrival"],
            "output": "set-valued metric position plus yaw, or NONE",
            "pose_tolerance_m": 0.50,
            "yaw_tolerance_deg": 20,
            "manual_frame_labels": 0,
            "oracle_referent": True,
            "oracle_geometry": True,
        },
        "cohort": {
            "procedural_metric_2_5d": True,
            "development_buildings": 12,
            "held_out_buildings": 36,
            "held_out_categories": ["counter", "door", "panel", "shelf"],
            "scene_disjoint": True,
            "instance_disjoint": True,
        },
        "splits": splits,
        "gates": gates,
        "terminal": "GRAIL_M0_PROCEDURAL_ORACLE_UPPER_BOUND_ESTABLISHED" if all(gates.values()) else "GRAIL_M0_ORACLE_UPPER_BOUND_NOT_ESTABLISHED",
        "claim_ceiling": "procedural metric 2.5D task and teacher mechanics only; no RGB, natural 3D scene, learned model, camera, Android, user, product, or safety evidence",
        "successor": "M1 frozen-encoder B0/B1/B2/GRAIL comparison on building-disjoint 3D-derived Development data" if all(gates.values()) else "STOP_BEFORE_M1_AND_REPAIR_TASK_OR_TEACHER",
    }
    return report, rows


def render_svg(path: Path, scene: Scene, row: dict) -> None:
    scale, pad = 72.0, 42.0
    height = int(8.0 * scale + 2 * pad)
    width = 1050
    def px(x: float) -> float: return pad + x * scale
    def py(y: float) -> float: return height - pad - y * scale
    truth = interaction_pose_set(scene)
    oracle_data = row["methods"]["B3_ORACLE_SET_FIELD"]["pose"]
    oracle = None if oracle_data is None else Pose(**oracle_data)
    path_points = shortest_path(scene, (oracle.x, oracle.y)) if oracle else None
    elements = [f'<rect width="{width}" height="{height}" fill="#0d1720"/>',
                f'<rect x="{pad}" y="{pad}" width="{8*scale}" height="{8*scale}" fill="#182631" stroke="#5b7083"/>']
    for obstacle in scene.obstacles:
        elements.append(f'<rect x="{px(obstacle.x0):.1f}" y="{py(obstacle.y1):.1f}" width="{(obstacle.x1-obstacle.x0)*scale:.1f}" height="{(obstacle.y1-obstacle.y0)*scale:.1f}" fill="#52616b"/>')
    for target, color in ((scene.target, "#ffca3a"), (scene.distractor, "#d45d79")):
        rect = target.footprint
        elements.append(f'<rect x="{px(rect.x0):.1f}" y="{py(rect.y1):.1f}" width="{(rect.x1-rect.x0)*scale:.1f}" height="{(rect.y1-rect.y0)*scale:.1f}" fill="{color}"/>')
    if path_points:
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in path_points)
        elements.append(f'<polyline points="{points}" fill="none" stroke="#3a86ff" stroke-width="6" opacity=".8"/>')
    for pose in truth:
        elements.append(f'<circle cx="{px(pose.x):.1f}" cy="{py(pose.y):.1f}" r="7" fill="#52d273" opacity=".78"/>')
    elements.append(f'<circle cx="{px(scene.start_x):.1f}" cy="{py(scene.start_y):.1f}" r="10" fill="#3a86ff"/>')
    if oracle:
        elements.append(f'<circle cx="{px(oracle.x):.1f}" cy="{py(oracle.y):.1f}" r="13" fill="none" stroke="#ffffff" stroke-width="4"/>')
    elements.extend([
        '<text x="26" y="28" fill="#ffffff" font-family="sans-serif" font-size="19">GRAIL M0 held-out scene: green=set-valued truth, white=oracle choice, blue=reachable path</text>',
        '<text x="26" y="52" fill="#b8cad6" font-family="sans-serif" font-size="15">yellow=goal referent, red=same-class distractor; neither bbox center nor object proximity defines arrival</text>',
    ])
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(elements) + '</svg>', encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = make_cohort()
    report, rows = evaluate(scenes)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = next(i for i, scene in enumerate(scenes) if scene.split == "HELD_OUT" and scene.expected_pose_state == "VALID_SET")
    render_svg(args.output_dir / "held_out_interaction_pose.svg", scenes[index], rows[index])
    print(json.dumps({"terminal": report["terminal"], "held_out": report["splits"]["HELD_OUT"], "gates": report["gates"], "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
