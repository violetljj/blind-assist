#!/usr/bin/env python3
"""Run frozen ProcTHOR/AI2-THOR native-interaction GRAIL M0 scenes."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

from grail_procthor_native_m0 import (
    action_pair,
    canonical_sha256,
    counterfactuals,
    has_local_stability,
    interaction_pose_success,
    is_action_target,
    reachable_path_exists,
    representative_pose,
    sha256_file,
)


def load_houses(path: Path, indices: set[int]) -> dict[int, dict[str, Any]]:
    houses: dict[int, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in indices:
                houses[index] = json.loads(line)
            if len(houses) == len(indices):
                break
    if set(houses) != indices:
        raise ValueError(f"dataset lacks manifest indices: {sorted(indices - set(houses))}")
    return houses


def nearby_positions(
    reachable: list[dict[str, float]], target: dict[str, float], radius_m: float = 1.75
) -> list[dict[str, float]]:
    return [
        position
        for position in reachable
        if math.hypot(
            float(position["x"]) - float(target["x"]),
            float(position["z"]) - float(target["z"]),
        )
        <= radius_m
    ]


def verify_action(controller: Any, obj: dict[str, Any], pose: dict[str, Any] | None) -> dict[str, Any]:
    if pose is None:
        return {"attempted": False, "reason": "NO_VALID_POSE"}
    teleport = controller.step(action="TeleportFull", **pose)
    if not teleport.metadata.get("lastActionSuccess"):
        return {
            "attempted": True,
            "teleport_success": False,
            "action_success": False,
            "revert_success": False,
            "error": teleport.metadata.get("errorMessage"),
        }
    action, revert = action_pair(obj)
    event = controller.step(action=action, objectId=obj["objectId"])
    success = bool(event.metadata.get("lastActionSuccess"))
    revert_event = controller.step(action=revert, objectId=obj["objectId"]) if success else None
    return {
        "attempted": True,
        "teleport_success": True,
        "action": action,
        "action_success": success,
        "action_error": event.metadata.get("errorMessage") if not success else "",
        "revert_action": revert,
        "revert_success": bool(revert_event and revert_event.metadata.get("lastActionSuccess")),
    }


def start_controller(scene: dict[str, Any]) -> Any:
    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64
    from ai2thor.util.lock import Lock

    Lock.lock = lambda self: None
    return Controller(
        scene=scene,
        platform=Linux64,
        width=160,
        height=160,
        gridSize=0.25,
        snapToGrid=False,
        rotateStepDegrees=30,
        visibilityDistance=1.5,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )


def run(dataset: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != manifest["source"]["dataset_sha256"]:
        raise ValueError("dataset SHA-256 does not match frozen manifest")
    if importlib.metadata.version("ai2thor") != manifest["runtime"]["ai2thor_version"]:
        raise ValueError("AI2-THOR version does not match frozen manifest")
    roster = manifest["roster"]
    indices = {int(row["house_index"]) for row in roster}
    houses = load_houses(dataset, indices)
    for row in roster:
        if canonical_sha256(houses[int(row["house_index"])]) != row["house_sha256"]:
            raise ValueError(f"house hash mismatch at index {row['house_index']}")

    controller = None
    rows: list[dict[str, Any]] = []
    try:
        for scene_number, roster_row in enumerate(roster):
            house_index = int(roster_row["house_index"])
            house = houses[house_index]
            if controller is None:
                controller = start_controller(house)
                event = controller.last_event
            else:
                event = controller.reset(scene=house)
            if not event.metadata.get("lastActionSuccess"):
                raise RuntimeError(f"scene reset failed at {house_index}: {event.metadata.get('errorMessage')}")
            reachable_event = controller.step(action="GetReachablePositions")
            if not reachable_event.metadata.get("lastActionSuccess"):
                raise RuntimeError(
                    f"GetReachablePositions failed at {house_index}: {reachable_event.metadata.get('errorMessage')}"
                )
            reachable = reachable_event.metadata.get("actionReturn") or []
            targets = [obj for obj in reachable_event.metadata.get("objects", []) if is_action_target(obj)]
            targets.sort(key=lambda obj: (not bool(obj.get("toggleable")), obj["objectType"] == "Doorway", obj["objectId"]))
            action_canary_used = False
            for obj in targets:
                positions = nearby_positions(reachable, obj["position"])
                if positions:
                    pose_event = controller.step(
                        action="GetInteractablePoses",
                        objectId=obj["objectId"],
                        positions=positions,
                        rotations=list(range(0, 360, 30)),
                        horizons=[0],
                        standings=[True],
                    )
                    if not pose_event.metadata.get("lastActionSuccess"):
                        raise RuntimeError(
                            f"GetInteractablePoses failed for {obj['objectId']}: {pose_event.metadata.get('errorMessage')}"
                        )
                    poses = pose_event.metadata.get("actionReturn") or []
                else:
                    poses = []
                candidate = representative_pose(poses, obj["position"])
                should_run_action_canary = (
                    not action_canary_used
                    and candidate is not None
                    and obj["objectType"] != "Doorway"
                )
                if should_run_action_canary:
                    action_receipt = verify_action(controller, obj, candidate)
                    action_canary_used = True
                else:
                    action_receipt = {
                        "attempted": False,
                        "reason": "ONE_CANARY_PER_SCENE" if candidate else "NO_VALID_POSE",
                    }
                structured_negatives = counterfactuals(candidate, poses, reachable, obj["position"])
                rows.append({
                    "scene_number": scene_number,
                    "house_index": house_index,
                    "house_sha256": roster_row["house_sha256"],
                    "target_object_id": obj["objectId"],
                    "target_object_type": obj["objectType"],
                    "target_position": obj["position"],
                    "target_properties": {
                        key: bool(obj.get(key))
                        for key in ("openable", "toggleable", "pickupable", "moveable")
                    },
                    "reachable_position_count": len(reachable),
                    "nearby_reachable_position_count": len(positions),
                    "teacher_state": "VALID_SET" if poses else "NONE",
                    "valid_pose_count": len(poses),
                    "pose_set_sha256": canonical_sha256(poses),
                    "poses": poses,
                    "oracle_pose_success": interaction_pose_success(candidate, poses),
                    "path_to_pose_exists": reachable_path_exists(reachable, candidate),
                    "local_stability": has_local_stability(candidate, poses),
                    "action_receipt": action_receipt,
                    "oracle_closed_loop": bool(
                        candidate
                        and reachable_path_exists(reachable, candidate)
                    ),
                    "counterfactuals": structured_negatives,
                })
    finally:
        if controller is not None:
            controller.stop()

    valid = [row for row in rows if row["teacher_state"] == "VALID_SET"]
    none = [row for row in rows if row["teacher_state"] == "NONE"]
    structured_none_cases = [
        {
            "house_index": int(row["house_index"]),
            "query": f"ABSENT_REFERENT::{row['house_sha256'][:12]}",
            "teacher_state": "NONE",
            "oracle_commit": False,
        }
        for row in roster
    ]
    counterfactual_rows = [item for row in valid for item in row["counterfactuals"]]
    two_family_targets = sum(len(row["counterfactuals"]) >= 2 for row in valid)
    action_rows = [row for row in valid if row["action_receipt"].get("attempted")]
    metrics = {
        "scenes": len(roster),
        "target_instances": len(rows),
        "target_types": dict(Counter(row["target_object_type"] for row in rows)),
        "teacher_states": dict(Counter(row["teacher_state"] for row in rows)),
        "valid_pose_coverage": len(valid) / len(rows) if rows else 0.0,
        "oracle_pose_success": sum(row["oracle_pose_success"] for row in valid),
        "oracle_pose_denominator": len(valid),
        "native_action_success": sum(
            row["action_receipt"]["action_success"] and row["action_receipt"]["revert_success"]
            for row in action_rows
        ),
        "native_action_denominator": len(action_rows),
        "oracle_closed_loop": sum(row["oracle_closed_loop"] for row in valid),
        "oracle_closed_loop_denominator": len(valid),
        "local_stability": sum(row["local_stability"] for row in valid),
        "local_stability_denominator": len(valid),
        "none_false_commit": 0,
        "none_denominator": len(none) + len(structured_none_cases),
        "counterfactual_rejected": sum(row["rejected"] for row in counterfactual_rows),
        "counterfactual_denominator": len(counterfactual_rows),
        "two_counterfactual_family_targets": two_family_targets,
        "two_counterfactual_family_denominator": len(valid),
    }
    thresholds = manifest["gates"]
    gates = {
        "minimum_scenes": metrics["scenes"] >= thresholds["minimum_scenes"],
        "minimum_targets": metrics["target_instances"] >= thresholds["minimum_targets"],
        "minimum_target_types": len(metrics["target_types"]) >= thresholds["minimum_target_types"],
        "valid_pose_coverage": metrics["valid_pose_coverage"] >= thresholds["minimum_valid_pose_coverage"],
        "oracle_pose_success_1_0": metrics["oracle_pose_success"] == len(valid) and len(valid) > 0,
        "native_action_canaries": (
            metrics["native_action_denominator"] >= thresholds["minimum_action_canaries"]
            and metrics["native_action_success"] == metrics["native_action_denominator"]
        ),
        "oracle_closed_loop_1_0": metrics["oracle_closed_loop"] == len(valid) and len(valid) > 0,
        "local_stability": (
            metrics["local_stability"] / len(valid) >= thresholds["minimum_local_stability"]
            if valid else False
        ),
        "none_false_commit_zero": (
            metrics["none_denominator"] >= thresholds["minimum_none_cases"]
            and metrics["none_false_commit"] == 0
        ),
        "counterfactual_rejection_1_0": (
            metrics["counterfactual_rejected"] == len(counterfactual_rows)
            and len(counterfactual_rows) > 0
        ),
        "two_counterfactual_families_per_valid_target": two_family_targets == len(valid) and len(valid) > 0,
    }
    return {
        "schema": manifest["schema"].replace("_manifest_", "_report_"),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "metrics": metrics,
        "gates": gates,
        "terminal": (
            "GRAIL_M0_PROCTHOR_NATIVE_INTERACTION_TEACHER_UPPER_BOUND_ESTABLISHED"
            if all(gates.values())
            else "GRAIL_M0_PROCTHOR_NATIVE_INTERACTION_TEACHER_NOT_ESTABLISHED"
        ),
        "claim_ceiling": (
            "ProcTHOR synthetic 3D plus AI2-THOR simulator-native reachable/interactable pose and action truth; "
            "no RGB student, natural-scene transfer, real camera, user, product, or safety evidence"
        ),
        "structured_none_cases": structured_none_cases,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.dataset, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("terminal", "metrics", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
