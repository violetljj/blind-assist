from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from l10_abotn_poibench_source_audit import HF_REVISION, hf_file, hf_tree, inspect_episode


def load_episodes() -> tuple[list[dict[str, Any]], str]:
    episodes: list[dict[str, Any]] = []
    manifest = hashlib.sha256()
    scenes = sorted(
        (row for row in hf_tree("annotations") if row.get("type") == "directory"),
        key=lambda row: row["path"],
    )
    for scene in scenes:
        files = sorted(hf_tree(scene["path"]), key=lambda row: row["path"])
        for row in files:
            path = row["path"]
            if row.get("type") != "file" or not path.endswith(".json"):
                continue
            payload = hf_file(path)
            episode = inspect_episode(path, payload)
            episodes.append(episode)
            manifest.update(path.encode())
            manifest.update(b"\0")
            manifest.update(episode["sha256"].encode())
            manifest.update(b"\0")
            manifest.update(str(episode["bytes"]).encode())
            manifest.update(b"\n")
    return episodes, manifest.hexdigest()


def _endpoint(row: dict[str, Any]) -> tuple[float, float]:
    value = row.get("end_point")
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"missing 2D endpoint: {row.get('path')}")
    return float(value[0]), float(value[1])


def _episode_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_path": row["path"],
        "annotation_sha256": row["sha256"],
        "endpoint_id": str(row["endpoint_id"]),
        "goal_label": row["goal_label"],
        "endpoint_xy_evaluator_private": list(_endpoint(row)),
    }


def freeze_cohort(episodes: list[dict[str, Any]], manifest_sha256: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in episodes:
        grouped.setdefault(row["scene_id"], []).append(row)
    if len(grouped) < 3:
        raise ValueError("at least three scenes are required")

    ranked = sorted(grouped, key=lambda scene_id: (len(grouped[scene_id]), scene_id))
    selected_ids = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    roles = ("LOW_POI_DENSITY", "MEDIAN_POI_DENSITY", "HIGH_POI_DENSITY")
    selected: list[dict[str, Any]] = []
    for role, scene_id in zip(roles, selected_ids, strict=True):
        rows = sorted(grouped[scene_id], key=lambda row: row["path"])
        candidates = []
        for left, right in itertools.combinations(rows, 2):
            candidates.append(
                (
                    math.dist(_endpoint(left), _endpoint(right)),
                    left["path"],
                    right["path"],
                    left,
                    right,
                )
            )
        distance, _, _, left, right = min(candidates)
        selected.append(
            {
                "selection_role": role,
                "scene_id": scene_id,
                "scene_episode_count": len(rows),
                "scene_reference_pose_count_median": statistics.median(
                    row["trajectory_points"] for row in rows
                ),
                "frozen_pair": [_episode_ref(left), _episode_ref(right)],
                "endpoint_separation_analysis_only": distance,
                "pair_status": "SIBLING_CANDIDATE_REQUIRES_PIXEL_BLIND_OWNERSHIP_ADJUDICATION",
                "pair_failure_rule": "NOT_EVALUABLE_DO_NOT_SUBSTITUTE_AFTER_PIXELS_ARE_OPENED",
            }
        )

    for index, scene in enumerate(selected):
        local_goals = {row["goal_label"] for row in grouped[scene["scene_id"]]}
        foreign = selected[(index + 1) % len(selected)]["frozen_pair"]
        control = next((row for row in foreign if row["goal_label"] not in local_goals), None)
        if control is None:
            raise ValueError(f"no cross-scene target-absent control for {scene['scene_id']}")
        scene["target_absent_control"] = {
            "source_scene_id": selected[(index + 1) % len(selected)]["scene_id"],
            "goal_label": control["goal_label"],
            "target_present_in_local_annotation_roster": False,
            "control_status": "TARGET_ABSENT_CANDIDATE_REQUIRES_PIXEL_BLIND_ADJUDICATION",
        }

    return {
        "schema": "blindassist_l10_abotn_cohort_freeze_v1",
        "status": "FROZEN_METADATA_ONLY_PIXELS_UNOPENED_NOT_AUTHORITY_ADMITTED",
        "source": {
            "huggingface_dataset": "acvlab/ABotN-POIBench",
            "huggingface_revision": HF_REVISION,
            "annotation_manifest_sha256": manifest_sha256,
        },
        "selection": {
            "scene_rule": "lowest, median, and highest POI-count scenes; scene_id breaks ties",
            "pair_rule": "minimum 2D endpoint separation; annotation path breaks ties",
            "pair_semantics": "candidate generation only; separation is never authority or same-facade truth",
            "substitution_rule": "none after any selected-scene pixels are opened",
            "distance_role": "analysis_only_never_transition_or_admission",
        },
        "cohort": selected,
        "admission_gate": (
            "Each frozen pair must receive independently adjudicated exact facade, entrance, ownership, "
            "same-facade sibling, target-absence, endpoint visibility/orientation/stand-off, and visual handoff truth. "
            "A failed or unknown field makes that scene NOT_EVALUABLE and cannot trigger substitution."
        ),
        "runtime_boundary": (
            "endpoint coordinates, endpoint IDs, source-scene control identity, ownership labels, and "
            "terminal truth remain evaluator-private"
        ),
        "claim_boundary": (
            "This freeze is a preregistered capture/adjudication cohort, not evidence of sibling status, "
            "facade or entrance binding, action utility, recovery, arrival, handoff, user benefit, or safety."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes, manifest_sha256 = load_episodes()
    result = freeze_cohort(episodes, manifest_sha256)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
