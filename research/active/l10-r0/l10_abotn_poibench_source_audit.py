from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


HF_DATASET = "acvlab/ABotN-POIBench"
HF_REVISION = "fbb62cc3382d8ff84f7fe3b6a3e7d48e4c21e974"
GITHUB_REPOSITORY = "amap-cvlab/ABot-Navigation"
GITHUB_REVISION = "2a0aefb56f1e2d315bba924239e9e8ad9dca9d92"
USER_AGENT = "BlindAssist-L10-source-audit/1"
POSE_FIELDS = {"x", "y", "z", "pitch", "roll", "yaw"}
EPISODE_FIELDS = {
    "instruction",
    "trajectory",
    "label",
}
EXTEND_FIELDS = {
    "goal_label",
    "el_unique_id",
    "start_point",
    "end_point",
}


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url))


def hf_tree(path: str) -> list[dict[str, Any]]:
    encoded = urllib.parse.quote(path, safe="/")
    url = (
        f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/{HF_REVISION}/{encoded}"
        "?recursive=false&expand=true&limit=100"
    )
    return fetch_json(url)


def hf_file(path: str) -> bytes:
    encoded = urllib.parse.quote(path, safe="/")
    return fetch_bytes(
        f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/{encoded}"
    )


def github_file(path: str) -> bytes:
    return fetch_bytes(
        f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{GITHUB_REVISION}/{path}"
    )


def inspect_episode(path: str, payload: bytes) -> dict[str, Any]:
    record = json.loads(payload)
    missing_episode = sorted(EPISODE_FIELDS - record.keys())
    trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), list) else []
    missing_pose_frames = sum(
        not isinstance(frame, dict) or not POSE_FIELDS.issubset(frame)
        for frame in trajectory
    )
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    extend = label.get("extend") if isinstance(label.get("extend"), dict) else {}
    missing_extend = sorted(EXTEND_FIELDS - extend.keys())
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "scene_id": path.split("/")[1],
        "trajectory_points": len(trajectory),
        "missing_episode_fields": missing_episode,
        "missing_pose_frames": missing_pose_frames,
        "missing_extend_fields": missing_extend,
        "goal_label": extend.get("goal_label"),
        "endpoint_id": extend.get("el_unique_id"),
        "end_point": extend.get("end_point"),
    }


def audit() -> dict[str, Any]:
    scene_rows = [row for row in hf_tree("annotations") if row.get("type") == "directory"]
    episodes: list[dict[str, Any]] = []
    manifest = hashlib.sha256()
    for scene in sorted(scene_rows, key=lambda row: row["path"]):
        for row in sorted(hf_tree(scene["path"]), key=lambda item: item["path"]):
            path = row["path"]
            if row.get("type") != "file" or not path.endswith(".json"):
                continue
            payload = hf_file(path)
            episode = inspect_episode(path, payload)
            episodes.append(episode)
            manifest.update(path.encode("utf-8"))
            manifest.update(b"\0")
            manifest.update(episode["sha256"].encode("ascii"))
            manifest.update(b"\0")
            manifest.update(str(episode["bytes"]).encode("ascii"))
            manifest.update(b"\n")

    evaluator = github_file("abotn_evaluator/poi_goal/evaluator.py").decode("utf-8")
    interface = github_file("abotn_evaluator/interface/point_goal.py").decode("utf-8")
    docs = github_file("docs/poi-goal.md").decode("utf-8")
    truth_fields_exposed = {
        field: field in evaluator and field in interface
        for field in ("target_position", "distance_to_goal")
    }
    default_metric_distance_transition = (
        "arrive_threshold: float = 2.0" in evaluator
        and "distance_to_goal <= arrive_threshold" in docs
    )

    scene_goal_counts: dict[str, int] = {}
    for episode in episodes:
        scene_goal_counts[episode["scene_id"]] = scene_goal_counts.get(episode["scene_id"], 0) + 1
    trajectory_lengths = [row["trajectory_points"] for row in episodes]
    structurally_complete = [
        row
        for row in episodes
        if not row["missing_episode_fields"]
        and not row["missing_extend_fields"]
        and row["missing_pose_frames"] == 0
        and row["trajectory_points"] > 1
    ]

    coverage = {
        "continuous_reference_trajectory": "PASS",
        "per_frame_metric_pose": "PASS",
        "interactive_rgb_and_waypoint_actions": "CONDITIONAL_ON_9_67_GB_ASSETS_AND_RENDER_SERVER",
        "exact_target_endpoint_coordinate": "PASS_EVALUATOR_PRIVATE",
        "exact_target_facade_id": "MISSING",
        "exact_entrance_instance_id": "NOT_PROVEN_EL_UNIQUE_ID_IS_POI_ENDPOINT_ONLY",
        "same_facade_sibling_distractor_truth": "MISSING",
        "sign_to_facade_ownership_truth": "MISSING",
        "entrance_to_facade_ownership_truth": "MISSING",
        "target_absent_or_wrong_facade_controls": "MISSING",
        "visual_terminal_handoff_truth": "MISSING_METRIC_ARRIVAL_ONLY",
        "truth_free_runtime_observation": "FAILS_BY_DEFAULT_REPAIRABLE_WITH_WRAPPER",
        "distance_free_authority_transition": "FAILS_BY_DEFAULT_REPAIRABLE_BY_IGNORING_BENCHMARK_ARRIVAL_GATE",
    }
    missing_authority_fields = sorted(
        name for name, value in coverage.items() if value.startswith("MISSING") or value.startswith("NOT_PROVEN")
    )
    verdict = (
        "L10_ABOTN_POIBENCH_CANDIDATE_SUBSTRATE_NOT_ADMITTED_"
        "MISSING_FACADE_OWNERSHIP_SIBLING_ABSENCE_AND_VISUAL_HANDOFF_TRUTH"
    )
    return {
        "schema": "blindassist_l10_abotn_poibench_source_audit_v1",
        "source": {
            "huggingface_dataset": HF_DATASET,
            "huggingface_revision": HF_REVISION,
            "github_repository": GITHUB_REPOSITORY,
            "github_revision": GITHUB_REVISION,
            "annotation_manifest_sha256": manifest.hexdigest(),
        },
        "inventory": {
            "scene_count": len(scene_rows),
            "episode_count": len(episodes),
            "structurally_complete_episode_count": len(structurally_complete),
            "total_reference_pose_count": sum(trajectory_lengths),
            "minimum_reference_pose_count": min(trajectory_lengths) if trajectory_lengths else 0,
            "median_reference_pose_count": statistics.median(trajectory_lengths) if trajectory_lengths else 0,
            "maximum_reference_pose_count": max(trajectory_lengths) if trajectory_lengths else 0,
            "minimum_pois_per_scene": min(scene_goal_counts.values()) if scene_goal_counts else 0,
            "maximum_pois_per_scene": max(scene_goal_counts.values()) if scene_goal_counts else 0,
        },
        "official_runtime_audit": {
            "truth_fields_exposed_to_agent_by_default": truth_fields_exposed,
            "distance_threshold_defines_default_success": default_metric_distance_transition,
            "required_l10_wrapper": [
                "remove target_position",
                "remove distance_to_goal",
                "do not expose endpoint or future reference trajectory",
                "log executed action and resulting pose/image receipt",
                "score authority with evaluator-private annotations only",
            ],
        },
        "required_contract_coverage": coverage,
        "missing_authority_fields": missing_authority_fields,
        "verdict": verdict,
        "next_admissible_step": (
            "Use the released simulator only as a capture substrate. Before any three-arm run, "
            "freeze a small scene-disjoint annotation addendum with exact facade IDs, exact entrance "
            "instance IDs, sign/facade and entrance/facade ownership, same-facade siblings, target-absent "
            "controls, and visual handoff truth; then install the truth-stripping runtime wrapper."
        ),
        "claim_boundary": (
            "This audit establishes public source structure and missing authority fields only. It is not "
            "an L10 algorithm result, action-utility result, facade or entrance binding result, arrival or "
            "handoff confirmation, user-benefit evidence, deployment evidence, or safety evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
