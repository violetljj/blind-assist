"""Freeze one deterministic fresh ABotN official-pixel V0 episode."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


SCHEMA = "blindassist_abotn_official_prospective_freeze_v0"
EPISODE_PREFIX = "abotn-"
HANDOFF_DISTANCE_LIMIT_M = 3.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_id(path: Path) -> str:
    return f"{EPISODE_PREFIX}{path.parent.name}-{path.stem.replace('_', '-')}"


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repository.resolve()
    annotations_root = args.annotations_root.resolve()
    annotation = args.annotation.resolve()
    graph_dir = args.action_graph_dir.resolve()
    provider_lock_path = args.provider_lock.resolve()
    polygon = args.polygon.resolve()
    excluded = set(args.excluded_episode_id)
    candidates = sorted(
        (
            path
            for path in annotations_root.glob("*/traj_*.json")
            if _episode_id(path) not in excluded
        ),
        key=lambda path: path.relative_to(annotations_root).as_posix(),
    )
    if not candidates or annotation != candidates[0].resolve():
        raise ValueError("annotation is not the lexicographic first non-excluded task")
    episode_id = _episode_id(annotation)
    public_path = graph_dir / "public-graph.json"
    private_path = graph_dir / "evaluator-private.json"
    graph_freeze_path = graph_dir / "freeze-receipt.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    graph_freeze = json.loads(graph_freeze_path.read_text(encoding="utf-8"))
    provider_lock = json.loads(provider_lock_path.read_text(encoding="utf-8"))
    if public.get("episode_id") != episode_id or public.get("private_truth_access") is not False:
        raise ValueError("public graph episode or firewall drift")
    if graph_freeze.get("terminal") != "ABOTN_V0_ACTION_GRAPH_FROZEN_ELIGIBLE":
        raise ValueError("action graph is not eligible")
    if graph_freeze["inputs"]["annotation_sha256"] != _sha256(annotation):
        raise ValueError("action graph annotation binding drift")
    if provider_lock.get("schema_version") != "blindassist_last_10m_p0_provider_lock_v1":
        raise ValueError("provider lock schema mismatch")
    repo_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if repo_head != args.expected_repository_commit:
        raise ValueError("repository commit drift")
    if polygon.stat().st_size != args.expected_polygon_bytes:
        raise ValueError("polygon byte-size drift")
    if _sha256(polygon) != args.expected_polygon_sha256:
        raise ValueError("polygon SHA-256 drift")
    render_helper = args.render_helper.resolve()
    download_helper = args.download_helper.resolve()
    result = {
        "schema_version": SCHEMA,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": "ABOTN_OFFICIAL_PROSPECTIVE_EPISODE_FROZEN",
        "selection": {
            "rule": "LEXICOGRAPHIC_FIRST_ANNOTATION_EXCLUDING_PREVIOUSLY_USED_EPISODES",
            "episode_id": episode_id,
            "annotation_relative_path": annotation.relative_to(annotations_root).as_posix(),
            "excluded_episode_ids": sorted(excluded),
            "pixel_or_provider_outcome_observed_before_selection": False,
        },
        "repository": {
            "commit": repo_head,
        },
        "dataset": {
            "id": "acvlab/ABotN-POIBench",
            "revision": args.dataset_revision,
            "annotation": {
                "path": str(annotation),
                "bytes": annotation.stat().st_size,
                "sha256": _sha256(annotation),
            },
            "scene_id": annotation.parent.name,
            "point_cloud": {
                "url": args.point_cloud_url,
                "bytes": args.expected_point_cloud_bytes,
                "sha256": args.expected_point_cloud_sha256,
            },
            "polygon": {
                "path": str(polygon),
                "bytes": polygon.stat().st_size,
                "sha256": _sha256(polygon),
            },
        },
        "official_renderer": {
            "repository": "amap-cvlab/ABot-Navigation",
            "commit": args.official_renderer_commit,
            "render_scale": 1.5,
            "camera": {
                "width": 720,
                "height": 640,
                "fx": 252.075,
                "fy": 252.075,
                "cx": 360.0,
                "cy": 320.0,
                "extrinsic_height": 0.65,
                "views": ["front"],
            },
            "render_helper_path": str(render_helper),
            "render_helper_sha256": _sha256(render_helper),
            "download_helper_path": str(download_helper),
            "download_helper_sha256": _sha256(download_helper),
        },
        "action_graph": {
            "public_graph_path": str(public_path),
            "public_graph_sha256": _sha256(public_path),
            "private_truth_path": str(private_path),
            "private_truth_sha256": _sha256(private_path),
            "freeze_receipt_path": str(graph_freeze_path),
            "freeze_receipt_sha256": _sha256(graph_freeze_path),
            "pose_count": graph_freeze["pose_count"],
            "node_count": graph_freeze["node_count"],
            "shortest_start_to_arrival_steps": graph_freeze["shortest_start_to_arrival_steps"],
        },
        "provider": {
            "lock_path": str(provider_lock_path),
            "lock_sha256": _sha256(provider_lock_path),
            "identity": provider_lock,
            "private_truth_access": False,
        },
        "frozen_budget": {
            "episodes": 1,
            "official_render_calls": graph_freeze["node_count"],
            "official_render_retries": 0,
            "provider_observations_maximum": 15,
            "provider_schema_attempts_per_observation_maximum": 2,
            "teacher_calls": 0,
            "sealed_episode_reruns": 0,
        },
        "termination_contract": (
            {
                "mode": "HANDOFF_V1",
                "current_frame_cue": "CENTERED_CANDIDATE_HEIGHT_GTE_0_55",
                "controller_effect": "STOP_AUTOMATIC_MOTION_AND_EMIT_HANDOFF_READY",
                "handoff_distance_limit_m": HANDOFF_DISTANCE_LIMIT_M,
                "completion_authority": ["USER_EXPLICIT", "TRUSTED_INTERACTION"],
                "forbidden_controller_outputs": ["ARRIVED", "COMPLETE", "COMPLETED_BY_USER"],
                "success_rule": "HANDOFF_READY_AND_NATIVE_DISTANCE_TO_GOAL_LTE_3M",
                "claim_boundary": "HANDOFF_READY_IS_NOT_ARRIVED_OR_COMPLETED",
            }
            if args.termination_mode == "HANDOFF_V1"
            else {
                "mode": "V0_COMPLETION",
                "success_rule": "LEGACY_FROZEN_V0_CONTROL_CONTRACT",
            }
        ),
        "truth_and_claim_boundary": {
            "metric_endpoint_and_trajectory_authority": "NATIVE_GT",
            "functional_entrance_region_truth": "NOT_EVALUABLE_NOT_RELEASED",
            "selection_accuracy": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
            "evaluated_outcomes": [
                "metric goal progress",
                "metric arrival",
                "false arrival",
                "instruction and abstention behavior",
            ],
            "claim_ceiling": "OFFICIAL_RENDERER_SINGLE_PUBLIC_TASK_ENGINEERING_ONLY",
        },
        "invariants": [
            "Provider sees only the public graph goal and current official RGB frame.",
            "Evaluator-private endpoint and distance open only after all provider calls are terminal.",
            "No prompt, threshold, provider, goal, action contract, or teacher change after freeze.",
            "UNKNOWN and NOT_EVALUABLE denominators are not rescued or replaced.",
        ],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("prospective freeze already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-repository-commit", required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--excluded-episode-id", action="append", default=[])
    parser.add_argument("--action-graph-dir", type=Path, required=True)
    parser.add_argument("--provider-lock", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--point-cloud-url", required=True)
    parser.add_argument("--expected-point-cloud-bytes", type=int, required=True)
    parser.add_argument("--expected-point-cloud-sha256", required=True)
    parser.add_argument("--polygon", type=Path, required=True)
    parser.add_argument("--expected-polygon-bytes", type=int, required=True)
    parser.add_argument("--expected-polygon-sha256", required=True)
    parser.add_argument("--official-renderer-commit", required=True)
    parser.add_argument(
        "--termination-mode",
        choices=("V0_COMPLETION", "HANDOFF_V1"),
        default="V0_COMPLETION",
    )
    parser.add_argument("--render-helper", type=Path, required=True)
    parser.add_argument("--download-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args)
    print(json.dumps({
        "terminal": result["terminal"],
        "episode_id": result["selection"]["episode_id"],
        "render_calls": result["frozen_budget"]["official_render_calls"],
        "provider_observations_maximum": result["frozen_budget"]["provider_observations_maximum"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
