"""Freeze a deterministic same-scene ABotN cohort before pixels or provider output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


SCHEMA = "blindassist_abotn_official_cohort_freeze_v0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_id(path: Path) -> str:
    return f"abotn-{path.parent.name}-{path.stem.replace('_', '-')}"


def select_tasks(
    annotations_root: Path,
    *,
    scene_id: str,
    count: int,
    excluded_episode_ids: Iterable[str],
) -> list[Path]:
    excluded = set(excluded_episode_ids)
    candidates = sorted(
        (
            path
            for path in (annotations_root / scene_id).glob("traj_*.json")
            if _episode_id(path) not in excluded
        ),
        key=lambda path: path.relative_to(annotations_root).as_posix(),
    )
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} unused tasks are available; need {count}")
    return candidates[:count]


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    annotations_root = args.annotations_root.resolve()
    provider_lock_path = args.provider_lock.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("cohort freeze already exists")
    repository_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if repository_commit != args.expected_repository_commit:
        raise ValueError("repository commit drift")
    provider_lock = json.loads(provider_lock_path.read_text(encoding="utf-8"))
    if provider_lock.get("schema_version") != "blindassist_last_10m_p0_provider_lock_v1":
        raise ValueError("provider lock schema mismatch")
    selected = select_tasks(
        annotations_root,
        scene_id=args.scene_id,
        count=args.episode_count,
        excluded_episode_ids=args.excluded_episode_id,
    )
    tasks = []
    render_calls = 0
    for index, path in enumerate(selected):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        trajectory = annotation.get("trajectory")
        extension = annotation.get("label", {}).get("extend", {})
        goal_name = str(extension.get("goal_label") or "").strip()
        if not isinstance(trajectory, list) or len(trajectory) < 2 or not goal_name:
            raise ValueError(f"invalid task annotation: {path}")
        node_count = len(trajectory) * 5
        render_calls += node_count
        tasks.append({
            "cohort_index": index,
            "episode_id": _episode_id(path),
            "annotation_relative_path": path.relative_to(annotations_root).as_posix(),
            "annotation_bytes": path.stat().st_size,
            "annotation_sha256": _sha256(path),
            "goal_name": goal_name,
            "pose_count": len(trajectory),
            "expected_action_node_count": node_count,
        })
    helpers = {}
    for helper in args.helper:
        resolved = helper.resolve()
        helpers[resolved.name] = {"path": str(resolved), "sha256": _sha256(resolved)}
    result = {
        "schema_version": SCHEMA,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": "ABOTN_OFFICIAL_FRESH_COHORT_FROZEN",
        "selection": {
            "rule": "LEXICOGRAPHIC_FIRST_N_UNUSED_TASKS_WITHIN_FIXED_SCENE",
            "scene_id": args.scene_id,
            "episode_count": args.episode_count,
            "excluded_episode_ids": sorted(args.excluded_episode_id),
            "pixel_or_provider_outcome_observed_before_selection": False,
            "tasks": tasks,
        },
        "repository": {"commit": repository_commit},
        "dataset": {"id": "acvlab/ABotN-POIBench", "revision": args.dataset_revision},
        "provider": {
            "lock_path": str(provider_lock_path),
            "lock_sha256": _sha256(provider_lock_path),
            "identity": provider_lock,
            "private_truth_access": False,
        },
        "helpers": helpers,
        "frozen_budget": {
            "episodes": args.episode_count,
            "official_render_calls": render_calls,
            "official_render_retries": 0,
            "provider_observations_maximum": args.episode_count * 15,
            "provider_schema_attempts_per_observation_maximum": 2,
            "teacher_calls": 0,
            "sealed_episode_reruns": 0,
        },
        "truth_and_claim_boundary": {
            "metric_endpoint_and_trajectory_authority": "NATIVE_GT",
            "functional_entrance_region_truth": "NOT_EVALUABLE_NOT_RELEASED",
            "selection_accuracy": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
            "claim_ceiling": "OFFICIAL_RENDERER_SINGLE_SCENE_EIGHT_TASK_ENGINEERING_COHORT_ONLY",
        },
        "invariants": [
            "All task identities are frozen before any cohort pixel or provider output.",
            "Provider sees only the public goal/action graph and current official RGB frame.",
            "Evaluator-private endpoint and distance open only after each provider run is terminal.",
            "No task replacement, rerun, prompt, threshold, provider, goal, teacher, or action-contract change.",
            "Missing functional pixel truth remains NOT_EVALUABLE and cannot establish LOST.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-repository-commit", required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--episode-count", type=int, default=8)
    parser.add_argument("--excluded-episode-id", action="append", default=[])
    parser.add_argument("--provider-lock", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--helper", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args)
    print(json.dumps({
        "terminal": result["terminal"],
        "episodes": result["frozen_budget"]["episodes"],
        "official_render_calls": result["frozen_budget"]["official_render_calls"],
        "provider_observations_maximum": result["frozen_budget"]["provider_observations_maximum"],
        "episode_ids": [row["episode_id"] for row in result["selection"]["tasks"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
