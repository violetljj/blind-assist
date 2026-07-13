#!/usr/bin/env python3
"""Plan a deterministic, session-atomic SANPO P3 train/dev split.

The planner consumes only official SANPO train candidates.  It reads native,
original-resolution masks, maps them to BlindAssist's four semantic classes,
and searches complete per-scene train/dev combinations.  It never reads an
official test/blind manifest and writes nothing unless a feasible plan exists.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SCENES = (
    "parallel_boundary",
    "step_curb",
    "center_obstacle",
    "lateral_pedestrian_or_ebike",
)
CLASS_NAMES = (
    "walkable",
    "boundary_step_curb",
    "obstacle",
    "unknown_nonwalkable",
)
SANPO_MAP = {
    0: 3, 1: 0, 2: 1, 3: 0, 4: 2, 5: 0, 6: 0, 7: 3,
    8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 1,
    16: 2, 17: 0, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2,
    24: 2, 25: 2, 26: 2, 27: 3, 28: 2, 29: 3, 30: 3,
}
MIN_TRAIN, MAX_TRAIN = 4, 6
MIN_DEV, MAX_DEV = 2, 3
DEFAULT_MAX_GLOBAL_COMBINATIONS = 2_000_000
MAX_CLASS_SHARE_RATIO = 2.0
MAX_DEV_BOUNDARY_CONTRIBUTION = 0.50
MAX_OTHER_CLASS_CONTRIBUTION = 0.60
MIN_CONTRIBUTING_SESSIONS = 2
MIN_DEV_BOUNDARY_CONTRIBUTING_SESSIONS = 3


class PlanningError(ValueError):
    """A fail-closed planning error."""


@dataclass(frozen=True)
class SessionStats:
    source_id: str
    native_session_id: str
    global_session_id: str
    scene_bucket: str
    package_root: str
    manifest_path: str
    manifest_sha256: str
    frame_count: int
    pixel_counts: tuple[int, int, int, int]
    raw_mask_sha256: tuple[str, ...]
    raw_mask_set_sha256: str
    sequence: dict[str, Any]


@dataclass(frozen=True)
class SceneAssignment:
    train: tuple[SessionStats, ...]
    dev: tuple[SessionStats, ...]
    reserve: tuple[SessionStats, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise PlanningError(f"path escapes package root: {relative}") from error
    if not path.is_file():
        raise PlanningError(f"missing input file: {path}")
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise PlanningError(f"cannot load manifest {path}: {error}") from error


def row_session_id(row: dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    return str(source.get("session_id") or row.get("session_id") or "").strip()


def native_mask_counts(path: Path) -> tuple[tuple[int, int, int, int], str]:
    digest = sha256_file(path)
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., 0]
    native_counts = np.bincount(array.reshape(-1), minlength=256)
    unknown = [int(value) for value in np.flatnonzero(native_counts) if int(value) not in SANPO_MAP]
    if unknown:
        raise PlanningError(f"native mask {path} contains unmapped class IDs {unknown}")
    mapped = [0, 0, 0, 0]
    for native_id, target_id in SANPO_MAP.items():
        mapped[target_id] += int(native_counts[native_id])
    return tuple(mapped), digest


def _preflight_sequences(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    sequences = recipe.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise PlanningError("candidate recipe requires a non-empty sequences list")
    # This pass intentionally happens before any manifest path is opened.
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, dict):
            raise PlanningError(f"sequence {index} must be an object")
        if str(sequence.get("official_split", "")).strip() != "train":
            raise PlanningError(
                f"sequence {index} is not official train; official test/blind candidates are forbidden"
            )
        if str(sequence.get("scene_bucket", "")) not in SCENES:
            raise PlanningError(f"sequence {index} has unsupported scene_bucket")
    return sequences


def collect_sessions(recipe_path: Path) -> tuple[list[SessionStats], dict[str, Any]]:
    recipe_path = recipe_path.resolve()
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanningError(f"cannot load candidate recipe: {error}") from error
    sequences = _preflight_sequences(recipe)
    sessions: list[SessionStats] = []
    seen: set[str] = set()
    manifest_hashes: dict[str, str] = {}
    for sequence in sequences:
        package_root = Path(str(sequence.get("package_root", ""))).resolve()
        manifest = safe_file(package_root, str(sequence.get("manifest_path", "")))
        manifest_digest = sha256_file(manifest)
        manifest_hashes[str(manifest)] = manifest_digest
        native_session = str(sequence.get("native_session_id", "")).strip()
        source_id = str(sequence.get("source_id", "sanpo_real_v0")).strip()
        if not native_session or not source_id:
            raise PlanningError("every sequence requires source_id and native_session_id")
        global_session = f"{source_id}:{native_session}"
        if global_session in seen:
            raise PlanningError(f"duplicate candidate session: {global_session}")
        seen.add(global_session)
        rows = [row for row in load_jsonl(manifest) if row_session_id(row) == native_session]
        rows.sort(key=lambda row: int(row.get("frame_index", -1)))
        expected_count = int(sequence.get("expected_frame_count", len(rows)))
        indexes = [int(row.get("frame_index", -1)) for row in rows]
        if expected_count <= 0 or len(rows) != expected_count or indexes != list(range(expected_count)):
            raise PlanningError(
                f"{global_session}: requires contiguous 0..{expected_count - 1} frames, got {indexes[:8]}"
            )
        row_official_splits = {
            str(
                (row.get("source") if isinstance(row.get("source"), dict) else {}).get(
                    "official_split", row.get("official_split", "")
                )
            ).strip()
            for row in rows
        }
        if row_official_splits != {"train"}:
            raise PlanningError(
                f"{global_session}: source manifest is not exclusively official train: "
                f"{sorted(row_official_splits)}"
            )
        counts = np.zeros(4, dtype=np.int64)
        mask_hashes: list[str] = []
        for row in rows:
            sample_id = str(row.get("id", "")).strip()
            mask_value = str(
                row.get("source_mask_path")
                or (f"source_masks/test/{sample_id}.png" if sample_id else "")
            ).strip()
            if not mask_value:
                raise PlanningError(f"{global_session}: cannot resolve the raw source mask path")
            mask = safe_file(package_root, mask_value)
            mask_counts, mask_digest = native_mask_counts(mask)
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            provenance = row.get("label_provenance") if isinstance(row.get("label_provenance"), dict) else {}
            declared = str(
                row.get("source_mask_sha256")
                or provenance.get("source_mask_sha256")
                or source.get("mask_sha256")
                or ""
            ).strip()
            if declared and declared != mask_digest:
                raise PlanningError(f"{global_session}: declared raw mask SHA256 mismatch")
            counts += np.asarray(mask_counts, dtype=np.int64)
            mask_hashes.append(mask_digest)
        sessions.append(SessionStats(
            source_id=source_id,
            native_session_id=native_session,
            global_session_id=global_session,
            scene_bucket=str(sequence["scene_bucket"]),
            package_root=str(package_root),
            manifest_path=str(manifest),
            manifest_sha256=manifest_digest,
            frame_count=len(rows),
            pixel_counts=tuple(int(value) for value in counts),
            raw_mask_sha256=tuple(mask_hashes),
            raw_mask_set_sha256=canonical_sha256(mask_hashes),
            sequence=dict(sequence),
        ))
    input_hashes = {
        "candidate_recipe_sha256": sha256_file(recipe_path),
        "manifest_sha256": dict(sorted(manifest_hashes.items())),
        "all_raw_mask_sha256": canonical_sha256(sorted(
            digest for session in sessions for digest in session.raw_mask_sha256
        )),
    }
    return sorted(sessions, key=lambda item: item.global_session_id), input_hashes


def scene_assignments(sessions: list[SessionStats]) -> list[SceneAssignment]:
    result: list[SceneAssignment] = []
    ordered = tuple(sorted(sessions, key=lambda item: item.global_session_id))
    for dev_count in range(MIN_DEV, min(MAX_DEV, len(ordered)) + 1):
        for dev in itertools.combinations(ordered, dev_count):
            dev_ids = {item.global_session_id for item in dev}
            remaining = tuple(item for item in ordered if item.global_session_id not in dev_ids)
            for train_count in range(MIN_TRAIN, min(MAX_TRAIN, len(remaining)) + 1):
                for train in itertools.combinations(remaining, train_count):
                    train_ids = {item.global_session_id for item in train}
                    reserve = tuple(item for item in remaining if item.global_session_id not in train_ids)
                    train_hashes = {value for item in train for value in item.raw_mask_sha256}
                    dev_hashes = {value for item in dev for value in item.raw_mask_sha256}
                    if train_hashes & dev_hashes:
                        continue
                    result.append(SceneAssignment(train=tuple(train), dev=tuple(dev), reserve=reserve))
    return result


def _sum_counts(sessions: Iterable[SessionStats]) -> np.ndarray:
    values = [item.pixel_counts for item in sessions]
    return np.asarray(values, dtype=np.float64).sum(axis=0) if values else np.zeros(4, dtype=np.float64)


def _shares(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    return counts / total if total else np.zeros_like(counts)


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2.0
    def kl(first: np.ndarray, second: np.ndarray) -> float:
        keep = first > 0
        return float(np.sum(first[keep] * np.log2(first[keep] / second[keep])))
    return (kl(left, midpoint) + kl(right, midpoint)) / 2.0


def split_metrics(sessions: tuple[SessionStats, ...]) -> dict[str, Any]:
    counts = _sum_counts(sessions)
    shares = _shares(counts)
    concentration: dict[str, float] = {}
    effective: dict[str, float] = {}
    contributing: dict[str, int] = {}
    for class_index, name in enumerate(CLASS_NAMES):
        contributions = np.asarray([item.pixel_counts[class_index] for item in sessions], dtype=np.float64)
        total = float(contributions.sum())
        concentration[name] = float(contributions.max() / total) if total else 1.0
        effective[name] = float(total * total / np.square(contributions).sum()) if total and np.square(contributions).sum() else 0.0
        contributing[name] = int(np.count_nonzero(contributions))
    return {
        "session_count": len(sessions),
        "frame_count": sum(item.frame_count for item in sessions),
        "total_pixels": int(counts.sum()),
        "class_pixels": {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)},
        "class_shares": {name: float(shares[index]) for index, name in enumerate(CLASS_NAMES)},
        "max_session_contribution": concentration,
        "effective_session_count": effective,
        "contributing_session_count": contributing,
    }


def distribution_gate(
    train_metrics: dict[str, Any], dev_metrics: dict[str, Any],
) -> tuple[dict[str, float | None], dict[str, float | None], list[str]]:
    """Apply pre-registered P3 distribution gates to one complete assignment."""
    ratios: dict[str, float | None] = {}
    log2_gaps: dict[str, float | None] = {}
    failures: list[str] = []
    for name in CLASS_NAMES:
        train_share = float(train_metrics["class_shares"][name])
        dev_share = float(dev_metrics["class_shares"][name])
        if train_share <= 0 or dev_share <= 0:
            ratios[name] = None
            log2_gaps[name] = None
            failures.append(f"{name}: class must have pixels in both train and dev")
        else:
            ratio = max(train_share / dev_share, dev_share / train_share)
            ratios[name] = ratio
            log2_gaps[name] = abs(math.log2(train_share / dev_share))
            if ratio > MAX_CLASS_SHARE_RATIO + 1e-12:
                failures.append(
                    f"{name}: train/dev class-share ratio {ratio:.6f} exceeds {MAX_CLASS_SHARE_RATIO:.1f}"
                )
        for split_name, metrics in (("train", train_metrics), ("dev", dev_metrics)):
            minimum = (
                MIN_DEV_BOUNDARY_CONTRIBUTING_SESSIONS
                if split_name == "dev" and name == "boundary_step_curb"
                else MIN_CONTRIBUTING_SESSIONS
            )
            actual = int(metrics["contributing_session_count"][name])
            if actual < minimum:
                failures.append(
                    f"{split_name}/{name}: contributing session count {actual} is below {minimum}"
                )
            limit = (
                MAX_DEV_BOUNDARY_CONTRIBUTION
                if split_name == "dev" and name == "boundary_step_curb"
                else MAX_OTHER_CLASS_CONTRIBUTION
            )
            concentration = float(metrics["max_session_contribution"][name])
            if concentration > limit + 1e-12:
                failures.append(
                    f"{split_name}/{name}: max session contribution {concentration:.6f} exceeds {limit:.2f}"
                )
    return ratios, log2_gaps, failures


def _objective(assignments: tuple[SceneAssignment, ...], pool_shares: np.ndarray) -> tuple[Any, ...]:
    train = tuple(item for assignment in assignments for item in assignment.train)
    dev = tuple(item for assignment in assignments for item in assignment.dev)
    train_shares, dev_shares = _shares(_sum_counts(train)), _shares(_sum_counts(dev))
    if np.any(train_shares == 0) or np.any(dev_shares == 0):
        distribution_gap = math.inf
    else:
        distribution_gap = float(np.max(np.abs(np.log2(train_shares / dev_shares))))
    train_metrics, dev_metrics = split_metrics(train), split_metrics(dev)
    concentration = max(
        list(train_metrics["max_session_contribution"].values())
        + list(dev_metrics["max_session_contribution"].values())
    )
    selected = len(train) + len(dev)
    assignment_key = {
        "train": sorted(item.global_session_id for item in train),
        "dev": sorted(item.global_session_id for item in dev),
    }
    return (
        -selected,
        distribution_gap,
        _js_divergence(train_shares, pool_shares) + _js_divergence(dev_shares, pool_shares),
        concentration,
        canonical_sha256(assignment_key),
    )


def plan_split(
    recipe_path: Path, max_global_combinations: int = DEFAULT_MAX_GLOBAL_COMBINATIONS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sessions, input_hashes = collect_sessions(recipe_path)
    candidate_recipe = json.loads(recipe_path.resolve().read_text(encoding="utf-8"))
    by_scene = {scene: [item for item in sessions if item.scene_bucket == scene] for scene in SCENES}
    insufficient = {scene: len(items) for scene, items in by_scene.items() if len(items) < MIN_TRAIN + MIN_DEV}
    if insufficient:
        raise PlanningError(f"insufficient candidate sessions by scene: {insufficient}")
    options: list[list[SceneAssignment]] = []
    for scene in SCENES:
        feasible = scene_assignments(by_scene[scene])
        if not feasible:
            raise PlanningError(f"scene {scene} has no leakage-free train/dev assignment")
        options.append(feasible)
    search_space = math.prod(len(items) for items in options)
    if search_space > max_global_combinations:
        raise PlanningError(
            f"exact search space {search_space} exceeds fail-closed limit {max_global_combinations}"
        )
    pool_shares = _shares(_sum_counts(sessions))
    best: tuple[SceneAssignment, ...] | None = None
    best_objective: tuple[Any, ...] | None = None
    for candidate in itertools.product(*options):
        train_hashes = {value for assignment in candidate for item in assignment.train for value in item.raw_mask_sha256}
        dev_hashes = {value for assignment in candidate for item in assignment.dev for value in item.raw_mask_sha256}
        if train_hashes & dev_hashes:
            continue
        candidate_train = tuple(item for assignment in candidate for item in assignment.train)
        candidate_dev = tuple(item for assignment in candidate for item in assignment.dev)
        _, _, gate_failures = distribution_gate(
            split_metrics(candidate_train), split_metrics(candidate_dev),
        )
        if gate_failures:
            continue
        objective = _objective(candidate, pool_shares)
        if best_objective is None or objective < best_objective:
            best, best_objective = candidate, objective
    if best is None or best_objective is None:
        raise PlanningError("no global assignment satisfies leakage and P3 distribution gates")
    train = tuple(item for assignment in best for item in assignment.train)
    dev = tuple(item for assignment in best for item in assignment.dev)
    reserve = tuple(item for assignment in best for item in assignment.reserve)
    train_metrics, dev_metrics = split_metrics(train), split_metrics(dev)
    ratios, log2_gaps, gate_failures = distribution_gate(train_metrics, dev_metrics)
    if gate_failures:  # Defensive assertion: search admits only gate-green candidates.
        raise PlanningError("selected assignment unexpectedly failed P3 distribution gates")
    split_by_id = {
        **{item.global_session_id: "train" for item in train},
        **{item.global_session_id: "dev" for item in dev},
        **{item.global_session_id: "reserve" for item in reserve},
    }
    planned_sequences = []
    for session in sorted(sessions, key=lambda item: item.global_session_id):
        if split_by_id[session.global_session_id] == "reserve":
            continue
        sequence = dict(session.sequence)
        sequence["split"] = split_by_id[session.global_session_id]
        planned_sequences.append(sequence)
    selected_frame_counts = {item.frame_count for item in train + dev}
    if len(selected_frame_counts) != 1:
        raise PlanningError(
            "selected train/dev sessions must have one fixed frame count for the assembly coverage policy"
        )
    coverage_policy = deepcopy(
        candidate_recipe.get("coverage_policy")
        if isinstance(candidate_recipe.get("coverage_policy"), dict)
        else {}
    )
    coverage_policy.update({
        "format": "blindassist_sanpo_v4_coverage_policy_v1",
        "sequence_frame_count": next(iter(selected_frame_counts)),
        "min_train_sessions": len(SCENES) * MIN_TRAIN,
        "min_dev_sessions": len(SCENES) * MIN_DEV,
        "required_scene_sessions": {
            scene: {"train": MIN_TRAIN, "dev": MIN_DEV, "total": MIN_TRAIN + MIN_DEV}
            for scene in SCENES
        },
        "official_split_by_target_split": {"train": "train", "dev": "train", "blind": "test"},
    })
    plan = deepcopy(candidate_recipe)
    plan["split_plan_schema"] = "blindassist_sanpo_p3_session_split_plan_v1"
    plan["coverage_policy"] = coverage_policy
    plan["sequences"] = planned_sequences
    plan["assignment_sha256"] = canonical_sha256(split_by_id)
    inventory = []
    for session in sorted(sessions, key=lambda item: item.global_session_id):
        counts = np.asarray(session.pixel_counts, dtype=np.float64)
        shares = _shares(counts)
        inventory.append({
            "global_session_id": session.global_session_id,
            "scene_bucket": session.scene_bucket,
            "assignment": split_by_id[session.global_session_id],
            "frame_count": session.frame_count,
            "total_pixels": int(counts.sum()),
            "class_pixels": {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)},
            "class_shares": {name: float(shares[index]) for index, name in enumerate(CLASS_NAMES)},
            "manifest_sha256": session.manifest_sha256,
            "raw_mask_set_sha256": session.raw_mask_set_sha256,
            "raw_mask_sha256": list(session.raw_mask_sha256),
        })
    report = {
        "schema": "blindassist_sanpo_p3_session_split_report_v1",
        "status": "green",
        "blind_access": "not_accessed",
        "official_split_consumed": "train_only",
        "input_hashes": input_hashes,
        "constraints": {
            "train_sessions_per_scene": [MIN_TRAIN, MAX_TRAIN],
            "dev_sessions_per_scene": [MIN_DEV, MAX_DEV],
            "session_atomic": True,
            "raw_mask_sha_cross_split_forbidden": True,
            "max_train_dev_class_share_ratio": MAX_CLASS_SHARE_RATIO,
            "max_dev_boundary_session_contribution": MAX_DEV_BOUNDARY_CONTRIBUTION,
            "max_other_session_class_contribution": MAX_OTHER_CLASS_CONTRIBUTION,
            "min_contributing_sessions_per_split_class": MIN_CONTRIBUTING_SESSIONS,
            "min_dev_boundary_contributing_sessions": MIN_DEV_BOUNDARY_CONTRIBUTING_SESSIONS,
        },
        "search": {
            "method": "deterministic_exact_cartesian_combination_search",
            "search_space": search_space,
            "objective": {
                "selected_session_count": -int(best_objective[0]),
                "max_abs_log2_class_share_ratio": best_objective[1],
                "train_plus_dev_js_to_candidate_pool": best_objective[2],
                "max_session_class_contribution": best_objective[3],
                "tie_break_sha256": best_objective[4],
            },
        },
        "scene_session_counts": {
            scene: {
                "candidate": len(by_scene[scene]),
                "train": len(best[index].train),
                "dev": len(best[index].dev),
                "reserve": len(best[index].reserve),
            }
            for index, scene in enumerate(SCENES)
        },
        "splits": {"train": train_metrics, "dev": dev_metrics},
        "train_dev_class_share_ratio": ratios,
        "train_dev_abs_log2_class_share_gap": log2_gaps,
        "distribution_gate": {"status": "green", "failures": []},
        "session_inventory": inventory,
        "plan_sha256": canonical_sha256(plan),
    }
    return plan, report


def write_success(plan: dict[str, Any], report: dict[str, Any], output_plan: Path, output_report: Path) -> None:
    output_plan, output_report = output_plan.resolve(), output_report.resolve()
    if output_plan.exists() or output_report.exists():
        raise PlanningError("refusing to overwrite an existing plan or report")
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-recipe", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-global-combinations", type=int, default=DEFAULT_MAX_GLOBAL_COMBINATIONS)
    args = parser.parse_args()
    try:
        plan, report = plan_split(args.candidate_recipe, args.max_global_combinations)
        write_success(plan, report, args.output_plan, args.report)
    except (PlanningError, OSError, ValueError) as error:
        print(f"p3_split_plan_ok=false error={error}", file=sys.stderr)
        return 1
    print(
        f"p3_split_plan_ok=true assignment_sha256={plan['assignment_sha256']} "
        f"report={args.report.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
