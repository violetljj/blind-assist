from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import artvideo_dual_state_replay as sc1w
import artvideo_ocr_replay as replay
import artvideo_opportunity_active_search as sc2


SEMANTIC_GATE = 0.58
HORIZON_TRACK_FRAMES = 3
MIN_OPPORTUNITIES = 12
MIN_PER_ALIGNMENT_ARM = 4
MIN_MEAN_GAIN_DELTA = 0.10
MIN_IMPROVEMENT_RATE_DELTA = 0.20
MAX_WRONG_GATE_RATE_PENALTY = 0.05

CENTERING_ACTIONS = {
    "PAN_LEFT_TO_TEXT",
    "PAN_RIGHT_TO_TEXT",
    "SCAN_LAST_LEFT",
    "SCAN_LAST_FORWARD",
    "SCAN_LAST_RIGHT",
}
APPROACH_ACTIONS = {"APPROACH_FOR_TEXT"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_geometry(
    video: dict[str, Any],
    frame_id: int,
    target_id: int,
) -> tuple[float, float]:
    row = next(
        item
        for item in video["by_frame"].get(frame_id, [])
        if int(item["obj_id"]) == target_id
    )
    width, height = video["frame_sizes"][frame_id]
    points = np.asarray(row["point"], dtype=np.float64)
    center_x = float(points[:, 0].mean()) / width
    box_width = float(points[:, 0].max() - points[:, 0].min()) / width
    box_height = float(points[:, 1].max() - points[:, 1].min()) / height
    return center_x, max(1e-9, box_width * box_height)


def semantic_scores(
    video: dict[str, Any],
    frame_id: int,
    target_id: int,
    goal: str,
) -> tuple[float, float]:
    target_scores: list[float] = []
    wrong_scores: list[float] = []
    for candidate in video["candidates"].get(frame_id, []):
        score = float(replay.lexical(goal, candidate.text))
        if target_id in candidate.truth_ids:
            target_scores.append(score)
        else:
            wrong_scores.append(score)
    return max(target_scores, default=0.0), max(wrong_scores, default=0.0)


def action_geometry_gain(
    action: str,
    current: tuple[float, float],
    future: tuple[float, float],
) -> tuple[str, float] | None:
    current_x, current_area = current
    future_x, future_area = future
    if action in CENTERING_ACTIONS:
        gain = abs(current_x - 0.5) - abs(future_x - 0.5)
        return "CENTERING", gain
    if action in APPROACH_ACTIONS:
        gain = math.log(future_area / current_area)
        return "APPROACH", gain
    return None


def run_track(
    video: dict[str, Any],
    target_id: int,
    goal: str,
    frame_ids: list[int],
) -> list[dict[str, Any]]:
    controller = sc2.L10SC2OpportunityActiveSearch(goal)
    rows: list[dict[str, Any]] = []
    for position, frame_id in enumerate(frame_ids):
        active = controller.step(video["candidates"].get(frame_id, []))
        future_position = position + HORIZON_TRACK_FRAMES
        if future_position >= len(frame_ids):
            continue
        future_frame_id = frame_ids[future_position]
        action = active.observation_action
        current_geometry = target_geometry(video, frame_id, target_id)
        future_geometry = target_geometry(video, future_frame_id, target_id)
        geometric = action_geometry_gain(action, current_geometry, future_geometry)
        if geometric is None:
            continue
        current_target, current_wrong = semantic_scores(video, frame_id, target_id, goal)
        future_target, future_wrong = semantic_scores(video, future_frame_id, target_id, goal)
        geometry_kind, geometry_gain = geometric
        rows.append(
            {
                "video": video["name"],
                "target_id": target_id,
                "goal": goal,
                "frame_id": frame_id,
                "future_frame_id": future_frame_id,
                "horizon_track_frames": HORIZON_TRACK_FRAMES,
                "action": action,
                "source_action_mode": active.source_action_mode,
                "source_semantic_state": active.source_semantic_state,
                "normalized_semantic_state": active.decision.semantic_state,
                "evidence_deficit": active.decision.evidence_deficit,
                "geometry_kind": geometry_kind,
                "geometry_gain": round(geometry_gain, 6),
                "alignment": "ALIGNED" if geometry_gain > 0.0 else (
                    "OPPOSED" if geometry_gain < 0.0 else "NEUTRAL"
                ),
                "current_target_lexical": round(current_target, 6),
                "future_target_lexical": round(future_target, 6),
                "target_semantic_gain": round(future_target - current_target, 6),
                "target_improved": future_target > current_target,
                "target_gate_crossed": current_target < SEMANTIC_GATE <= future_target,
                "current_wrong_lexical": round(current_wrong, 6),
                "future_wrong_lexical": round(future_wrong, 6),
                "wrong_gate_created": current_wrong < SEMANTIC_GATE <= future_wrong,
            }
        )
    return rows


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = [float(row["target_semantic_gain"]) for row in rows]
    return {
        "opportunities": len(rows),
        "mean_target_semantic_gain": round(statistics.fmean(gains), 4) if gains else None,
        "median_target_semantic_gain": round(statistics.median(gains), 4) if gains else None,
        "target_improved": sum(bool(row["target_improved"]) for row in rows),
        "target_improvement_rate": rate(sum(bool(row["target_improved"]) for row in rows), len(rows)),
        "target_gate_crossed": sum(bool(row["target_gate_crossed"]) for row in rows),
        "target_gate_crossing_rate": rate(sum(bool(row["target_gate_crossed"]) for row in rows), len(rows)),
        "wrong_gate_created": sum(bool(row["wrong_gate_created"]) for row in rows),
        "wrong_gate_creation_rate": rate(sum(bool(row["wrong_gate_created"]) for row in rows), len(rows)),
    }


def adjudicate(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    aligned = [row for row in rows if row["alignment"] == "ALIGNED"]
    opposed = [row for row in rows if row["alignment"] == "OPPOSED"]
    aligned_summary = arm_summary(aligned)
    opposed_summary = arm_summary(opposed)
    enough_support = (
        len(rows) >= MIN_OPPORTUNITIES
        and len(aligned) >= MIN_PER_ALIGNMENT_ARM
        and len(opposed) >= MIN_PER_ALIGNMENT_ARM
    )
    gain_delta = None
    improvement_delta = None
    wrong_penalty = None
    if aligned and opposed:
        gain_delta = round(
            float(aligned_summary["mean_target_semantic_gain"])
            - float(opposed_summary["mean_target_semantic_gain"]),
            4,
        )
        improvement_delta = round(
            float(aligned_summary["target_improvement_rate"])
            - float(opposed_summary["target_improvement_rate"]),
            4,
        )
        wrong_penalty = round(
            float(aligned_summary["wrong_gate_creation_rate"])
            - float(opposed_summary["wrong_gate_creation_rate"]),
            4,
        )
    checks = {
        "enough_total_opportunities": len(rows) >= MIN_OPPORTUNITIES,
        "enough_aligned_opportunities": len(aligned) >= MIN_PER_ALIGNMENT_ARM,
        "enough_opposed_opportunities": len(opposed) >= MIN_PER_ALIGNMENT_ARM,
        "mean_semantic_gain_delta_pass": gain_delta is not None and gain_delta >= MIN_MEAN_GAIN_DELTA,
        "improvement_rate_delta_pass": (
            improvement_delta is not None and improvement_delta >= MIN_IMPROVEMENT_RATE_DELTA
        ),
        "wrong_gate_penalty_pass": (
            wrong_penalty is not None and wrong_penalty <= MAX_WRONG_GATE_RATE_PENALTY
        ),
    }
    if not enough_support:
        decision = "SC15_NOT_EVALUABLE_INSUFFICIENT_PASSIVE_ACTION_TRANSITIONS"
    elif all(checks.values()):
        decision = "SC15_PASSIVE_ACTION_CONDITIONED_OBSERVATION_GAIN_SIGNAL"
    else:
        decision = "SC15_PASSIVE_ACTION_CONDITIONED_OBSERVATION_GAIN_GATE_NOT_MET"
    comparison = {
        "aligned": aligned_summary,
        "opposed": opposed_summary,
        "mean_semantic_gain_delta": gain_delta,
        "target_improvement_rate_delta": improvement_delta,
        "wrong_gate_creation_rate_penalty": wrong_penalty,
    }
    return decision, checks, comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit passive natural view transitions against frozen SC2 deficit actions."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=12)
    parser.add_argument("--gap-length", type=int, default=4)
    args = parser.parse_args()

    videos, _specs = sc2.load_partition(args)
    rows: list[dict[str, Any]] = []
    track_count = 0
    for video in videos:
        for target_id, goal, frame_ids in replay.eligible_tracks(video, args.minimum_frames):
            track_count += 1
            rows.extend(run_track(video, target_id, goal, frame_ids))

    decision, checks, comparison = adjudicate(rows)
    actions = Counter(row["action"] for row in rows)
    result = {
        "schema_version": 1,
        "experiment": "l10_sc15_passive_action_conditioned_observation_gain_v0",
        "decision": decision,
        "source": {
            "videos": args.videos,
            "eligible_tracks": track_count,
            "authority": "reused Development natural video; evaluator GT used only after frozen action output",
            "input_hashes_sha256": {
                str(args.ocr_cache): sha256_file(args.ocr_cache),
                str(args.embedding_cache): sha256_file(args.embedding_cache),
                str(args.embedding_index): sha256_file(args.embedding_index),
            },
        },
        "frozen_protocol": {
            "controller": "unchanged L10SC2OpportunityActiveSearch",
            "horizon_track_frames": HORIZON_TRACK_FRAMES,
            "semantic_gate": SEMANTIC_GATE,
            "alignment": {
                "PAN_OR_SCAN": "absolute target-center error decreases",
                "APPROACH": "target normalized area increases",
            },
            "outcome": "future minus current target-associated OCR lexical score",
            "independence": "action is emitted before evaluator target geometry and association are read",
            "gates": {
                "minimum_opportunities": MIN_OPPORTUNITIES,
                "minimum_per_alignment_arm": MIN_PER_ALIGNMENT_ARM,
                "minimum_mean_gain_delta": MIN_MEAN_GAIN_DELTA,
                "minimum_improvement_rate_delta": MIN_IMPROVEMENT_RATE_DELTA,
                "maximum_wrong_gate_rate_penalty": MAX_WRONG_GATE_RATE_PENALTY,
            },
            "sweeps": [],
        },
        "opportunities": {
            "total": len(rows),
            "by_action": dict(sorted(actions.items())),
            "by_alignment": dict(sorted(Counter(row["alignment"] for row in rows).items())),
        },
        "comparison": comparison,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "rows": rows,
        "claim_boundary": (
            "Passive natural-transition Development evidence only. The recorded camera motion was not caused by "
            "the controller, so a positive association is not executed active-view causality, live-camera "
            "capability, metric arrival, product benefit, user benefit, or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "opportunities": result["opportunities"],
                "comparison": comparison,
                "gate": result["gate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
