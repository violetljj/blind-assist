from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import artvideo_dual_state_replay as sc1w
import artvideo_ocr_replay as replay
import artvideo_semantic_visual_replay as sc0


@dataclass(frozen=True)
class ActiveDecision:
    decision: sc1w.DualStateDecision
    source_action_mode: str
    source_semantic_state: str
    observation_action: str


class L10SC2OpportunityActiveSearch:
    """Turn observable semantic deficits into bounded camera-relative actions.

    SC1W still owns identity and continuity. This layer removes one invalid
    inference: a non-exhaustive OCR candidate set cannot prove semantic NONE.
    """

    name = "L10_SC2_opportunity_active_search"

    def __init__(self, goal: str):
        self.tracker = sc1w.L10SC1DualState(goal)

    @staticmethod
    def _center_action(center: tuple[float, float] | None, centered_action: str) -> str:
        if center is None:
            return "HOLD_STEADY_RECOGNIZE"
        bearing = sc1w._direction(center)
        if bearing == "LEFT":
            return "PAN_LEFT_TO_TEXT"
        if bearing == "RIGHT":
            return "PAN_RIGHT_TO_TEXT"
        return centered_action

    def _plan(self, decision: sc1w.DualStateDecision) -> str:
        if decision.action_mode == "NAVIGATE":
            bearing = sc1w._direction(decision.carrier_center or self.tracker.center or (0.5, 0.5))
            return f"NAVIGATE_{bearing}"
        if decision.action_mode == "SEARCH":
            if self.tracker.long is None or self.tracker.center is None:
                return "SWEEP_SEARCH"
            return f"SCAN_LAST_{sc1w._direction(self.tracker.center)}"
        if decision.action_mode == "OBSERVE":
            if decision.evidence_deficit == "REACQUIRE_CONFIRMATION_PENDING":
                return "HOLD_STEADY_CONFIRM"
            if decision.evidence_deficit == "ASSOCIATION_GATE_FAILED":
                return self._center_action(decision.carrier_center, "SIDESTEP_FOR_DISAMBIGUATION")
            return self._center_action(decision.carrier_center, "APPROACH_FOR_TEXT")
        return "STOP"

    def step(self, candidates: list[Any]) -> ActiveDecision:
        source = self.tracker.step(candidates)
        normalized = source
        if source.semantic_state == sc1w.SemanticState.NONE.value and source.evidence_deficit == "CLEAR_CANDIDATE_SET_NONE":
            normalized = replace(
                source,
                semantic_state=sc1w.SemanticState.UNKNOWN.value,
                action_mode="SEARCH",
                evidence_deficit="TARGET_NOT_PROPOSED",
            )
        return ActiveDecision(
            decision=normalized,
            source_action_mode=source.action_mode,
            source_semantic_state=source.semantic_state,
            observation_action=self._plan(normalized),
        )


def _identity_correct(decision: sc1w.DualStateDecision, candidates: list[Any], target_id: int) -> bool:
    return decision.identity_index is not None and target_id in candidates[decision.identity_index].truth_ids


def _target_detection_exists(candidates: list[Any], target_id: int) -> bool:
    return any(target_id in candidate.truth_ids for candidate in candidates)


def _target_lexical_exists(candidates: list[Any], target_id: int, goal: str) -> bool:
    return any(
        target_id in candidate.truth_ids and replay.lexical(goal, candidate.text) >= sc0.REACQUIRE_GATES[0]
        for candidate in candidates
    )


def _action_alignment(
    action: str,
    video: dict[str, Any],
    frame_id: int,
    target_id: int,
) -> bool | None:
    target_bearing = sc1w._direction(sc1w._target_center(video, frame_id, target_id))
    if action == "PAN_LEFT_TO_TEXT":
        return target_bearing == "LEFT"
    if action == "PAN_RIGHT_TO_TEXT":
        return target_bearing == "RIGHT"
    if action.startswith("SCAN_LAST_"):
        return action.removeprefix("SCAN_LAST_") == target_bearing
    return None


def run_episode(
    video: dict[str, Any],
    target_id: int,
    goal: str,
    frame_ids: list[int],
    gap_start: int,
    gap_length: int,
) -> dict[str, Any]:
    controller = L10SC2OpportunityActiveSearch(goal)
    gap_positions = set(range(gap_start, min(len(frame_ids), gap_start + gap_length)))
    gap_end_position = min(len(frame_ids), gap_start + gap_length)
    counts: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    deficits: Counter[str] = Counter()
    pre_gap_identity_established = False
    pre_gap_target_detection = False
    pre_gap_target_lexical = False
    reacquire_latency: int | None = None

    for position, frame_id in enumerate(frame_ids):
        full_candidates = video["candidates"].get(frame_id, [])
        candidates = full_candidates
        if position in gap_positions:
            candidates = [candidate for candidate in full_candidates if target_id not in candidate.truth_ids]
        active = controller.step(candidates)
        decision = active.decision
        action = active.observation_action
        actions[action] += 1
        deficits[decision.evidence_deficit or "NONE"] += 1
        counts["processed_frames"] += 1
        if active.source_action_mode == "STOP":
            counts["source_stop_frames"] += 1
        if decision.action_mode == "STOP":
            counts["normalized_stop_frames"] += 1
        if decision.action_mode != "NAVIGATE":
            counts["non_navigation_frames"] += 1
            if action != "STOP":
                counts["explicit_non_navigation_action_frames"] += 1
        if decision.identity_index is not None and not decision.fresh_semantic:
            counts["appearance_only_identity_violations"] += 1
        if decision.action_mode == "NAVIGATE" and decision.identity_index is None:
            counts["continuity_only_navigation_violations"] += 1

        if position < gap_start:
            pre_gap_target_detection = pre_gap_target_detection or _target_detection_exists(full_candidates, target_id)
            pre_gap_target_lexical = pre_gap_target_lexical or _target_lexical_exists(full_candidates, target_id, goal)
            pre_gap_identity_established = pre_gap_identity_established or _identity_correct(decision, candidates, target_id)

        if position in gap_positions:
            counts["gap_frames"] += 1
            if decision.identity_index is not None:
                counts["false_identity_gap_frames"] += 1
            continue

        counts["target_present_frames"] += 1
        if active.source_semantic_state == sc1w.SemanticState.NONE.value:
            counts["source_false_none_target_present_frames"] += 1
        if decision.semantic_state == sc1w.SemanticState.NONE.value:
            counts["normalized_false_none_target_present_frames"] += 1
        if decision.semantic_state == sc1w.SemanticState.UNKNOWN.value:
            counts["unknown_target_present_frames"] += 1
        if _identity_correct(decision, candidates, target_id):
            counts["correct_identity_target_frames"] += 1
            if position >= gap_end_position and reacquire_latency is None:
                reacquire_latency = position - gap_end_position
        elif decision.identity_index is None:
            counts["identity_missed_target_frames"] += 1
        else:
            counts["wrong_identity_target_present_frames"] += 1

        if decision.action_mode in {"SEARCH", "OBSERVE"}:
            counts["active_request_target_present_frames"] += 1
        aligned = _action_alignment(action, video, frame_id, target_id)
        if aligned is not None:
            counts["evaluable_directional_active_actions"] += 1
            counts["aligned_directional_active_actions" if aligned else "misaligned_directional_active_actions"] += 1

    reacquired_within_window = reacquire_latency is not None and reacquire_latency <= 5
    if pre_gap_identity_established:
        failure_layer = "REACQUIRED" if reacquired_within_window else "POST_LOCK_REACQUIRE_FAILED"
    elif not pre_gap_target_detection:
        failure_layer = "NO_CANDIDATE"
    elif not pre_gap_target_lexical:
        failure_layer = "CANDIDATE_REJECTED"
    else:
        failure_layer = "CONFIRMATION_FAILED"
    return {
        "video": video["name"],
        "target_id": target_id,
        "goal": goal,
        "gap_start": gap_start,
        "counts": dict(counts),
        "actions": dict(actions),
        "deficits": dict(deficits),
        "pre_gap_identity_established": pre_gap_identity_established,
        "pre_gap_target_detection": pre_gap_target_detection,
        "pre_gap_target_lexical": pre_gap_target_lexical,
        "reacquire_latency": reacquire_latency,
        "reacquired_within_window": reacquired_within_window,
        "end_to_end_success": pre_gap_identity_established and reacquired_within_window,
        "failure_layer": failure_layer,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    deficits: Counter[str] = Counter()
    layers: Counter[str] = Counter()
    for row in rows:
        counts.update(row["counts"])
        actions.update(row["actions"])
        deficits.update(row["deficits"])
        layers[row["failure_layer"]] += 1
    opportunities = [row for row in rows if row["pre_gap_identity_established"]]
    conditional_successes = [row for row in opportunities if row["reacquired_within_window"]]
    end_to_end = [row for row in rows if row["end_to_end_success"]]
    latencies = [row["reacquire_latency"] for row in conditional_successes]
    directional = counts["evaluable_directional_active_actions"]
    non_navigation = counts["non_navigation_frames"]
    never_acquired_targets = sorted(
        {(row["video"], row["target_id"], row["goal"]) for row in rows if not row["pre_gap_identity_established"]}
    )
    return {
        "episodes": len(rows),
        "pre_gap_identity_established_episodes": len(opportunities),
        "pre_gap_identity_not_established_episodes": len(rows) - len(opportunities),
        "acquisition_conditioned_reacquired_episodes": len(conditional_successes),
        "acquisition_conditioned_reacquire_rate": round(len(conditional_successes) / max(1, len(opportunities)), 4),
        "end_to_end_successful_episodes": len(end_to_end),
        "end_to_end_success_rate": round(len(end_to_end) / max(1, len(rows)), 4),
        "median_acquisition_conditioned_reacquire_frames": statistics.median(latencies) if latencies else None,
        "failure_layers": dict(sorted(layers.items())),
        "never_acquired_targets": [
            {"video": video, "target_id": target_id, "goal": goal}
            for video, target_id, goal in never_acquired_targets
        ],
        "source_stop_frames": counts["source_stop_frames"],
        "normalized_stop_frames": counts["normalized_stop_frames"],
        "source_false_none_target_present_frames": counts["source_false_none_target_present_frames"],
        "normalized_false_none_target_present_frames": counts["normalized_false_none_target_present_frames"],
        "unknown_target_present_frames": counts["unknown_target_present_frames"],
        "explicit_non_navigation_action_coverage": round(
            counts["explicit_non_navigation_action_frames"] / max(1, non_navigation), 4
        ),
        "directional_active_action_alignment": round(
            counts["aligned_directional_active_actions"] / max(1, directional), 4
        ),
        "directional_active_action_denominator": directional,
        "action_frames": dict(sorted(actions.items())),
        "deficit_frames": dict(sorted(deficits.items())),
        "authority_violations": {
            "appearance_only_identity": counts["appearance_only_identity_violations"],
            "continuity_only_navigation": counts["continuity_only_navigation_violations"],
        },
    }


def load_partition(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]]]:
    ocr_cache = json.loads(args.ocr_cache.read_text(encoding="utf-8"))
    index_payload = json.loads(args.embedding_index.read_text(encoding="utf-8"))
    with np.load(args.embedding_cache) as payload:
        pooled_patch = payload["pooled_patch"].astype(np.float32, copy=True)
    annotation_paths = [args.dataset / "Test" / "json" / f"{video}.json" for video in args.videos]
    videos = [replay.load_video(args.dataset, path, ocr_cache) for path in annotation_paths]
    sc1w.attach_frame_geometry(videos, annotation_paths, ocr_cache)
    sc0.attach_embeddings(videos, annotation_paths, index_payload, pooled_patch)
    specs: list[tuple[Any, ...]] = []
    for video in videos:
        for target_id, goal, frame_ids in replay.eligible_tracks(video, args.minimum_frames):
            for fraction in (0.30, 0.50, 0.70):
                start = max(2, min(len(frame_ids) - args.gap_length - 3, int(len(frame_ids) * fraction)))
                specs.append((video, target_id, goal, frame_ids, start, args.gap_length))
    return videos, specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L10-SC2 opportunity-correct active-search replay.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=12)
    parser.add_argument("--gap-length", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    _videos, specs = load_partition(args)
    source_rows = [sc1w.run_dual_episode(*spec) for spec in specs]
    active_rows = [run_episode(*spec) for spec in specs]
    source_metrics = sc1w.summarize_dual(source_rows)
    active_metrics = summarize(active_rows)
    checks = {
        "identity_authority_violations_zero": active_metrics["authority_violations"]["appearance_only_identity"] == 0,
        "continuity_only_navigation_violations_zero": active_metrics["authority_violations"]["continuity_only_navigation"] == 0,
        "incomplete_ocr_set_never_emits_none": active_metrics["normalized_false_none_target_present_frames"] == 0,
        "no_terminal_stop_from_ocr_non_match": active_metrics["normalized_stop_frames"] == 0,
        "every_non_navigation_frame_has_explicit_action": active_metrics["explicit_non_navigation_action_coverage"] == 1.0,
        "opportunity_denominators_reconcile": (
            active_metrics["pre_gap_identity_established_episodes"]
            + active_metrics["pre_gap_identity_not_established_episodes"]
            == active_metrics["episodes"]
        ),
        "end_to_end_success_not_above_conditioned_success": (
            active_metrics["end_to_end_successful_episodes"]
            <= active_metrics["acquisition_conditioned_reacquired_episodes"]
        ),
    }
    result = {
        "schema": "l10_sc2_opportunity_active_search_replay_v0",
        "status": "SC2_ACTIVE_SEARCH_REPRESENTATION_EFFECT" if all(checks.values()) else "SC2_ACTIVE_SEARCH_GATE_NOT_MET",
        "videos": args.videos,
        "algorithm": {
            "identity_and_continuity": "Frozen SC1W; this layer cannot create identity or navigation authority.",
            "absence_correction": "A non-exhaustive OCR candidate set cannot prove semantic NONE; lexical non-match becomes UNKNOWN/TARGET_NOT_PROPOSED and continues SEARCH.",
            "opportunity_accounting": "LOST-to-REACQUIRE is scored only after evaluator-confirmed pre-gap identity establishment; end-to-end acquisition-plus-reacquisition keeps the full denominator.",
            "deficit_actions": {
                "no prior anchor": "SWEEP_SEARCH",
                "lost after anchor": "SCAN_LAST_LEFT/FORWARD/RIGHT",
                "off-center readable proposal": "PAN_LEFT/RIGHT_TO_TEXT",
                "centered unreadable token": "APPROACH_FOR_TEXT",
                "association ambiguity": "SIDESTEP_FOR_DISAMBIGUATION",
                "two-hit pending": "HOLD_STEADY_CONFIRM",
            },
        },
        "episode_protocol": {
            "eligible_tracks": len(specs) // 3,
            "episodes": len(specs),
            "minimum_track_frames": args.minimum_frames,
            "artificial_gap_frames": args.gap_length,
            "gap_positions": [0.30, 0.50, 0.70],
            "reacquire_window_frames": 5,
        },
        "frozen_sc1w_metrics": source_metrics,
        "sc2_metrics": active_metrics,
        "per_episode": active_rows,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "runtime_s": round(time.perf_counter() - started, 4),
        "claim_ceiling": (
            "ArTVideo replay representation and action-interface evidence only. Directional alignment is evaluator-posthoc; "
            "the replay does not execute camera actions or establish causal readability gain. It is not live active-view, "
            "open-world absence, metric arrival, product, user-benefit, or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "metrics": active_metrics, "gate": result["gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
