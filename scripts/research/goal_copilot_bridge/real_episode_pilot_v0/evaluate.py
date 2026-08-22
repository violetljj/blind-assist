"""Evaluate speech/action-time and episode metrics for the exploratory pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from .truth_contract import TruthAuthorityTier, validate_annotation


KS = (1, 3, 5, 10)
RANGE_ORDER = {"RANGE_FAR": 0, "RANGE_APPROACHING": 1, "RANGE_NEAR": 2}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


TIER_EVIDENCE_LABELS = {
    "NATIVE_GT": "STRONG_RESULT",
    "MAP_TRAJECTORY_DERIVED": "NARROW_MAP_OR_GEOMETRY_RESULT",
    "TEACHER_SUPPORTED": "WEAK_EVIDENCE_RESULT",
    "TEACHER_ONLY_WEAK": "EXPLORATORY_ONLY_NOT_FUNCTIONAL_TRUTH",
    "UNKNOWN": "NOT_EVALUABLE",
}


def _tier_counter() -> dict[str, Any]:
    return {
        "total": 0,
        "functional_authority_established": 0,
        "visibility_eligible": 0,
        "visible": 0,
        "visibility_unknown": 0,
        "proposal_eligible": 0,
        "proposal_hits": Counter(),
        "selection_eligible": 0,
        "selection_correct": 0,
        "confident_eligible": 0,
        "confident_correct": 0,
        "confident_unknown": 0,
        "range_errors": [],
        "range_unknown": 0,
    }


def _serialize_tier_metrics(value: dict[str, Any], tier: str) -> dict[str, Any]:
    return {
        "evidence_label": TIER_EVIDENCE_LABELS[tier],
        "observations": value["total"],
        "functional_authority_established": value["functional_authority_established"],
        "target_visibility": {
            "visible": value["visible"],
            "eligible": value["visibility_eligible"],
            "unknown": value["visibility_unknown"],
            "rate": _rate(value["visible"], value["visibility_eligible"]),
        },
        "proposal_recall_at_k_given_visible_and_functional_authority": {
            str(k): {
                "hits": value["proposal_hits"][k],
                "eligible": value["proposal_eligible"],
                "rate": _rate(value["proposal_hits"][k], value["proposal_eligible"]),
            }
            for k in KS
        },
        "selection_accuracy_given_legal_candidate_and_functional_authority": {
            "correct": value["selection_correct"],
            "eligible": value["selection_eligible"],
            "rate": _rate(value["selection_correct"], value["selection_eligible"]),
        },
        "confident_guidance_referent_correctness": {
            "correct": value["confident_correct"],
            "eligible": value["confident_eligible"],
            "unknown": value["confident_unknown"],
            "rate": _rate(value["confident_correct"], value["confident_eligible"]),
        },
        "range_bucket_error": {
            "eligible": len(value["range_errors"]) + value["range_unknown"],
            "unknown_predictions": value["range_unknown"],
            "mean_ordinal_error": (
                sum(value["range_errors"]) / len(value["range_errors"])
                if value["range_errors"] else None
            ),
        },
    }


def evaluate(annotation: dict, prediction: dict) -> dict:
    validate_annotation(annotation)
    if prediction.get("schema_version") != "blindassist_real_episode_baseline_prediction_v0":
        raise ValueError("prediction schema mismatch")
    predicted = {row["observation_id"]: row for row in prediction["predictions"]}
    visibility_eligible = visible = visibility_unknown = 0
    proposal_hits = Counter()
    proposal_denominator = selection_denominator = selection_correct = 0
    confident = confident_correct = confident_unknown = 0
    state_counts, state_calibration = Counter(), Counter()
    range_errors, range_unknown = [], 0
    episodes_out, attributions = [], Counter()
    tier_metrics = {tier.value: _tier_counter() for tier in TruthAuthorityTier}
    tier_attributions = {tier.value: Counter() for tier in TruthAuthorityTier}
    for episode in annotation["episodes"]:
        episode_rows = []
        for truth in episode["observations"]:
            row = predicted.get(truth["observation_id"])
            if row is None:
                raise ValueError(f"missing prediction for {truth['observation_id']}")
            state_counts[row["decision_state"]] += 1
            tier = str(truth["truth_authority_tier"])
            tier_metric = tier_metrics[tier]
            tier_metric["total"] += 1
            functional_authority = truth["functional_authority"] == "ESTABLISHED"
            tier_metric["functional_authority_established"] += functional_authority
            allowed = set(truth.get("allowed_decision_states", []))
            if allowed:
                state_calibration["eligible"] += 1
                state_calibration["correct"] += row["decision_state"] in allowed
            visibility = truth["target_visibility"]
            if visibility == "UNKNOWN":
                tier_metric["visibility_unknown"] += 1
            else:
                tier_metric["visibility_eligible"] += 1
                tier_metric["visible"] += visibility == "VISIBLE"
            if visibility == "UNKNOWN" or not functional_authority or tier == "UNKNOWN":
                visibility_unknown += 1
            else:
                visibility_eligible += 1
                visible += visibility == "VISIBLE"
            legal = set(truth.get("legal_candidate_ids", []))
            candidates = row["candidate_ids"]
            present = any(candidate in legal for candidate in candidates)
            truth_eligible = functional_authority and tier != "UNKNOWN"
            if visibility == "VISIBLE" and legal and truth_eligible:
                proposal_denominator += 1
                tier_metric["proposal_eligible"] += 1
                for k in KS:
                    hit = any(candidate in legal for candidate in candidates[:k])
                    proposal_hits[k] += hit
                    tier_metric["proposal_hits"][k] += hit
            if present and truth_eligible:
                selection_denominator += 1
                selection_correct += row.get("selected_referent") in legal
                tier_metric["selection_eligible"] += 1
                tier_metric["selection_correct"] += row.get("selected_referent") in legal
            if row.get("confident_spoken_guidance"):
                if visibility == "VISIBLE" and legal and truth_eligible:
                    confident += 1
                    confident_correct += row.get("selected_referent") in legal
                    tier_metric["confident_eligible"] += 1
                    tier_metric["confident_correct"] += row.get("selected_referent") in legal
                else:
                    confident_unknown += 1
                    tier_metric["confident_unknown"] += 1
            truth_range = truth.get("range_truth", "RANGE_UNKNOWN")
            predicted_range = row.get("range_bucket", "RANGE_UNKNOWN")
            if truth_range in RANGE_ORDER:
                if predicted_range in RANGE_ORDER:
                    error = abs(RANGE_ORDER[truth_range] - RANGE_ORDER[predicted_range])
                    range_errors.append(error)
                    tier_metric["range_errors"].append(error)
                else:
                    range_unknown += 1
                    tier_metric["range_unknown"] += 1
            observation_attribution = None
            if tier == "UNKNOWN" or not functional_authority:
                observation_attribution = "TRUTH_OR_CONTRACT_INSUFFICIENT"
            elif visibility == "NOT_VISIBLE":
                observation_attribution = "CAMERA_POINTING_OR_VISIBILITY"
            elif visibility == "VISIBLE" and legal and not present:
                observation_attribution = "PROPOSAL_MISS"
            elif present and row.get("selected_referent") not in legal:
                observation_attribution = "REFERENT_SELECTION"
            elif truth_range in RANGE_ORDER and predicted_range != truth_range:
                observation_attribution = "RANGE_OR_GEOMETRY"
            if observation_attribution:
                tier_attributions[tier][observation_attribution] += 1
            episode_rows.append((truth, row, present))

        attribution = None
        if any(
            truth["target_visibility"] == "UNKNOWN"
            or truth["truth_authority_tier"] == "UNKNOWN"
            or truth["functional_authority"] != "ESTABLISHED"
            for truth, _, _ in episode_rows
        ):
            attribution = "TRUTH_OR_CONTRACT_INSUFFICIENT"
        elif not any(truth["target_visibility"] == "VISIBLE" for truth, _, _ in episode_rows):
            attribution = "CAMERA_POINTING_OR_VISIBILITY"
        elif any(truth["target_visibility"] == "VISIBLE" and truth.get("legal_candidate_ids") and not present for truth, _, present in episode_rows):
            attribution = "PROPOSAL_MISS"
        elif any(present and row.get("confident_spoken_guidance") and row.get("selected_referent") not in set(truth.get("legal_candidate_ids", [])) for truth, row, present in episode_rows):
            attribution = "REFERENT_SELECTION"
        elif any(truth.get("range_truth") in RANGE_ORDER and row.get("range_bucket") != truth.get("range_truth") for truth, row, _ in episode_rows):
            attribution = "RANGE_OR_GEOMETRY"
        elif episode.get("post_visibility_reacquisition_failure"):
            attribution = "POST_VISIBILITY_REACQUISITION"
        elif not episode.get("completed_by_user") or episode.get("user_denial_count", 0) > 0:
            attribution = "INTERACTION_OR_CONTROL"
        if attribution:
            attributions[attribution] += 1
        episodes_out.append({
            "episode_id": episode["episode_id"],
            "completed_by_user": bool(episode.get("completed_by_user")),
            "completion_time_ms": episode.get("completion_time_ms"),
            "instruction_count": int(episode.get("instruction_count", 0)),
            "correction_count": int(episode.get("correction_count", 0)),
            "user_denial_count": int(episode.get("user_denial_count", 0)),
            "handoff_additional_actions": episode.get("handoff_additional_actions"),
            "failure_attribution": attribution,
        })
    episode_count = len(episodes_out)
    completed = sum(row["completed_by_user"] for row in episodes_out)
    return {
        "schema_version": "blindassist_real_episode_exploratory_evaluation_v0",
        "claim_ceiling": "EXPLORATORY_FAILURE_ATTRIBUTION_ONLY_NO_USER_SAFETY_OR_PRODUCT_CLAIM",
        "observation_metrics": {
            "total_observations": sum(value["total"] for value in tier_metrics.values()),
            "truth_authority_distribution": {
                tier: value["total"] for tier, value in tier_metrics.items()
            },
            "by_truth_authority_tier": {
                tier: _serialize_tier_metrics(value, tier)
                for tier, value in tier_metrics.items()
            },
            "unconditional_target_visibility": {"visible": visible, "eligible": visibility_eligible, "unknown": visibility_unknown, "rate": _rate(visible, visibility_eligible)},
            "proposal_recall_at_k_given_visible": {str(k): {"hits": proposal_hits[k], "eligible": proposal_denominator, "rate": _rate(proposal_hits[k], proposal_denominator)} for k in KS},
            "selection_accuracy_given_legal_candidate_present": {"correct": selection_correct, "eligible": selection_denominator, "rate": _rate(selection_correct, selection_denominator)},
            "confident_spoken_guidance_referent_correctness": {"correct": confident_correct, "eligible": confident, "unknown": confident_unknown, "rate": _rate(confident_correct, confident)},
            "confident_wrong_referent_guidance": {"wrong": confident - confident_correct, "eligible": confident, "unknown": confident_unknown, "rate": _rate(confident - confident_correct, confident)},
            "decision_state_counts": dict(state_counts),
            "decision_state_calibration": {"correct": state_calibration["correct"], "eligible": state_calibration["eligible"], "rate": _rate(state_calibration["correct"], state_calibration["eligible"])},
            "range_bucket_error": {"eligible": len(range_errors) + range_unknown, "unknown_predictions": range_unknown, "mean_ordinal_error": sum(range_errors) / len(range_errors) if range_errors else None},
        },
        "episode_metrics": {
            "episode_count": episode_count,
            "completed": completed,
            "completion_rate": _rate(completed, episode_count),
            "total_instructions": sum(row["instruction_count"] for row in episodes_out),
            "total_corrections": sum(row["correction_count"] for row in episodes_out),
            "total_user_denials": sum(row["user_denial_count"] for row in episodes_out),
            "handoff_additional_actions": [row["handoff_additional_actions"] for row in episodes_out if row["handoff_additional_actions"] is not None],
        },
        "failure_attribution_counts": dict(attributions),
        "failure_attribution_by_truth_authority_tier": {
            tier: dict(counts) for tier, counts in tier_attributions.items()
        },
        "episodes": episodes_out,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("evaluation output already exists")
    payload = evaluate(
        json.loads(args.annotation.read_text(encoding="utf-8")),
        json.loads(args.prediction.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"episode_metrics": payload["episode_metrics"], "failure_attribution_counts": payload["failure_attribution_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
