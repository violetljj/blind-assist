"""Fail-closed mechanics for P0-D2 commitment calibration."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


AUTHORITY_STATES = {"RESOLVABLE", "REFERENT_AMBIGUOUS", "INSUFFICIENT"}
FORBIDDEN_FEATURE_KEYS = {
    "acceptable_spatial_regions", "end_to_end", "goal_reference_resolution", "grounding_expectation",
    "target_visible", "valid_target_instances", "reviewer_id", "resolution", "safe_commit_label",
}
REQUIRED_FEATURE_FAMILIES = {
    "PLACE_IDENTITY", "ENTRANCE_RELATION", "CANDIDATE_COMPETITION", "BRAIN_RANK_MARGIN",
}


class CalibrationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def conformal_quantile(nonconformity_scores: Sequence[float], alpha: float) -> float:
    """Return the finite-sample split-conformal quantile.

    The caller is responsible for supplying one exchangeable score per venue
    parent.  Frames from the same parent must be reduced before this function.
    """
    _require(0.0 < alpha < 1.0, "alpha must be in (0,1)")
    scores = sorted(float(value) for value in nonconformity_scores)
    _require(scores and all(math.isfinite(value) for value in scores), "finite parent scores required")
    minimum = math.ceil(1.0 / alpha) - 1
    _require(len(scores) >= minimum, f"at least {minimum} independent calibration parents required for alpha={alpha}")
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1.0 - alpha)))
    return scores[rank - 1]


def calibrated_action(authority_state: str, referent_ids: Sequence[str]) -> str:
    _require(authority_state in AUTHORITY_STATES, "unknown evidence-authority state")
    referents = [str(value) for value in referent_ids]
    _require(len(referents) == len(set(referents)), "duplicate referent")
    if authority_state == "INSUFFICIENT":
        _require(not referents, "INSUFFICIENT cannot carry referents")
        return "ABSTAIN"
    if authority_state == "REFERENT_AMBIGUOUS":
        _require(not referents, "REFERENT_AMBIGUOUS cannot masquerade as a legal referent set")
        return "AMBIGUOUS"
    _require(referents, "RESOLVABLE requires at least one referent")
    return "COMMIT" if len(referents) == 1 else "SET"


def validate_runtime_features(row: Mapping[str, Any]) -> None:
    keys = set(row)
    leaked = sorted(keys & FORBIDDEN_FEATURE_KEYS)
    _require(not leaked, f"evaluator-only feature leakage: {', '.join(leaked)}")
    families = row.get("observed_feature_families")
    _require(isinstance(families, list), "observed_feature_families missing")
    unknown = set(str(value) for value in families) - REQUIRED_FEATURE_FAMILIES
    _require(not unknown, f"unknown feature families: {sorted(unknown)}")


def audit_data_frontdoor(cohorts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_parent: dict[str, set[str]] = defaultdict(set)
    episodes = 0
    for cohort in cohorts:
        _require(
            cohort.get("claim_ceiling") == "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY",
            "input exceeds or differs from Silver-B Development authority",
        )
        for episode in cohort.get("episodes", []):
            evaluator = episode["evaluator_episode"]
            parent = str(evaluator["goal_spec"]["target_name"])
            resolution = str(evaluator["goal_reference_resolution"])
            _require(resolution in {"UNIQUE", "SET_VALUED", "AMBIGUOUS"}, "unknown resolution")
            by_parent[parent].add(resolution)
            episodes += 1
    parent_counts = {
        resolution: sum(resolution in values for values in by_parent.values())
        for resolution in ("UNIQUE", "SET_VALUED", "AMBIGUOUS")
    }
    requirements = {
        "UNIQUE": 8,
        "SET_VALUED": 4,
        "AMBIGUOUS": 12,
        "RESOLVABLE_TOTAL": 12,
    }
    observed_resolvable = len({parent for parent, values in by_parent.items() if values & {"UNIQUE", "SET_VALUED"}})
    failures = []
    for resolution in ("UNIQUE", "SET_VALUED", "AMBIGUOUS"):
        if parent_counts[resolution] < requirements[resolution]:
            failures.append(f"{resolution}_PARENTS_{parent_counts[resolution]}_LT_{requirements[resolution]}")
    if observed_resolvable < requirements["RESOLVABLE_TOTAL"]:
        failures.append(f"RESOLVABLE_PARENTS_{observed_resolvable}_LT_{requirements['RESOLVABLE_TOTAL']}")
    return {
        "status": "P0_D2_DATA_FRONTDOOR_PASS" if not failures else "P0_D2_DATA_FRONTDOOR_INSUFFICIENT",
        "episode_count": episodes,
        "venue_parent_count": len(by_parent),
        "parent_counts": parent_counts,
        "resolvable_parent_count": observed_resolvable,
        "requirements": requirements,
        "failures": failures,
        "calibrator_fit_authorized": not failures,
        "conformal_fit_authorized": not failures,
        "claim_ceiling": "CONSUMED_DEVELOPMENT_DATA_SUFFICIENCY_ONLY_NO_CALIBRATION_OR_MODEL_PERFORMANCE_CLAIM",
    }
