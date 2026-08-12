#!/usr/bin/env python3
"""Frozen source-only rule search for FARO-clear parent enrichment."""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


SCHEMA = "blindassist.taro.o1r.r9_source_only_clear_enrichment_selector.v1"
SELECTOR_ID = "TARO_R9_SOURCE_ONLY_CLEAR_ENRICHMENT_GRID_SEARCH_V1"
STATE_POLICIES = ("UNKNOWN_ONLY", "NON_OCCUPIED")
MINIMUM_FAR_ANCHORS = (0, 6, 12, 18, 24, 30)
MAXIMUM_FAR_ANCHORS = (18, 24, 30, 36, 1000000)
FAR_FRACTION_INDICES = (0, 1, 2)
MAXIMUM_FAR_FRACTIONS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
MINIMUM_SUPPORT_POINTS = (0, 128, 10000, 50000, 100000, 200000, 400000)
SELECTED_PARENT_COUNT = 8
MINIMUM_NONZERO_SELECTED_PARENTS = 4
MINIMUM_CLEAR_QUERIES = 50
MINIMUM_CLEAR_PARENTS = 4


class ClearEnrichmentFitError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ClearEnrichmentFitError(code, message)


def _rule_id(rule: Mapping[str, Any]) -> str:
    return adapter.canonical_sha256(dict(rule))[:16]


def candidate_rules() -> list[dict[str, Any]]:
    output = []
    for state_policy, minimum_anchors, maximum_anchors, fraction_index, maximum_fraction, minimum_support in itertools.product(
        STATE_POLICIES,
        MINIMUM_FAR_ANCHORS,
        MAXIMUM_FAR_ANCHORS,
        FAR_FRACTION_INDICES,
        MAXIMUM_FAR_FRACTIONS,
        MINIMUM_SUPPORT_POINTS,
    ):
        if maximum_anchors < minimum_anchors:
            continue
        rule = {
            "state_policy": state_policy,
            "minimum_far_valid_anchor_count": minimum_anchors,
            "maximum_far_valid_anchor_count": maximum_anchors,
            "far_fraction_index": fraction_index,
            "maximum_far_fraction": maximum_fraction,
            "minimum_observed_support_points": minimum_support,
            "require_query_receipt": True,
            "require_positive_obstacle_veto_false": True,
            "require_all_occupied_hits_false": True,
        }
        rule["rule_id"] = _rule_id(rule)
        output.append(rule)
    require(len({row["rule_id"] for row in output}) == len(output), "R9_RULE_ID_COLLISION", "source-only rule id collision")
    return output


def eligible(feature: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    if feature.get("query_receipt") is None:
        return False
    state = feature.get("r6_state")
    if rule["state_policy"] == "UNKNOWN_ONLY" and state != "UNKNOWN":
        return False
    if rule["state_policy"] == "NON_OCCUPIED" and state == "OCCUPIED_OBSERVED":
        return False
    if feature.get("positive_obstacle_veto") is not False:
        return False
    hits = feature.get("occupied_hits")
    if not isinstance(hits, list) or any(bool(value) for height in hits for forward in height for value in forward):
        return False
    anchors = feature.get("far_valid_anchor_count")
    fractions = feature.get("far_fractions")
    support = feature.get("observed_support_points")
    if not isinstance(anchors, int) or not isinstance(support, int) or not isinstance(fractions, list) or len(fractions) != 3:
        return False
    return (
        rule["minimum_far_valid_anchor_count"] <= anchors <= rule["maximum_far_valid_anchor_count"]
        and float(fractions[rule["far_fraction_index"]]) <= rule["maximum_far_fraction"]
        and support >= rule["minimum_observed_support_points"]
    )


def _parent_identities(sources: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    identities = sorted({(str(source["parent_id"]), str(source["video_id"])) for source in sources})
    require(len(identities) >= SELECTED_PARENT_COUNT, "R9_PARENT_COUNT", "source-only fit needs at least eight parents")
    return identities


def fit_selector(sources: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(sources) == len(labels) and len(sources) > 0, "R9_FIT_COUNT", "source/label frame count drift")
    identities = _parent_identities(sources)
    parent_truth: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    rows = []
    for source, label_record in zip(sources, labels, strict=True):
        identity = (str(source["parent_id"]), str(source["video_id"]))
        require(source["physical_frame_id"] == label_record["physical_frame_id"] and len(source["query_features"]) == len(label_record["query_labels"]) == 9, "R9_FIT_ALIGNMENT", "source/label alignment drift")
        for feature, label in zip(source["query_features"], label_record["query_labels"], strict=True):
            require(feature["query_id"] == label["query_id"] and label["state"] in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}, "R9_FIT_LABEL", "source/label query drift")
            parent_truth[identity][label["state"]] += 1
            rows.append((identity, feature, label["state"]))
    candidate_count = 0
    best = None
    for rule in candidate_rules():
        candidate_count += 1
        scores: dict[tuple[str, str], int] = Counter()
        matched_truth: Counter[str] = Counter()
        for identity, feature, truth in rows:
            if eligible(feature, rule):
                scores[identity] += 1
                matched_truth[truth] += 1
        ranked = sorted(identities, key=lambda identity: (-scores[identity], adapter.canonical_sha256(list(identity))))
        selected = ranked[:SELECTED_PARENT_COUNT]
        selected_truth: Counter[str] = Counter()
        for identity in selected:
            selected_truth.update(parent_truth[identity])
        clear_parents = sum(parent_truth[identity]["CLEAR_OBSERVED"] > 0 for identity in selected)
        nonzero_selected = sum(scores[identity] > 0 for identity in selected)
        definite_matched = matched_truth["CLEAR_OBSERVED"] + matched_truth["OCCUPIED_OBSERVED"]
        matched_clear_precision = matched_truth["CLEAR_OBSERVED"] / definite_matched if definite_matched else 0.0
        target_met = selected_truth["CLEAR_OBSERVED"] >= MINIMUM_CLEAR_QUERIES and clear_parents >= MINIMUM_CLEAR_PARENTS and nonzero_selected >= MINIMUM_NONZERO_SELECTED_PARENTS
        objective = (
            int(target_met),
            min(clear_parents, MINIMUM_CLEAR_PARENTS),
            min(selected_truth["CLEAR_OBSERVED"], MINIMUM_CLEAR_QUERIES),
            nonzero_selected,
            matched_clear_precision,
            matched_truth["CLEAR_OBSERVED"],
            -matched_truth["OCCUPIED_OBSERVED"],
            -int(rule["rule_id"], 16),
        )
        if best is None or objective > best[0]:
            best = (objective, rule, scores, selected, selected_truth, clear_parents, nonzero_selected, matched_truth, matched_clear_precision, target_met)
    require(best is not None, "R9_FIT_EMPTY", "source-only rule search produced no candidate")
    _objective, rule, scores, selected, selected_truth, clear_parents, nonzero_selected, matched_truth, precision, target_met = best
    pool_truth: Counter[str] = Counter()
    for counts in parent_truth.values():
        pool_truth.update(counts)
    return {
        "schema": SCHEMA,
        "selector_id": SELECTOR_ID,
        "candidate_rule_count": candidate_count,
        "chosen_rule": rule,
        "development_parent_count": len(identities),
        "development_query_count": len(rows),
        "development_label_state_counts": {state: int(pool_truth[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
        "selected_parent_count": SELECTED_PARENT_COUNT,
        "selected_parent_identities": [list(identity) for identity in selected],
        "selected_parent_scores": [{"parent_id": identity[0], "video_id": identity[1], "eligible_query_count": int(scores[identity])} for identity in selected],
        "selected_nonzero_score_parent_count": nonzero_selected,
        "selected_label_state_counts": {state: int(selected_truth[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
        "selected_parents_with_clear": clear_parents,
        "matched_rule_label_state_counts": {state: int(matched_truth[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
        "matched_rule_clear_precision_on_definite_labels": precision,
        "development_target": {"minimum_clear_queries": MINIMUM_CLEAR_QUERIES, "minimum_clear_parents": MINIMUM_CLEAR_PARENTS, "minimum_nonzero_selected_parents": MINIMUM_NONZERO_SELECTED_PARENTS, "passed": bool(target_met)},
        "selection_uses_only_source_features": True,
        "confirmation_authority": False,
        "unknown_is_negative": False,
    }


def validate_selector(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require(record.get("schema") == SCHEMA and record.get("selector_id") == SELECTOR_ID and record.get("candidate_rule_count") == len(candidate_rules()), "R9_SELECTOR_IDENTITY", "selector identity/search-space drift")
    require(record.get("selected_parent_count") == SELECTED_PARENT_COUNT and record.get("selection_uses_only_source_features") is True and record.get("confirmation_authority") is False and record.get("unknown_is_negative") is False, "R9_SELECTOR_AUTHORITY", "selector authority drift")
    require(record.get("chosen_rule") in candidate_rules(), "R9_SELECTOR_RULE", "chosen selector rule leaves frozen search space")
    return record


__all__ = ["ClearEnrichmentFitError", "MINIMUM_CLEAR_PARENTS", "MINIMUM_CLEAR_QUERIES", "SELECTOR_ID", "candidate_rules", "eligible", "fit_selector", "validate_selector"]
