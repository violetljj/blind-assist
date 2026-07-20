#!/usr/bin/env python3
"""Frozen mechanism-specific multi-cone risk-evidence policy."""

from __future__ import annotations

import copy
from typing import Any, Sequence


POLICY_ID = "multi_cone_corridor_v1"
CLASS_NAME = "traffic cone"


def validate_policy(contract: dict[str, Any]) -> dict[str, Any]:
    policy = contract.get("risk_evidence_policy")
    if not isinstance(policy, dict):
        raise ValueError("multi-cone risk-evidence policy is missing")
    if policy.get("policy_id") != POLICY_ID:
        raise ValueError("unexpected multi-cone policy ID")
    if policy.get("class_name") != CLASS_NAME:
        raise ValueError("multi-cone policy class differs from the frozen class")
    if policy.get("minimum_count_per_frame") != 2:
        raise ValueError("multi-cone count threshold differs from the frozen value")
    if policy.get("other_semantic_classes_used") != []:
        raise ValueError("multi-cone policy must not consume other semantic classes")
    if policy.get("geometry_gate_used") is not False:
        raise ValueError("multi-cone policy geometry gate differs from the frozen value")
    return policy


def apply_policy(
    samples: Sequence[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    minimum_count = int(policy["minimum_count_per_frame"])
    class_name = str(policy["class_name"])
    result: list[dict[str, Any]] = []
    for sample in samples:
        row = copy.deepcopy(sample)
        class_counts = sample.get("semantic_class_counts", {})
        count = int(class_counts.get(class_name, 0)) if isinstance(class_counts, dict) else 0
        row["semantic_group_counts"] = (
            {"barrier_structure": 1} if count >= minimum_count else {}
        )
        row["semantic_class_counts"] = (
            {class_name: count} if count >= minimum_count else {}
        )
        result.append(row)
    return result
