#!/usr/bin/env python3
"""Frozen chromatic construction-marker risk-evidence policy."""

from __future__ import annotations

import copy
from typing import Any, Sequence


POLICY_ID = "chromatic_construction_marker_v1"
TARGET_CLASSES = ("barricade", "traffic cone")


def validate_policy(contract: dict[str, Any]) -> dict[str, Any]:
    policy = contract.get("risk_evidence_policy")
    if not isinstance(policy, dict):
        raise ValueError("chromatic marker policy is missing")
    if policy.get("policy_id") != POLICY_ID:
        raise ValueError("unexpected chromatic marker policy ID")
    if policy.get("target_classes") != list(TARGET_CLASSES):
        raise ValueError("chromatic marker target classes differ from the frozen set")
    if policy.get("detection_acceptance") != "high_saturation_fraction > dark_fraction":
        raise ValueError("chromatic marker color rule differs from the frozen rule")
    if policy.get("minimum_accepted_detections_per_active_frame") != 1:
        raise ValueError("chromatic marker frame threshold differs from the frozen value")
    if policy.get("absolute_color_threshold_used") is not False:
        raise ValueError("chromatic marker policy must not use an absolute color threshold")
    if policy.get("geometry_gate_used") is not False:
        raise ValueError("chromatic marker policy geometry gate differs from the frozen value")
    return policy


def validate_extractor_binding(
    contract: dict[str, Any],
    *,
    weights_sha256: str,
    sample_interval_ms: int,
    image_size: int,
    confidence: float,
    target_classes: set[str],
) -> None:
    policy = validate_policy(contract)
    scan = contract["scan"]
    if weights_sha256 != contract["model"]["weights_sha256"]:
        raise ValueError("weights do not match the frozen chromatic marker contract")
    if sample_interval_ms != scan["sample_interval_ms"]:
        raise ValueError("sample interval does not match the frozen chromatic marker contract")
    if image_size != scan["image_size"]:
        raise ValueError("image size does not match the frozen chromatic marker contract")
    if abs(confidence - float(scan["confidence"])) > 1e-12:
        raise ValueError("confidence does not match the frozen chromatic marker contract")
    if target_classes != set(policy["target_classes"]):
        raise ValueError("extractor target classes do not match the frozen chromatic marker contract")


def apply_policy(
    samples: Sequence[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    target_classes = set(policy["target_classes"])
    for sample in samples:
        row = copy.deepcopy(sample)
        detections = sample.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError("feature sample detections must be a list")
        accepted = [
            detection for detection in detections
            if detection.get("class_name") in target_classes
            and float(detection["features"]["high_saturation_fraction"])
            > float(detection["features"]["dark_fraction"])
        ]
        row["semantic_group_counts"] = (
            {"barrier_structure": len(accepted)} if accepted else {}
        )
        row["semantic_class_counts"] = (
            {"chromatic construction marker": len(accepted)} if accepted else {}
        )
        result.append(row)
    return result
