"""Validation helpers for the frozen r7.30 asymmetric lifecycle contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_radial_lifecycle_gap_bridge_contract_v1"


def validate_contract(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SCHEMA:
        raise ValueError("unexpected r7.30 contract schema")
    lc = value.get("lifecycle", {})
    expected = {
        "sample_interval_ms": 1000,
        "entry_requires_frozen_r725_radial_candidate": True,
        "post_entry_chromatic_evidence_resets_absence_run": True,
        "absence_state_before_clear": "uncertain",
        "clear_absent_samples": 9,
        "continuation_emits_second_reminder": False,
        "reopen_after_clear_requires_new_frozen_r725_radial_candidate": True,
        "learned_parameters": 0,
    }
    if lc != expected:
        raise ValueError("r7.30 lifecycle differs from the frozen rule")
    acceptance = value.get("acceptance", {})
    if acceptance.get("same_visual_episode_reminder_count") != 1:
        raise ValueError("same-episode reminder contract differs from one")
    if acceptance.get("independent_negative_with_true_radial_entry_required") is not True:
        raise ValueError("true-entry negative stress requirement is missing")
    for key in (
        "radial_approach_contract_sha256",
        "chromatic_marker_contract_sha256",
        "r729_false_clear_result_sha256",
        "gap_bridge_probe_sha256",
    ):
        digest = value.get("bound_inputs", {}).get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid bound input hash: {key}")
    authorization = value.get("authorization", {})
    if any(authorization.values()):
        raise ValueError("r7.30 contract must not authorize downstream use")
    return value


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    value = lifecycle.verify_json_sidecar(path)
    validate_contract(value)
    return value, {"path": str(path.resolve()), "sha256": common.sha256_file(path)}
