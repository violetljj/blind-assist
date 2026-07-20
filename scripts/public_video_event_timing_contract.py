"""Validation helpers for the frozen prospective r7.50 event timing contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_event_timing_contract_v1"
JAPAN_SOURCE_ID = "wikimedia_commons_japan_rural_riverside_walk_2025"


def _require_sha256(value: dict[str, Any], key: str) -> None:
    digest = value.get("bound_inputs", {}).get(key)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"invalid bound input hash: {key}")


def validate_contract(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SCHEMA:
        raise ValueError("unexpected r7.50 contract schema")
    disclosure = str(value.get("derivation_disclosure", ""))
    if "3000 ms" not in disclosure or "r7.49 Japan" not in disclosure or "remains an r7.49 timing failure" not in disclosure:
        raise ValueError("r7.50 derivation disclosure must preserve the Japan failure")
    expected_entry = {
        "requires_frozen_r725_radial_prefix_pass": True,
        "requires_route_intrusion_above_trusted_pre_risk_clear_baseline": True,
        "route_relation_implementation": "exactly_bound_r747_report",
        "learned_parameters": 0,
        "post_review_threshold_or_rule_changes_forbidden": True,
    }
    if value.get("entry") != expected_entry:
        raise ValueError("r7.50 entry differs from the frozen rule")
    timing = value.get("review_timing_fields", {})
    if timing != {
        "material_risk_onset_ms_required": True,
        "latest_useful_reminder_ms_required": True,
        "stable_post_clear_window_ms_required": True,
        "maximum_early_warning_lead_ms": 3000,
        "accepted_reminder_interval": "material_risk_onset_ms - 3000 <= reminder_timestamp_ms <= latest_useful_reminder_ms",
    }:
        raise ValueError("r7.50 timing band differs from the frozen rule")
    if value.get("lifecycle") != {
        "sample_interval_ms": 1000,
        "post_entry_chromatic_evidence_resets_absence_run": True,
        "clear_absent_samples": 9,
        "same_visual_episode_reminder_count": 1,
        "reopen_after_clear_requires_new_frozen_r725_radial_candidate": True,
    }:
        raise ValueError("r7.50 lifecycle differs from the frozen rule")
    isolation = value.get("prospective_isolation", {})
    if isolation.get("source_must_not_have_influenced_r725_r730_r747_r748_or_r749") is not True:
        raise ValueError("r7.50 prospective isolation is incomplete")
    if isolation.get("japan_source_id_forbidden_for_acceptance") != JAPAN_SOURCE_ID:
        raise ValueError("Japan must remain forbidden for r7.50 acceptance")
    acceptance = value.get("acceptance", {})
    for key in (
        "independent_real_positive_event_required",
        "independent_real_true_radial_safe_lateral_negative_required",
        "negative_must_contain_true_frozen_radial_entry",
        "negative_must_be_vetoed_by_frozen_route_relation",
        "positive_reminder_must_land_in_fixed_interval",
        "positive_must_not_clear_before_reviewed_risk_end",
        "positive_clear_must_land_inside_stable_post_clear_window",
        "all_required_roles_must_use_distinct_real_source_sha256",
    ):
        if acceptance.get(key) is not True:
            raise ValueError(f"missing r7.50 acceptance gate: {key}")
    if acceptance.get("same_visual_episode_reminder_count") != 1:
        raise ValueError("r7.50 reminder count must remain one")
    if acceptance.get("synthetic_or_gpt_only_examples_receive_gate_credit") is not False:
        raise ValueError("synthetic or GPT-only evidence cannot receive r7.50 gate credit")
    for key in (
        "radial_approach_contract_sha256",
        "gap_bridge_contract_sha256",
        "explicit_route_relation_report_sha256",
        "event_risk_profile_lifecycle_report_sha256",
        "japan_causal_replay_report_sha256",
        "true_radial_negative_source_search_sha256",
    ):
        _require_sha256(value, key)
    if any(value.get("authorization", {}).values()):
        raise ValueError("r7.50 contract must not authorize downstream use")
    return value


def validate_review_template(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != "blindassist_public_video_event_timing_review_v1":
        raise ValueError("unexpected r7.50 review schema")
    if value.get("review_status") != "template_only_not_reviewed":
        raise ValueError("r7.50 review template must remain unreviewed")
    if value.get("role") != "prospective_positive_event_or_true_radial_safe_lateral_negative":
        raise ValueError("r7.50 review template must expose both required roles")
    positive = value.get("prospective_positive_event", {})
    negative = value.get("true_radial_safe_lateral_negative", {})
    if not {"material_risk_onset_ms", "latest_useful_reminder_ms", "stable_post_clear_window_ms"}.issubset(positive):
        raise ValueError("r7.50 positive timing fields are incomplete")
    if not {"frozen_radial_entry_present", "obstacle_remains_safely_lateral_to_ego_route", "route_relation_should_veto_event_entry"}.issubset(negative):
        raise ValueError("r7.50 true-radial negative fields are incomplete")
    if any(value.get("authorization", {}).values()):
        raise ValueError("r7.50 review template must not authorize downstream use")
    return value


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    value = lifecycle.verify_json_sidecar(path)
    validate_contract(value)
    return value, {"path": str(path.resolve()), "sha256": common.sha256_file(path)}
