#!/usr/bin/env python3
"""Validate that the tracked G0 protocol matches the executable constants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import GroundAnchorPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "blindassist.ga_satom_g0.protocol.v1":
        raise ValueError("G0 protocol schema mismatch")
    policy = GroundAnchorPolicy()
    frozen = protocol["anchor_estimator"]
    expected = {
        "total_zone_budget": policy.total_zone_budget,
        "anchor_zone_ids": list(policy.anchor_zone_ids),
        "minimum_range_m": policy.minimum_range_m,
        "maximum_range_m": policy.maximum_range_m,
        "maximum_sigma_m": policy.maximum_sigma_m,
        "support_tolerance_m": policy.support_tolerance_m,
        "minimum_support_zones": policy.minimum_support_zones,
        "minimum_support_fraction": policy.minimum_support_fraction,
        "maximum_height_sigma_m": policy.maximum_height_sigma_m,
    }
    if frozen != expected:
        raise ValueError("G0 executable estimator drift")
    gates = protocol["winner_rule"]
    expected_gates = {
        "minimum_parent_count": 8,
        "parent_macro_coverage_min": 0.95,
        "worst_parent_coverage_min": 0.90,
        "parent_macro_height_mae_m_max": 0.08,
        "worst_parent_height_mae_m_max": 0.12,
        "worst_parent_height_error_mad_m_max": 0.08,
        "parent_macro_false_ground_support_rate_max": 0.02,
        "worst_parent_false_ground_support_rate_max": 0.05,
        "worst_parent_catastrophic_false_anchor_rate_max": 0.01,
        "worst_parent_temporal_jitter_p95_m_max": 0.08,
    }
    if gates != expected_gates:
        raise ValueError("G0 executable winner-rule drift")
    episode_grid = protocol["fresh_source_contract"]["episode_grid"]
    if episode_grid.get("episodes_per_parent") != 9:
        raise ValueError("G0 episode-count drift")
    if episode_grid.get("registered_time_slots_per_episode") != 300:
        raise ValueError("G0 registered time-slot drift")
    if episode_grid.get("registered_time_slots_per_parent") != 2700:
        raise ValueError("G0 parent denominator drift")
    if "INVALID" not in str(episode_grid.get("missing_capture_rule", "")):
        raise ValueError("G0 missing-capture denominator rule is absent")
    if protocol["authority"]["g0_runs_satom_arms"] is not False:
        raise ValueError("G0 must not authorize SATOM arms")
    print("GA_SATOM_G0_PROTOCOL_VALID")


if __name__ == "__main__":
    main()
