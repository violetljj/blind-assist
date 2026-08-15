#!/usr/bin/env python3
"""Validate the frozen VITG G0 protocol constants and authority boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .capture_contract import EPISODE_TYPES, PROTOCOL_SCHEMA, SURFACE_STRATA
from .evaluation import EvaluationPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("VITG G0 protocol schema mismatch")
    source = protocol["fresh_source_contract"]
    if source.get("parent_count") != 8 or source.get("minimum_sites") != 2:
        raise ValueError("VITG G0 parent/site contract drift")
    if set(source.get("surface_strata", [])) != SURFACE_STRATA:
        raise ValueError("VITG G0 surface-strata drift")
    if set(source.get("episode_types", [])) != EPISODE_TYPES:
        raise ValueError("VITG G0 episode-type drift")
    capture = protocol["capture_admission"]
    if capture.get("imu_effective_rate_hz") != [100, 400]:
        raise ValueError("VITG G0 IMU rate gate drift")
    if capture.get("maximum_imu_gap_ms") != 20 or capture.get("maximum_nearest_imu_delta_ms") != 5:
        raise ValueError("VITG G0 RGB/IMU timing gate drift")
    if protocol["authority"].get("external_tof_authorized") is not False:
        raise ValueError("VITG G0 cannot authorize external ToF")
    if protocol["authority"].get("training_authorized") is not False:
        raise ValueError("VITG G0 cannot authorize training")
    if protocol["activation_contract"].get("outcome_access_authorized") is not False:
        raise ValueError("VITG G0 outcome access must remain closed before activation")
    policy = EvaluationPolicy()
    winner = protocol["winner_rule"]
    expected_winner = {
        "observable_parent_macro_coverage_min": policy.observable_parent_macro_coverage_min,
        "observable_worst_parent_coverage_min": policy.observable_worst_parent_coverage_min,
        "parent_macro_camera_height_mae_m_max": policy.parent_macro_camera_height_mae_m_max,
        "worst_parent_camera_height_mae_m_max": policy.worst_parent_camera_height_mae_m_max,
        "worst_parent_height_error_mad_m_max": policy.worst_parent_height_error_mad_m_max,
        "worst_parent_scale_drift_p95_fraction_max": policy.worst_parent_scale_drift_p95_fraction_max,
        "parent_macro_sparse_anchor_absrel_max": policy.parent_macro_sparse_anchor_absrel_max,
        "worst_parent_false_wall_ground_support_rate_max": policy.worst_parent_false_wall_ground_support_rate_max,
        "parent_macro_false_wall_ground_support_rate_max": policy.parent_macro_false_wall_ground_support_rate_max,
        "worst_parent_degeneracy_unsafe_valid_rate_max": policy.worst_parent_degeneracy_unsafe_valid_rate_max,
    }
    if any(winner.get(key) != value for key, value in expected_winner.items()):
        raise ValueError("VITG G0 executable winner-rule drift")
    if winner.get("clearance_metrics_affect_g0_pass") is not False:
        raise ValueError("VITG G0 clearance must remain diagnostic-only")
    print("VITG_G0_PROTOCOL_VALID")


if __name__ == "__main__":
    main()
