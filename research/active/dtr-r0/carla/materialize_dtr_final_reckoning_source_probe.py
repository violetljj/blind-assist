"""Materialize the non-reusable 10-cell source probe for Final Reckoning R1."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
BASE_PROTOCOL = HERE / "dtr_carla_c44_x96_dropout_survival_protocol.json"
BASE_PROTOCOL_SHA256 = "7C940C55362585D553BF86E198BEA42B641F29AA0AD730B99E4E7479EF881555"
ROSTER_PROTOCOL = HERE / "dtr_final_reckoning_roster_protocol.json"
PROBE_SEED = 516936
PROBE_COHORT_ID = "DTR_FINAL_RECKONING_R1_SOURCE_PROBE_ONLY"
SCORE_END_S = 6.0
CAPTURE_END_S = 9.0


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not_object:{path}")
    return value


def trajectory(
    start_forward_m: float,
    start_right_m: float,
    segments: list[tuple[float, float, float]],
) -> dict[str, Any]:
    return {
        "start_forward_m": start_forward_m,
        "start_right_m": start_right_m,
        "segments": [
            {
                "start_s": start,
                "velocity_forward_mps": forward,
                "velocity_right_mps": right,
            }
            for start, forward, right in segments
        ],
    }


def plan_for(name: str, value: dict[str, Any], episode_number: int) -> dict[str, Any]:
    times = sorted(
        {0.0, CAPTURE_END_S}
        | {float(row["start_s"]) for row in value["segments"]}
    )
    session_id = f"dtr_fr_probe_session_{episode_number:02d}"
    return {
        "plan_id": f"dtr_fr_probe_plan_{episode_number:02d}_{name}",
        "session_id": session_id,
        "issued_at_s": 0.0,
        "expires_at_s": CAPTURE_END_S,
        "time_parameterized_waypoints": [
            {
                "time_s": time_s,
                "forward_m": c2.trajectory_position(value, time_s)[0],
                "right_m": c2.trajectory_position(value, time_s)[1],
            }
            for time_s in times
        ],
    }


def _cell(
    *,
    episode_number: int,
    stratum_id: str,
    layout_id: str,
    wearer: str,
    target: str,
    alias: str = "fr_far_static",
    occluder: str = "fr_far_static",
    expected_outcome: str,
) -> dict[str, Any]:
    suffix = layout_id.split("_")[-1]
    target_key = f"c8_{suffix}_target"
    alias_key = f"c8_{suffix}_alias"
    occluder_key = f"c8_{suffix}_occluder"
    return {
        "episode_id": f"ep_{episode_number:02d}",
        "layout_id": layout_id,
        "scenario_role": stratum_id,
        "twin_role": "probe_only_unpaired",
        "final_reckoning_stratum_id": stratum_id,
        "wearer_trajectory": wearer,
        "asset_trajectories": {
            target_key: target,
            alias_key: alias,
            occluder_key: occluder,
        },
        "expected_outcome": expected_outcome,
        "expected_responsible_assets": [target_key] if expected_outcome != "SAFE" else [],
        "navigation_session_id": f"dtr_fr_probe_session_{episode_number:02d}",
    }


def materialize(base: dict[str, Any], roster: dict[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(base)
    protocol["schema_version"] = 51
    protocol["cohort_id"] = PROBE_COHORT_ID
    protocol["objective"] = (
        "Probe only whether the ten frozen Final Reckoning R1 source cells are "
        "physically and visually materializable. These pixels may never enter fit or final scoring."
    )
    protocol["capture"]["seed"] = PROBE_SEED
    protocol["admission"]["expected_episode_count"] = 10
    protocol["twin_contracts"] = []

    library = protocol["trajectory_library"]
    library.update(
        {
            "fr_far_static": trajectory(20.0, 15.0, [(0.0, 0.0, 0.0)]),
            "fr_wearer_straight": trajectory(-6.0, 0.0, [(0.0, 2.0, 0.0)]),
            "fr_target_static_contact": trajectory(2.0, 0.0, [(0.0, 0.0, 0.0)]),
            "fr_target_lateral_contact": trajectory(0.0, 3.0, [(0.0, 0.0, -1.0), (3.2, 0.0, 0.0)]),
            "fr_target_receding_near_miss": trajectory(-6.0, 1.0, [(0.0, 3.0, 0.0)]),
            "fr_alias_near_cross": trajectory(2.0, 1.5, [(0.0, 0.0, -0.2), (3.0, 0.0, 0.2)]),
            "fr_wearer_curved": trajectory(
                -6.0, -2.0, [(0.0, 2.0, 0.0), (2.0, 2.0, 1.0), (4.0, 2.0, 0.0)]
            ),
            "fr_wearer_turn_safe": trajectory(
                -4.0, -2.0, [(0.0, 2.0, 0.0), (2.0, 0.0, 1.0), (4.0, 2.0, 0.0)]
            ),
            "fr_static_rotation_target": trajectory(0.0, 4.0, [(0.0, 0.0, 0.0)]),
            "fr_partial_alias": trajectory(1.4, 0.18, [(0.0, 0.0, 0.0)]),
            "fr_wearer_two_contact": trajectory(
                -2.0, 0.0, [(0.0, 2.0, 0.0), (1.0, -1.0, 0.0), (3.0, 2.0 / 3.0, 0.0), (6.0, 0.0, 0.0)]
            ),
            "fr_target_two_crossings": trajectory(
                0.5, 2.0, [(0.0, 0.0, -2.0), (2.0, 0.0, 0.0), (4.0, 0.0, 1.0), (6.2, 0.0, 0.0)]
            ),
            "fr_full_occluder": trajectory(
                -1.0, 8.0, [(0.0, 0.0, -4.0), (2.0, 0.0, 0.0), (4.0, 0.0, -4.0), (6.0, 0.0, 0.0)]
            ),
        }
    )

    cells = [
        _cell(episode_number=1, stratum_id="S01_CLEAN_CONSTANT_MOTION", layout_id="c8_l01", wearer="fr_wearer_straight", target="fr_target_static_contact", expected_outcome="CONTACT"),
        _cell(episode_number=2, stratum_id="S02_SINGLE_FRAME_DROPOUT", layout_id="c8_l02", wearer="fr_wearer_straight", target="fr_target_static_contact", expected_outcome="CONTACT"),
        _cell(episode_number=3, stratum_id="S03_MULTI_FRAME_DROPOUT", layout_id="c8_l03", wearer="fr_wearer_straight", target="fr_target_static_contact", expected_outcome="CONTACT"),
        _cell(episode_number=4, stratum_id="S04_LATERAL_CROSSING", layout_id="c8_l04", wearer="fr_wearer_straight", target="fr_target_lateral_contact", expected_outcome="CONTACT"),
        _cell(episode_number=5, stratum_id="S05_RECEDING_NEAR_MISS", layout_id="c8_l01", wearer="fr_wearer_straight", target="fr_target_receding_near_miss", expected_outcome="SAFE"),
        _cell(episode_number=6, stratum_id="S06_FRAGMENTATION_ID_INSTABILITY", layout_id="c8_l01", wearer="fr_wearer_straight", target="fr_target_static_contact", alias="fr_alias_near_cross", expected_outcome="CONTACT"),
        _cell(episode_number=7, stratum_id="S07_CURVED_WEARER_ROUTE", layout_id="c8_l03", wearer="fr_wearer_curved", target="fr_target_static_contact", expected_outcome="CONTACT"),
        _cell(episode_number=8, stratum_id="S08_STATIC_PSEUDO_MOTION_EGO_ROTATION", layout_id="c8_l04", wearer="fr_wearer_turn_safe", target="fr_static_rotation_target", expected_outcome="SAFE"),
        _cell(episode_number=9, stratum_id="S09_PARTIAL_VISIBILITY_SURFACE_FRAGMENTATION", layout_id="c8_l01", wearer="fr_wearer_straight", target="fr_target_static_contact", alias="fr_partial_alias", expected_outcome="CONTACT"),
        _cell(episode_number=10, stratum_id="S10_DISAPPEAR_REAPPEAR_CLEAR_REONSET", layout_id="c8_l01", wearer="fr_wearer_two_contact", target="fr_target_two_crossings", expected_outcome="CONTACT"),
    ]
    library["fr_wearer_turn_safe"]["yaw_segments"] = [
        {"start_s": 0.0, "yaw_rate_degrees_per_second": 0.0},
        {"start_s": 2.0, "yaw_rate_degrees_per_second": 30.0},
        {"start_s": 4.0, "yaw_rate_degrees_per_second": 0.0},
    ]
    for cell in cells:
        layout_suffix = cell["layout_id"].split("_")[-1]
        for shell_number in range(1, 7):
            cell["asset_trajectories"][
                f"c8_{layout_suffix}_c16_shell_{shell_number:02d}"
            ] = "fr_far_static"
    cells[-1]["asset_trajectories"]["c8_l01_occluder"] = "fr_full_occluder"
    for cell in cells:
        cell["issued_plan"] = plan_for(
            cell["wearer_trajectory"],
            library[cell["wearer_trajectory"]],
            int(cell["episode_id"].split("_")[-1]),
        )
    protocol["scenarios"] = cells

    contact = [row["episode_id"] for row in cells if row["expected_outcome"] == "CONTACT"]
    safe = [row["episode_id"] for row in cells if row["expected_outcome"] == "SAFE"]
    protocol["evaluation_contract"].update(
        {
            "contact_episodes": contact,
            "fresh_dynamic_contact_episodes": contact,
            "fresh_dynamic_safe_episodes": safe,
            "safe_episodes": safe,
            "safe_segment_start_seconds": {episode_id: 0.0 for episode_id in safe},
            "score_window_end_seconds": {row["episode_id"]: SCORE_END_S for row in cells},
            "dynamic_pairs": [],
        }
    )
    protocol["occlusion_contracts"] = [
        {
            "contract_id": "fr_probe_partial_visibility_ep_09",
            "episodes": ["ep_09"],
            "target_asset": "c8_l01_target",
            "occluder_asset": "c8_l01_occluder",
            "alias_asset": "c8_l01_alias",
            "planned_occlusion_window_s": [1.3, 2.4],
            "minimum_pre_track_frames": 10,
            "minimum_post_reappearance_frames": 8,
            "minimum_trackable_pixel_fraction": 0.0002,
            "complete_occlusion_pixel_fraction": 0.0,
            "minimum_complete_occlusion_seconds": 0.0,
            "maximum_complete_occlusion_seconds": 0.0,
            "required_outcomes": {"ep_09": "CONTACT"},
        },
        {
            "contract_id": "fr_probe_full_disappear_ep_10",
            "episodes": ["ep_10"],
            "target_asset": "c8_l01_target",
            "occluder_asset": "c8_l01_c16_shell_01",
            "alias_asset": "c8_l01_alias",
            "planned_occlusion_window_s": [2.0, 4.0],
            "minimum_pre_track_frames": 8,
            "minimum_post_reappearance_frames": 8,
            "minimum_trackable_pixel_fraction": 0.0002,
            "complete_occlusion_pixel_fraction": 0.0,
            "minimum_complete_occlusion_seconds": 0.6,
            "maximum_complete_occlusion_seconds": 2.2,
            "required_outcomes": {"ep_10": "CONTACT"},
        },
    ]
    roster_strata = {row["stratum_id"]: row for row in roster["source_design"]["strata"]}
    protocol["final_reckoning_source_probe"] = {
        "schema": "blindassist-dtr-final-reckoning-source-probe-v1",
        "roster_protocol_sha256": c2.sha256_file(ROSTER_PROTOCOL),
        "probe_seed": PROBE_SEED,
        "probe_pixels_reusable_as_fit_or_final": False,
        "method_predictions_or_scores_allowed": False,
        "cells": [
            {
                "episode_id": row["episode_id"],
                "stratum_id": row["final_reckoning_stratum_id"],
                "source_gate": roster_strata[row["final_reckoning_stratum_id"]]["source_gate"],
            }
            for row in cells
        ],
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-final-reckoning-source-probe-contract-v1",
        "probe_only": True,
        "fresh_pixels": True,
        "probe_seed_disjoint_from_fit_and_final": PROBE_SEED
        not in {row["capture_seed"] for row in roster["source_design"]["seed_groups"]},
        "probe_pixels_reusable": False,
        "capture_retry_policy": "ZERO_DURABLE_FRAMES_ONLY_ONCE",
        "nonzero_partial_shard_disposition": "PROBE_FAILED_NO_REUSE",
    }
    protocol["claim_boundary"] = [
        "This source is a geometry and visibility probe only.",
        "Probe pixels and truth cannot be reused for FIT_ONLY, FINAL_A, or FINAL_B.",
        "No DTR arm may be predicted or scored on this source.",
        "A failed cell may inform one final pre-capture geometry correction without creating method evidence.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def analytic_receipt(protocol: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scenario in protocol["scenarios"]:
        wearer = protocol["trajectory_library"][scenario["wearer_trajectory"]]
        target_key = next(key for key in scenario["asset_trajectories"] if key.endswith("_target"))
        target = protocol["trajectory_library"][scenario["asset_trajectories"][target_key]]
        distances = [
            math.dist(c2.trajectory_position(wearer, step / 10.0), c2.trajectory_position(target, step / 10.0))
            for step in range(int(SCORE_END_S * 10) + 1)
        ]
        inside = [distance <= 0.65 + 1e-9 for distance in distances]
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(inside + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index - 1))
                start = None
        row = (
            {
                "episode_id": scenario["episode_id"],
                "stratum_id": scenario["final_reckoning_stratum_id"],
                "minimum_center_distance_m": min(distances),
                "minimum_center_distance_time_s": distances.index(min(distances)) / 10.0,
                "contact_like_center_runs": [
                    {"start_s": start_index / 10.0, "end_s": end_index / 10.0}
                    for start_index, end_index in runs
                ],
            }
        )
        rows.append(row)
    by_stratum = {row["stratum_id"]: row for row in rows}
    for scenario in protocol["scenarios"]:
        row = by_stratum[scenario["final_reckoning_stratum_id"]]
        if scenario["expected_outcome"] == "SAFE":
            if row["contact_like_center_runs"]:
                raise RuntimeError(f"analytic_safe_contact:{scenario['episode_id']}")
        elif not row["contact_like_center_runs"]:
            raise RuntimeError(f"analytic_contact_missing:{scenario['episode_id']}")
    near_miss = by_stratum["S05_RECEDING_NEAR_MISS"]["minimum_center_distance_m"]
    if not 0.65 < near_miss <= 1.30:
        raise RuntimeError("analytic_near_miss_distance")
    two_contact = by_stratum["S10_DISAPPEAR_REAPPEAR_CLEAR_REONSET"][
        "contact_like_center_runs"
    ]
    if len(two_contact) != 2 or two_contact[1]["start_s"] - two_contact[0]["end_s"] < 0.80:
        raise RuntimeError("analytic_two_contact_windows")
    return {
        "status": "STATIC_TRAJECTORY_PROBE_PROTOCOL_VALID_PENDING_INSTANCE_WITNESS_PIXELS",
        "cohort_id": protocol["cohort_id"],
        "episode_count": len(rows),
        "cells": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(BASE_PROTOCOL) != BASE_PROTOCOL_SHA256:
        raise RuntimeError("base_protocol_hash_drift")
    roster = read_json(ROSTER_PROTOCOL)
    protocol = materialize(read_json(BASE_PROTOCOL), roster)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = analytic_receipt(protocol)
    receipt["output"] = str(output)
    receipt["protocol_sha256"] = c2.sha256_file(output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
