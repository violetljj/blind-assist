"""One non-reusable visual-shell probe; no FIT_ONLY or FINAL pixels or methods."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import materialize_dtr_final_reckoning_source_probe as base
import validate_dtr_final_reckoning_roster as roster_validator


PROBE_SEED = 516938


def materialize() -> dict:
    roster = base.read_json(base.ROSTER_PROTOCOL)
    repo = base.HERE.parents[3]
    roster_validator.validate(roster, repo_root=repo)
    if base.c2.sha256_file(base.BASE_PROTOCOL) != base.BASE_PROTOCOL_SHA256:
        raise RuntimeError("base_protocol_hash_drift")
    protocol = base.materialize(base.read_json(base.BASE_PROTOCOL), roster)
    protocol["cohort_id"] = "DTR_FINAL_RECKONING_R1_VISUAL_SHELL_PROBE_ONLY"
    protocol["capture"]["seed"] = PROBE_SEED
    protocol["objective"] = "One bounded non-collision visual shell probe with paired unobscured raster references. No method or fit/final use."
    protocol["asset_templates"]["fr_partial_visual_shell"] = {
        "blueprint_candidates": ["static.prop.advertisement"],
        "collision_relevant": False, "kind": "prop", "surface_offset_m": 0.55,
    }
    protocol["asset_templates"]["fr_full_visual_shell"] = {
        "blueprint_candidates": ["static.prop.advertisement"],
        "collision_relevant": False, "kind": "prop", "surface_offset_m": -0.40,
    }
    for asset in protocol["layouts"]["c8_l01"]["assets"]:
        if asset["asset_key"] in ("c8_l01_c16_shell_01", "c8_l01_c16_shell_02"):
            asset["template"] = "fr_partial_visual_shell" if asset["asset_key"].endswith("01") else "fr_full_visual_shell"
            asset["collisions_enabled"] = False
            asset["role"] = "probe_only_noncollision_visual_occluder"
    library = protocol["trajectory_library"]
    # Panel long axis is local Y: zero yaw presents the face to the camera.
    # Partial panel's raised bottom leaves the lower target surface visible.
    library["fr_visual_partial"] = base.trajectory(
        1.4, 8.0, [(0.0, 0.0, 0.0), (1.0, 0.0, -40.0),
                   (1.2, 0.0, 0.0), (2.5, 0.0, 40.0), (2.7, 0.0, 0.0)])
    # At t=5..5.6 the full panel follows the midpoint of wearer and target.
    # The earlier t=2 interval is outside the unobscured camera field of view;
    # it cannot establish physical occlusion or eight visible pretrack samples.
    # Translation into/out of that ray is fixed before any new raster access.
    library["fr_visual_full"] = base.trajectory(
        -0.15, 7.4, [(0.0, 0.0, 0.0), (4.8, 1.0 / 3.0, -39.5),
                    (5.0, 1.0 / 3.0, 0.5),
                    (5.7, 0.0, 40.0), (5.9, 0.0, 0.0)])
    for name in ("fr_visual_partial", "fr_visual_full"):
        library[name]["yaw_offset_degrees"] = 0.0
    partial, full = protocol["scenarios"][8:10]
    for cell in (partial, full):
        cell["asset_trajectories"]["c8_l01_alias"] = "fr_far_static"
        cell["asset_trajectories"]["c8_l01_occluder"] = "fr_far_static"
    partial["asset_trajectories"]["c8_l01_c16_shell_01"] = "fr_visual_partial"
    full["asset_trajectories"]["c8_l01_c16_shell_02"] = "fr_visual_full"
    for cell, reference_id, shell in (
        (partial, "ep_11", "c8_l01_c16_shell_01"),
        (full, "ep_12", "c8_l01_c16_shell_02"),
    ):
        reference = copy.deepcopy(cell)
        reference["episode_id"] = reference_id
        reference["scenario_role"] = "UNOBSCURED_RASTER_REFERENCE_ONLY"
        reference["twin_role"] = "probe_only_unobscured_reference"
        reference["asset_trajectories"][shell] = "fr_far_static"
        protocol["scenarios"].append(reference)
    protocol["admission"]["expected_episode_count"] = 12
    # Keep the shared capture schema; the paired-raster evaluator owns admission.
    protocol["occlusion_contracts"][0]["occluder_asset"] = "c8_l01_c16_shell_01"
    protocol["occlusion_contracts"][1]["occluder_asset"] = "c8_l01_c16_shell_02"
    protocol["occlusion_contracts"][1]["planned_occlusion_window_s"] = [5.0, 5.6]
    protocol["final_visual_shell_probe"] = {
        "schema": "blindassist-dtr-final-visual-shell-probe-v1",
        "maximum_capture_invocations": 1,
        "zero_durable_frame_retry_limit": 1,
        "raster_reference": "same-time target silhouette in separately captured unobscured twin",
        "pairs": [
            {"episode_id": "ep_09", "reference_episode_id": "ep_11", "target_asset": "c8_l01_target", "shell_asset": "c8_l01_c16_shell_01", "role": "PARTIAL", "window_s": [1.3, 2.4]},
            {"episode_id": "ep_10", "reference_episode_id": "ep_12", "target_asset": "c8_l01_target", "shell_asset": "c8_l01_c16_shell_02", "role": "FULL", "window_s": [5.0, 5.6]},
        ],
        "probe_pixels_reusable_as_fit_or_final": False,
        "method_predictions_or_scores_allowed": False,
        "witness_required": True,
        "positive_gate_is_not_full_roster_admission": True,
    }
    protocol["final_reckoning_source_probe"]["probe_seed"] = PROBE_SEED
    protocol["source_disjoint_contract"]["probe_seed_disjoint_from_fit_and_final"] = PROBE_SEED not in {
        row["capture_seed"] for row in roster["source_design"]["seed_groups"]}
    evaluation = protocol["evaluation_contract"]
    for key in ("contact_episodes", "fresh_dynamic_contact_episodes"):
        evaluation[key] += ["ep_11", "ep_12"]
    evaluation["score_window_end_seconds"].update({"ep_11": 6.0, "ep_12": 6.0})
    protocol["claim_boundary"] += [
        "The paired reference raster defines visibility only, never an observation for a method.",
        "The shell is an artificial non-collision visual intervention, not a physical navigable scene claim.",
        "A passing shell probe still requires remaining full-roster gates and execution-contract freeze.",
    ]
    base.c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = materialize()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "VISUAL_SHELL_PROBE_FROZEN_NO_PIXELS", "path": str(output),
                      "sha256": base.c2.sha256_file(output), "episodes": 12, "seed": PROBE_SEED}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
