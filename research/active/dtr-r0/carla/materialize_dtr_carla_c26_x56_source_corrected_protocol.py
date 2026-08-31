"""Materialize source-corrected fresh C26 confirmation for frozen X56."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c24_x54_source_corrected_generalization_protocol as c24
import materialize_dtr_carla_c25_x56_graceful_degradation_protocol as c25


COHORT_ID = "DTR_CARLA_C26_X56_SOURCE_CORRECTED_CONFIRMATION_V1"
CAPTURE_SEED = 260976
C25_PROTOCOL_SHA256 = (
    "F8697D66CB789C92886456C3D4391DD8EF5354AF298ADBF29269C340B7DAAA0F"
)
C25_SOURCE_RESULT_SHA256 = (
    "BA5D74DF4222FD3F010D3F59F73F17CD23626D563D9E5D436BDA7CADCD2EF0DD"
)
C25_OCCLUSION_REPORT_SHA256 = (
    "7CE3D2AF7EBBE283BC9D2F45C82EA1811FE97C972AC4E642DF18BF335E584192"
)
WEATHERS = {
    "c8_l01": "CloudyNight",
    "c8_l02": "WetNight",
    "c8_l03": "SoftRainNight",
    "c8_l04": "HardRainNight",
}
COMPONENT_SHA256 = dict(c25.COMPONENT_SHA256)
CAPTURE_SCRIPT_SHA256 = c25.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c25.JOIN_SCRIPT_SHA256


def materialize(base: dict) -> dict:
    protocol = c25.materialize(base)
    c24_reference = c24.materialize(base)
    protocol["schema_version"] = 26
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm frozen X56 once on fresh night/adverse-weather pixels after "
        "repairing only the C25 source-invalid l03/l04 occlusion profiles."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    restored_layouts = ["c8_l03", "c8_l04"]
    for layout_id in restored_layouts:
        for twin_role in ("contact", "safe"):
            key = f"{layout_id}_target_{twin_role}"
            protocol["trajectory_library"][key]["segments"] = (
                c24_reference["trajectory_library"][key]["segments"]
            )

    prereg = protocol.pop("c25_x56_preregistration")
    prereg["schema"] = "dtr-carla-c26-x56-preregistration-v1"
    protocol["c26_x56_preregistration"] = prereg
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c26-x56-source-corrected-contract-v1",
        "failed_parent_cohort_id": c25.COHORT_ID,
        "failed_parent_protocol_sha256": C25_PROTOCOL_SHA256,
        "failed_parent_source_result_sha256": C25_SOURCE_RESULT_SHA256,
        "failed_parent_occlusion_report_sha256": C25_OCCLUSION_REPORT_SHA256,
        "failed_parent_status": "SOURCE_NOT_EVALUABLE_NO_ALGORITHM_SCORE",
        "failed_parent_checks": {
            "ep_05_complete_occlusion_seconds": 0.5,
            "ep_05_required_minimum_seconds": 0.6,
            "ep_08_pre_track_frames": 0,
            "ep_08_required_minimum_pre_track_frames": 10,
        },
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "retained_c25_l01_mirrored_lateral_motion": True,
        "restored_c24_source_valid_motion_layouts": restored_layouts,
        "l04_weather_correction": "DustStorm_TO_HardRainNight",
        "source_corrections_inherited_from_c24": True,
        "preserved_route_occlusion_blueprint_and_lateral_twin_geometry": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
    }
    protocol["claim_boundary"] = [
        "C26 is frozen before capture and reuses no C22-C25 pixels.",
        "C26 retains C25 night/adverse-weather intent but replaces DustStorm with HardRainNight and restores only the l03/l04 source-valid C24 target motions after C25 failed its source gate.",
        "X55 and X56 remain byte-identical to their pre-C25 frozen versions; C25 never ran the algorithm.",
        "A C26 gate confirms X56 only as source-disjoint synthetic Development and only when X53-X56 successor mechanisms are exercised.",
        "C26 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed source gate is NOT_EVALUABLE; frozen X56 cannot be changed or rerun on C26 after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    frozen_files = {
        here / "capture_dtr_carla_c2_rich_scene.py": CAPTURE_SCRIPT_SHA256,
        here / "join_dtr_carla_c2_rich_scene.py": JOIN_SCRIPT_SHA256,
        here / "dtr_carla_c25_x56_graceful_degradation_protocol.json": (
            C25_PROTOCOL_SHA256
        ),
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C26 component drift: {path.name}")
    protocol = materialize(c17._read_json(c17.BASE_PROTOCOL))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "C26_X56_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "x56_sha256": c25.X56_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
