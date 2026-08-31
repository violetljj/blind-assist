"""Materialize fresh sensor/topology C29 confirmation for frozen X64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c28_x59_mixed_lighting_protocol as c28


COHORT_ID = "DTR_CARLA_C29_X64_SENSOR_TOPOLOGY_CONFIRMATION_V1"
CAPTURE_SEED = 291064
FOV_DEGREES = 100.0
WEATHERS = {
    "c8_l01": "DustStorm",
    "c8_l02": "ClearNoon",
    "c8_l03": "WetNight",
    "c8_l04": "MidRainyNoon",
}
C28_PROTOCOL_SHA256 = (
    "71C3C097301BAC9D23CA2AE161D1451BDE6D2639BBC325C5A333DC400B0F1E99"
)
C28_SOURCE_RESULT_SHA256 = (
    "6BFB82EC7112F3D5E83F7FA2740DF1D6BF2A313CBF1BD576BB7FDA046B63BF9E"
)
C26_X64_SUMMARY_SHA256 = (
    "2FBF9F1DB7155A7F0F97ADDC90868DA058205E7A85FE5626247ECC4A06A24CC5"
)
C27_X64_SUMMARY_SHA256 = (
    "4A2BC90267911BF471E3A2EC73D0A9FAA1683AB0E5E4C54E21E874C8D7BBA323"
)
C28_X64_SUMMARY_SHA256 = (
    "D85A710AA087DCDC19D23B3262F9E0E36F5582EA0755BBE5183E34DD7DE88973"
)
X64_SHA256 = "4A1B34C3CECF3635324DB909520AA4BE13578566FD3B9EDA28B8BE60364FE3DE"
COMPONENT_SHA256 = {
    **c28.COMPONENT_SHA256,
    "dtr_carla_x60_route_entry_credential_memory.py": (
        "D8AF22F33654C37D9DE41F8F12673F9E6622FF5FCC4B9F7E65DDC64997AEA091"
    ),
    "dtr_carla_x61_conflict_localized_route_entry.py": (
        "C03ED3BF976E61BD29920CA8320E2B43E2D9041145F73DC3FEF98846E77B3B73"
    ),
    "dtr_carla_x62_synchronized_conflict_handback.py": (
        "DAA8B5120DE7D89ABCBA17E1D0CB89867ED85714DC957DDDF8E1BF61BD2B12BC"
    ),
    "dtr_carla_x63_existence_only_object_permanence.py": (
        "FB872C14588C7514CD78D43B64EE69D79B16669AFFB9763212FE91212A637F59"
    ),
    "dtr_carla_x64_unanchored_crossing_release.py": X64_SHA256,
}
CAPTURE_SCRIPT_SHA256 = c28.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c28.JOIN_SCRIPT_SHA256


def _set_motion_topology(protocol: dict) -> dict[str, str]:
    topology: dict[str, str] = {}
    for layout_id in ("c8_l01", "c8_l02"):
        for suffix in ("target_contact", "target_safe"):
            key = f"{layout_id}_{suffix}"
            for segment in protocol["trajectory_library"][key]["segments"]:
                segment["velocity_right_mps"] = 0.0
            topology[key] = "STRICT_LONGITUDINAL_CORRIDOR"
    for layout_id in ("c8_l03", "c8_l04"):
        for suffix in ("target_contact", "target_safe"):
            key = f"{layout_id}_{suffix}"
            for segment in protocol["trajectory_library"][key]["segments"]:
                segment["velocity_right_mps"] = (
                    1.5 * float(segment["velocity_right_mps"])
                )
            topology[key] = "AMPLIFIED_CROSS_ROUTE_COMPONENT"
    return topology


def materialize(base: dict) -> dict:
    protocol = c28.materialize(base)
    protocol["schema_version"] = 29
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["capture"]["fov_degrees"] = FOV_DEGREES
    intrinsics = c2.camera_intrinsics(1280, 720, FOV_DEGREES)
    protocol["capture"]["camera_calibration"]["focal_length_pixels"] = [
        intrinsics[0][0],
        intrinsics[1][1],
    ]
    protocol["capture"]["wearable_relative_transform"].update(
        {"pitch_degrees": -8.0, "x_m": 0.10, "z_m": 0.62}
    )
    protocol["objective"] = (
        "Confirm frozen X64 once on fresh camera geometry, unseen render-domain "
        "combinations, and explicit longitudinal versus cross-route target motion."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather
    motion_topology = _set_motion_topology(protocol)

    old_prereg = protocol.pop("c28_x59_preregistration")
    gates = old_prereg["primary_transfer_gate"]
    protocol["c29_x64_preregistration"] = {
        "schema": "dtr-carla-c29-x64-preregistration-v1",
        "baselines": ["X24", "X54", "X64"],
        "frozen_capture_script_sha256": CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": JOIN_SCRIPT_SHA256,
        "frozen_component_sha256": COMPONENT_SHA256,
        "single_x64_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
        "score_window_seconds": old_prereg["score_window_seconds"],
        "outcomes": [
            "GATE_MET",
            "GATE_NOT_MET",
            "SOURCE_NOT_EVALUABLE",
            "MECHANISM_NOT_EXERCISED",
        ],
        "primary_transfer_gate": {
            "minimum_precision": 0.85,
            "minimum_recall": 0.70,
            "minimum_f1": 0.78,
            "minimum_each_contact_recall": 0.55,
            "maximum_safe_false_alert_segments_per_episode": gates[
                "maximum_safe_false_alert_segments_per_episode"
            ],
            "maximum_total_safe_false_alert_segments": gates[
                "maximum_total_safe_false_alert_segments"
            ],
            "minimum_x62_synchronized_conflict_handbacks": 1,
            "minimum_x64_unanchored_crossing_releases": 1,
            "required_zero_authority_invariants": gates[
                "required_zero_authority_invariants"
            ],
        },
        "stretch_target": old_prereg["stretch_target"],
        "inheritance_roles": {
            "X54_ROUTE_RISK_CORE": "RETAINED_CORE",
            "X59_MODALITY_EVIDENCE_RELIABILITY_ROUTER": "RETAINED_CORE",
            "X62_SYNCHRONIZED_CONFLICT_HANDBACK": "COMPONENT",
            "X64_UNANCHORED_CROSSING_RELEASE": "CHALLENGER",
        },
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c29-x64-sensor-topology-contract-v1",
        "development_parent_cohort_id": c28.COHORT_ID,
        "development_parent_protocol_sha256": C28_PROTOCOL_SHA256,
        "development_parent_source_result_sha256": C28_SOURCE_RESULT_SHA256,
        "development_parent_c26_x64_summary_sha256": C26_X64_SUMMARY_SHA256,
        "development_parent_c27_x64_summary_sha256": C27_X64_SUMMARY_SHA256,
        "development_parent_c28_x64_summary_sha256": C28_X64_SUMMARY_SHA256,
        "development_parent_status": "THREE_CONSUMED_COHORTS_DEVELOPMENT_ONLY",
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "changed_camera_fov_degrees": FOV_DEGREES,
        "changed_wearable_relative_transform": protocol["capture"][
            "wearable_relative_transform"
        ],
        "changed_target_motion_topology": motion_topology,
        "preserved_contact_safe_twin_offsets_and_route_plan": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
        "capture_retry_policy": (
            "ONE_FAILED_SERVER_SHARD_WITH_ZERO_DURABLE_FRAMES_MAY_BE_RETRIED_"
            "ONCE_AFTER_ATTEMPT_LOGS_ARE_PRESERVED"
        ),
        "capture_in_doubt_policy": (
            "ANY_NONZERO_PARTIAL_SHARD_IS_SOURCE_NOT_EVALUABLE_AND_MUST_NOT_RETRY"
        ),
        "completed_shard_rerun_allowed": False,
    }
    protocol["claim_boundary"] = [
        "C29 is frozen before capture and reuses no C20-C28 pixels.",
        "C29 changes seed, camera geometry, render domains, and target motion topology while preserving contact-safe twin offsets and the route plan.",
        "X64 and every imported algorithm component are byte-frozen before capture and consume no weather or lighting label.",
        "A C29 gate confirms X64 only as source-disjoint synthetic Development when both synchronized conflict handback and unanchored crossing release are exercised.",
        "C29 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed source gate is NOT_EVALUABLE; frozen X64 cannot be changed or rerun on C29 after pixels are opened.",
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
        here / "dtr_carla_c28_x59_mixed_lighting_protocol.json": (
            C28_PROTOCOL_SHA256
        ),
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C29 component drift: {path.name}")
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
                "status": "C29_X64_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "fov_degrees": FOV_DEGREES,
                "x64_sha256": X64_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
