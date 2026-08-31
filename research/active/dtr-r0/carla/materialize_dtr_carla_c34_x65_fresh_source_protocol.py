"""Materialize a new C34 fresh-source confirmation for byte-frozen X65."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL = HERE / "dtr_carla_c33_x65_render_transfer_protocol.json"
PARENT_PROTOCOL_SHA256 = (
    "421CA0C5B5518B87E2AA55679561E99F03BCA45CE5CE6AD7255DCB1B94446EF5"
)
PARENT_SOURCE_RESULT_SHA256 = (
    "A91477D068C16CC3E5747C2565909DB8A4206F813FB9C713045B45BBC0B04087"
)
PARENT_SOURCE_STATUS = (
    "DTR_CARLA_C33_SOURCE_NOT_EVALUABLE_IN_DOUBT_PARTIAL_DEPTH"
)
COHORT_ID = "DTR_CARLA_C34_X65_FRESH_SOURCE_CONFIRMATION_V1"
CAPTURE_SEED = 341066
WEATHERS = {
    "c8_l01": "ClearSunset",
    "c8_l02": "SoftRainNight",
    "c8_l03": "WetCloudyNoon",
    "c8_l04": "MidRainSunset",
}
X65_FILE = "dtr_carla_x65_ancestry_synchronized_conflict_handback.py"
X65_SHA256 = "B87E444384CF6BE4A2B69A4B8536F9EA4CD10FE8A46DD9B5D0499A60AB94E4F1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(parent: dict) -> dict:
    protocol = copy.deepcopy(parent)
    protocol["schema_version"] = 34
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm byte-frozen X65 once on a new cohort of fresh render-domain "
        "and seed-disjoint CARLA pixels after terminal unscored C33, while "
        "preserving the validated camera, route, geometry, and trajectories."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    old_prereg = protocol.pop("c33_x65_preregistration")
    protocol["c34_x65_preregistration"] = {
        **old_prereg,
        "schema": "dtr-carla-c34-x65-preregistration-v1",
        "baselines": ["X24", "X54", "X64", "X65"],
        "single_x65_scored_invocation": True,
    }
    protocol["c34_x65_preregistration"]["frozen_component_sha256"][
        X65_FILE
    ] = X65_SHA256

    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c34-x65-fresh-source-contract-v1",
        "terminal_parent_cohort_id": parent["cohort_id"],
        "terminal_parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "terminal_parent_source_result_sha256": PARENT_SOURCE_RESULT_SHA256,
        "terminal_parent_status": PARENT_SOURCE_STATUS,
        "terminal_parent_model_predictions_created": False,
        "terminal_parent_evaluator_opened": False,
        "c34_is_new_cohort_not_c33_retry": True,
        "source_change": "NEW_CAPTURE_SEED_AND_RENDER_DOMAIN_ASSIGNMENT_ONLY",
        "new_capture_seed": CAPTURE_SEED,
        "changed_weather_by_layout": WEATHERS,
        "camera_geometry_unchanged_from_c33": True,
        "target_and_occluder_trajectories_unchanged_from_c33": True,
        "route_and_truth_contract_unchanged_from_c33": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "fresh_pixels": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
        "capture_supervision": "DETACHED_DURABLE_SUPERVISOR",
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
        "C33 is terminal unscored source evidence and cannot be retried.",
        "C34 is a new cohort with new seed, render-domain assignment, and pixels; it does not reuse C33 payloads.",
        "X65 and every imported component remain byte-frozen before C34 capture.",
        "A C34 gate confirms X65 only when the pre-conflict credentialed handback is exercised and improves TP over X64 without increasing FP.",
        "C34 is source-disjoint synthetic Development, not unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety evidence.",
        "A failed source gate is NOT_EVALUABLE; frozen X65 cannot be changed or rerun on C34 after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c33-terminal-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("frozen C33 parent protocol drift")
    terminal_result_path = args.c33_terminal_result.resolve(strict=True)
    if c2.sha256_file(terminal_result_path) != PARENT_SOURCE_RESULT_SHA256:
        raise RuntimeError("frozen C33 terminal source result drift")
    terminal_result = read_json(terminal_result_path)
    if (
        terminal_result.get("status") != PARENT_SOURCE_STATUS
        or terminal_result.get("model_predictions_created") is not False
        or terminal_result.get("evaluator_opened") is not False
    ):
        raise RuntimeError("C33 terminal boundary drift")

    parent = read_json(PARENT_PROTOCOL)
    protocol = materialize(parent)
    for file_name, expected in protocol["c34_x65_preregistration"][
        "frozen_component_sha256"
    ].items():
        path = HERE / file_name
        if not path.is_file() or c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C34 component drift: {file_name}")
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
                "status": "C34_X65_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "x65_sha256": X65_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
