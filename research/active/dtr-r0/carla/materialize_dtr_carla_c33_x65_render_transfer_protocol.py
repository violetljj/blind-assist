"""Materialize the fresh C33 render-transfer confirmation for frozen X65."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL = HERE / "dtr_carla_c32_x64_l03_restored_protocol.json"
PARENT_PROTOCOL_SHA256 = (
    "3535EB75466A57ABD688D5DE4A2330E6CB093C58C851A8F810070605A5949E98"
)
PARENT_SOURCE_RESULT_SHA256 = (
    "A81187F51568303ECC5C0C6C5DAD2954DE3F64D458188FC4959769E08A0B7BFC"
)
COHORT_ID = "DTR_CARLA_C33_X65_RENDER_TRANSFER_CONFIRMATION_V1"
CAPTURE_SEED = 331065
WEATHERS = {
    "c8_l01": "DustStorm",
    "c8_l02": "ClearNight",
    "c8_l03": "WetSunset",
    "c8_l04": "HardRainNoon",
}
X65_FILE = "dtr_carla_x65_ancestry_synchronized_conflict_handback.py"
X65_SHA256 = "B87E444384CF6BE4A2B69A4B8536F9EA4CD10FE8A46DD9B5D0499A60AB94E4F1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(parent: dict) -> dict:
    protocol = copy.deepcopy(parent)
    protocol["schema_version"] = 33
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm byte-frozen X65 once on fresh render-domain and seed-disjoint "
        "CARLA pixels while preserving validated C32 geometry and trajectories."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    old_prereg = protocol.pop("c32_x64_preregistration")
    components = dict(old_prereg["frozen_component_sha256"])
    components[X65_FILE] = X65_SHA256
    protocol["c33_x65_preregistration"] = {
        **old_prereg,
        "schema": "dtr-carla-c33-x65-preregistration-v1",
        "baselines": ["X24", "X54", "X64", "X65"],
        "frozen_component_sha256": components,
        "inheritance_roles": {
            "X54_ROUTE_RISK_CORE": "RETAINED_CORE",
            "X59_MODALITY_EVIDENCE_RELIABILITY_ROUTER": "RETAINED_CORE",
            "X62_SYNCHRONIZED_CONFLICT_HANDBACK": "COMPONENT",
            "X64_UNANCHORED_CROSSING_RELEASE": "RETAINED_CORE",
            "X65_PRECONFLICT_CREDENTIAL_HANDOFF": "CHALLENGER",
        },
        "single_x65_scored_invocation": True,
        "primary_transfer_gate": {
            "minimum_precision": 0.85,
            "minimum_recall": 0.70,
            "minimum_f1": 0.78,
            "minimum_each_contact_recall": 0.55,
            "maximum_safe_false_alert_segments_per_episode": 4,
            "maximum_total_safe_false_alert_segments": 10,
            "minimum_x65_preconflict_credentialed_handbacks": 1,
            "minimum_x65_tp_delta_vs_x64": 1,
            "maximum_x65_fp_delta_vs_x64": 0,
            "required_zero_authority_invariants": old_prereg[
                "primary_transfer_gate"
            ]["required_zero_authority_invariants"],
        },
        "stretch_target": {
            "precision": 0.90,
            "recall": 0.75,
            "f1": 0.82,
        },
    }
    protocol["c33_x65_preregistration"].pop(
        "single_x64_scored_invocation", None
    )

    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c33-x65-render-transfer-contract-v1",
        "consumed_parent_cohort_id": parent["cohort_id"],
        "consumed_parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "consumed_parent_source_result_sha256": PARENT_SOURCE_RESULT_SHA256,
        "consumed_parent_status": "SOURCE_COMPLETE_X64_MECHANISM_NOT_EXERCISED",
        "x65_designed_after_parent_pixels_opened": True,
        "source_change": "NEW_CAPTURE_SEED_AND_RENDER_DOMAIN_ASSIGNMENT_ONLY",
        "new_capture_seed": CAPTURE_SEED,
        "changed_weather_by_layout": WEATHERS,
        "camera_geometry_unchanged_from_c32": True,
        "target_and_occluder_trajectories_unchanged_from_c32": True,
        "route_and_truth_contract_unchanged_from_c32": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "fresh_pixels": True,
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
        "C26-C28 and C32 are consumed and cannot provide fresh X65 authority.",
        "X65 and every imported component are byte-frozen before C33 capture.",
        "C33 changes only capture seed and render-domain assignment; validated C32 camera, route, trajectory, and truth geometry remain unchanged.",
        "A C33 gate confirms X65 only when the pre-conflict credentialed handback is exercised and improves TP over X64 without increasing FP.",
        "C33 is source-disjoint synthetic Development, not unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety evidence.",
        "A failed source gate is NOT_EVALUABLE; frozen X65 cannot be changed or rerun on C33 after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("frozen C32 parent protocol drift")
    parent = read_json(PARENT_PROTOCOL)
    protocol = materialize(parent)
    for file_name, expected in protocol["c33_x65_preregistration"][
        "frozen_component_sha256"
    ].items():
        path = HERE / file_name
        if not path.is_file() or c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C33 component drift: {file_name}")
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
                "status": "C33_X65_PROTOCOL_STATIC_VALID_PREREGISTERED",
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
