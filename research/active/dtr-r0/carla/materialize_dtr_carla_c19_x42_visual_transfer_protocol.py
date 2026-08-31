"""Materialize the minimal C19 visual-domain transfer for frozen X42."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17


COHORT_ID = "DTR_CARLA_C19_X42_VISUAL_TRANSFER_FALSIFIER_V1"
CAPTURE_SEED = 190927
MATCHED_TARGET_TEMPLATES = {
    "c8_l01": "w22",
    "c8_l02": "v_model3",
    "c8_l03": "v_sprinter",
    "c8_l04": "v_police",
}
C18_PROTOCOL_SHA256 = (
    "DA2EEB82BFE2E082864D10BAE919DAF9D40A79BB45A8CF13FD922F0A5775F26E"
)
C18_INSTANCE_RESULT_SHA256 = (
    "96FE8FC12BC924DB01DBFDF406FC3080773AA41433B8A72E28FBCD398312F624"
)


def materialize(base: dict) -> dict:
    protocol = c17.materialize(base)
    protocol["schema_version"] = 19
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Falsify frozen X42 once under a source-disjoint visual domain shift: "
        "new capture identity, weather, and category/size-matched but exact-"
        "blueprint-disjoint targets, while preserving the already admitted "
        "motion, route, occlusion, and near-miss geometry."
    )

    base_scenarios = {value["episode_id"]: value for value in base["scenarios"]}
    for scenario in protocol["scenarios"]:
        parent = base_scenarios[scenario["episode_id"]]
        scenario["wearer_trajectory"] = parent["wearer_trajectory"]
        scenario["asset_trajectories"] = copy.deepcopy(parent["asset_trajectories"])
        scenario["issued_plan"]["time_parameterized_waypoints"] = copy.deepcopy(
            parent["issued_plan"]["time_parameterized_waypoints"]
        )

    base_contracts = {
        value["episodes"][0]: value for value in base["occlusion_contracts"]
    }
    for contract in protocol["occlusion_contracts"]:
        parent = base_contracts[contract["episodes"][0]]
        contract["contract_id"] = str(parent["contract_id"]).replace(
            "c8_transport_cone", "c19_visual_transfer"
        )
        contract["planned_occlusion_window_s"] = copy.deepcopy(
            parent["planned_occlusion_window_s"]
        )

    for layout_id, layout in protocol["layouts"].items():
        parent = base["layouts"][layout_id]
        layout["anchor"] = copy.deepcopy(parent["anchor"])
        layout["witness"] = copy.deepcopy(parent["witness"])
        template = MATCHED_TARGET_TEMPLATES[layout_id]
        for asset in layout["assets"]:
            key = str(asset["asset_key"])
            if "target" in key or "alias" in key:
                asset["template"] = template

    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c19-x42-visual-transfer-contract-v1",
        "parent_development_cohort_id": base.get("cohort_id"),
        "parent_protocol_sha256": c17.BASE_PROTOCOL_SHA256,
        "new_capture_seed": CAPTURE_SEED,
        "changed_target_blueprints": MATCHED_TARGET_TEMPLATES,
        "changed_weather_by_layout": c17.WEATHERS,
        "preserved_admitted_motion_route_occlusion_geometry": True,
        "probe_or_formal_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
        "c17_c18_design_evidence_only": {
            "c18_protocol_sha256": C18_PROTOCOL_SHA256,
            "c18_instance_result_sha256": C18_INSTANCE_RESULT_SHA256,
            "status": "C17_AND_C18_SOURCE_NOT_EVALUABLE_1_OF_8",
        },
    }
    protocol["claim_boundary"] = [
        "C19 is a fresh visual-domain capture cohort; no C16, C17, or C18 pixels are reused.",
        "C19 tests visual transfer only. Motion, route, occlusion, and near-miss geometry are deliberately inherited from the already admitted C16 source.",
        "C19 remains synthetic Development and cannot establish geometric scenario transfer, deployment, product, user-benefit, reliability, or safety authority.",
        "The frozen X42 chain is scored once; failed source admission is NOT_EVALUABLE and cannot be repaired after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(c17.BASE_PROTOCOL) != c17.BASE_PROTOCOL_SHA256:
        raise RuntimeError("C16 base protocol drift")
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
                "status": "C19_X42_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "x42_sha256": c17.X42_PREDICTOR_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
