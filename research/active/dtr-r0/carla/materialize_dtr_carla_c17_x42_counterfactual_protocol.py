"""Materialize the preregistered C17 counterfactual X42 falsifier cohort.

C17 is a new synthetic Development source.  It preserves the C16 sensor and
route-risk contracts while changing capture identity, scene orientation,
target appearance, weather, occluder direction, and safe-twin clearance.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import dtr_carla_c2_rich_scene as c2


ROOT = Path(__file__).resolve().parents[4]
BASE_PROTOCOL = (
    ROOT
    / "artifacts.local"
    / "work"
    / "c16-delayed-release-shell-probe-20260831-060000-v1"
    / "protocol.json"
)
BASE_PROTOCOL_SHA256 = (
    "37E3DAD535C003B9AFD785C33E33C09F58630785A9A559988BC59C0A33DE4666"
)
COHORT_ID = "DTR_CARLA_C17_X42_COUNTERFACTUAL_FALSIFIER_V1"
CAPTURE_SEED = 170842
FORWARD_SCALE = 1.08
SAFE_RIGHT_SCALE = 0.90

X42_PREDICTOR_SHA256 = (
    "824439389706E5ED844A1011268833686DE40D9834656029AB381412A421F1CD"
)
X42_PAUSE_MANIFEST_SHA256 = (
    "D0428EFA7C5BE79FA0535AA79A820D0245CF2766B907C3AE8C087012683408B4"
)

TARGET_TEMPLATES = {
    "c8_l01": "w21",
    "c8_l02": "v_micro",
    "c8_l03": "v_prius",
    "c8_l04": "v_ambulance",
}
WEATHERS = {
    "c8_l01": "ClearNoon",
    "c8_l02": "CloudyNoon",
    "c8_l03": "SoftRainSunset",
    "c8_l04": "WetCloudyNoon",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _transform_trajectory(
    value: dict[str, Any], *, safe_twin: bool = False
) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["start_forward_m"] = round(
        float(result["start_forward_m"]) * FORWARD_SCALE, 6
    )
    right_scale = -SAFE_RIGHT_SCALE if safe_twin else -1.0
    result["start_right_m"] = round(
        float(result["start_right_m"]) * right_scale, 6
    )
    for segment in result["segments"]:
        segment["velocity_forward_mps"] = round(
            float(segment["velocity_forward_mps"]) * FORWARD_SCALE, 6
        )
        segment["velocity_right_mps"] = round(
            -float(segment["velocity_right_mps"]), 6
        )
    return result


def _transform_occluder(value: dict[str, Any]) -> dict[str, Any]:
    result = _transform_trajectory(value)
    # Move the same physical occluder across the sightline from the opposite
    # side and one sample later.  Contact/safe twins receive this exact motion.
    for segment in result["segments"]:
        start = float(segment["start_s"])
        if start > 0.0:
            segment["start_s"] = round(start + 0.1, 6)
    return result


def _replace_special_trajectories(protocol: dict[str, Any]) -> None:
    library = protocol["trajectory_library"]
    wearer_name = "c17_wearer_route"
    library[wearer_name] = _transform_trajectory(library["c8_wearer_route"])

    cache: dict[tuple[str, str], str] = {}
    for scenario in protocol["scenarios"]:
        scenario["wearer_trajectory"] = wearer_name
        scenario["navigation_session_id"] = (
            f"c17_{scenario['navigation_session_id']}"
        )
        scenario["scenario_role"] = str(scenario["scenario_role"]).replace(
            "c8_transport_cone", "c17_counterfactual"
        )
        plan = scenario["issued_plan"]
        plan["plan_id"] = f"c17_{plan['plan_id']}"
        plan["session_id"] = scenario["navigation_session_id"]
        for waypoint in plan["time_parameterized_waypoints"]:
            waypoint["forward_m"] = round(
                float(waypoint["forward_m"]) * FORWARD_SCALE, 6
            )
            waypoint["right_m"] = round(-float(waypoint["right_m"]), 6)

        replacements: dict[str, str] = {}
        for asset_key, old_name in scenario["asset_trajectories"].items():
            role = (
                "occluder"
                if "occluder" in asset_key
                else "safe_target"
                if "target" in asset_key and scenario["expected_outcome"] == "SAFE"
                else "target"
                if "target" in asset_key
                else "alias"
            )
            key = (str(old_name), role)
            if key not in cache:
                new_name = f"c17_{old_name}_{role}"
                if role == "occluder":
                    transformed = _transform_occluder(library[str(old_name)])
                else:
                    transformed = _transform_trajectory(
                        library[str(old_name)], safe_twin=role == "safe_target"
                    )
                library[new_name] = transformed
                cache[key] = new_name
            replacements[str(asset_key)] = cache[key]
        scenario["asset_trajectories"] = replacements


def _replace_scene_identity(protocol: dict[str, Any]) -> None:
    for layout_id, layout in protocol["layouts"].items():
        anchor = layout["anchor"]
        anchor["forward_xy"] = [round(-float(v), 9) for v in anchor["forward_xy"]]
        anchor["right_xy"] = [round(-float(v), 9) for v in anchor["right_xy"]]
        layout["weather"] = WEATHERS[layout_id]
        layout["witness"]["right_m"] = -float(layout["witness"]["right_m"])
        template = TARGET_TEMPLATES[layout_id]
        for asset in layout["assets"]:
            key = str(asset["asset_key"])
            if "target" in key or "alias" in key:
                asset["template"] = template
                asset["role"] = str(asset["role"]).replace("c8_", "c17_")
            if "occluder" in key:
                asset["role"] = "c17_opposite_direction_physical_occluder"
            if key.endswith("target"):
                asset["track_id"] = f"c17_{layout_id}_target"
            elif key.endswith("alias"):
                asset["track_id"] = f"c17_{layout_id}_alias"
            elif key.endswith("occluder"):
                asset["track_id"] = f"c17_{layout_id}_occluder"


def materialize(base: dict[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(base)
    protocol["schema_version"] = 17
    protocol["cohort_id"] = COHORT_ID
    protocol["evidence_class"] = (
        "synthetic_preregistered_source_disjoint_counterfactual_development"
    )
    protocol["objective"] = (
        "Falsify frozen X42 once on paired lateral near-miss and temporarily "
        "occluded true-contact twins under changed appearance, weather, scene "
        "orientation, capture identity, and occluder direction."
    )
    protocol["capture"]["seed"] = CAPTURE_SEED
    _replace_special_trajectories(protocol)
    _replace_scene_identity(protocol)

    for contract in protocol["occlusion_contracts"]:
        contract["contract_id"] = str(contract["contract_id"]).replace(
            "c8_transport_cone", "c17_counterfactual"
        )
        contract["planned_occlusion_window_s"] = [
            round(float(value) + 0.1, 6)
            for value in contract["planned_occlusion_window_s"]
        ]

    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c17-x42-source-disjoint-contract-v1",
        "parent_development_cohort_id": base.get("cohort_id"),
        "parent_protocol_sha256": BASE_PROTOCOL_SHA256,
        "new_capture_seed": CAPTURE_SEED,
        "changed_scene_orientation": True,
        "changed_target_blueprints": TARGET_TEMPLATES,
        "changed_weather_by_layout": WEATHERS,
        "opposite_occluder_direction": True,
        "safe_twin_clearance_scale": SAFE_RIGHT_SCALE,
        "target_and_wearer_forward_scale": FORWARD_SCALE,
        "probe_or_formal_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
    }
    protocol["c17_x42_preregistration"] = {
        "schema": "dtr-carla-c17-x42-preregistration-v1",
        "frozen_x42_predictor_sha256": X42_PREDICTOR_SHA256,
        "frozen_x42_pause_manifest_sha256": X42_PAUSE_MANIFEST_SHA256,
        "baselines": ["X24", "X42"],
        "score_window_seconds": [0.0, 6.0],
        "primary_gate": {
            "minimum_precision": 0.85,
            "minimum_recall": 0.75,
            "minimum_f1": 0.80,
            "minimum_f1_gain_over_x24": 0.10,
            "minimum_each_contact_recall": 0.65,
            "maximum_total_safe_false_alert_segments": 5,
            "maximum_safe_false_alert_segments_per_episode": 3,
            "required_continuous_contact_episodes": 4,
            "required_parent_ancestry_episodes": 4,
        },
        "stretch_target": {
            "precision": 0.90,
            "recall": 0.80,
            "f1": 0.85,
        },
        "outcomes": [
            "GATE_MET",
            "GATE_NOT_MET",
            "SOURCE_NOT_EVALUABLE",
        ],
        "no_post_capture_threshold_or_scenario_tuning": True,
    }
    protocol["claim_boundary"] = [
        "C17 is source-disjoint from C16 at capture identity, target appearance, weather, orientation, occluder direction, and safe clearance.",
        "C17 remains synthetic Development and cannot establish deployment, product, user-benefit, reliability, or safety authority.",
        "The frozen X42 chain is scored once; failed source admission is NOT_EVALUABLE and cannot be repaired after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(BASE_PROTOCOL) != BASE_PROTOCOL_SHA256:
        raise RuntimeError("C16 base protocol drift")
    protocol = materialize(_read_json(BASE_PROTOCOL))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "C17_X42_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "x42_sha256": X42_PREDICTOR_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
