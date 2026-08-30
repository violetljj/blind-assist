"""Score the C7 source-disjoint X24/X30 pre-truth development recovery.

The protocol, X30 predictor identity, prediction envelopes, and all decision
thresholds are fixed in this file.  Scoring is limited to uncensored prefixes;
the captured three-second future-truth tail is used only to adjudicate those
prefixes and is excluded from frame metrics.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x25_rigid_footprint_scorer as base  # noqa: E402


X24_SCHEMA = "blindassist-dtr-carla-x23-x24-predictions-v1"
X24_FREEZE_SCHEMA = "blindassist-dtr-carla-x24-freeze-v1"
X30_SCHEMA = "blindassist-dtr-carla-x30-adaptive-surface-interval-predictions-v1"
X30_FREEZE_SCHEMA = "blindassist-dtr-carla-x30-adaptive-surface-interval-freeze-v1"
ARM_X24 = "X24_ISSUED_PLAN_ADHERENCE"
ARM_X30 = "X30_ISSUED_PLAN_ADAPTIVE_SURFACE_CONTACT_INTERVAL"
EXPECTED_PROTOCOL_SCHEMA = 7
EXPECTED_COHORT_ID = "DTR_CARLA_C7_X30_SOURCE_DISJOINT_CONFIRMATION_V1"
EXPECTED_PROTOCOL_SHA256 = (
    "822976A7F34379CFCA48C800A829704C6774A5D8BFC3FC80A596E50C8B5EF55A"
)
EXPECTED_PARENT_COHORT_ID = "DTR_CARLA_C6_X29_SOURCE_DISJOINT_CONFIRMATION_V1"
EXPECTED_PARENT_PROTOCOL_CANONICAL_SHA256 = (
    "5530C1345F788C8C5745A8253AE7E9C249B70F86F7E80477D68FCB0337B5D1AE"
)
EXPECTED_X30_PREDICTOR_SHA256 = (
    "004419CEA2716D7E7BE8FA7BA41450DC6C0481DD3DF55F5D9DC9C08DEEF06ECA"
)
SOURCE_COMPLETE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
PREDICTION_STATUS = "SEALED_TRUTH_BLIND_PENDING_SCORE"
FREEZE_STATUS = "FROZEN_TRUTH_BLIND_PENDING_PREDICTION"
OCCLUSION_CONTRACT = "track_then_physical_loss_pair_01"

EPISODES = tuple(f"ep_{value:02d}" for value in range(1, 9))
CONTACT_EPISODES = ("ep_01", "ep_03", "ep_05", "ep_07")
SAFE_EPISODES = ("ep_02", "ep_04", "ep_06", "ep_08")
DYNAMIC_CONTACT_EPISODES = ("ep_03", "ep_05", "ep_07")
DYNAMIC_SAFE_EPISODES = ("ep_04", "ep_06", "ep_08")
RETAINED_OCCLUSION_EPISODES = ("ep_01", "ep_02")
SCORE_WINDOW_END_SECONDS = {
    "ep_01": 4.0,
    "ep_02": 4.0,
    "ep_03": 7.0,
    "ep_04": 7.0,
    "ep_05": 6.0,
    "ep_06": 6.0,
    "ep_07": 6.0,
    "ep_08": 6.0,
}
SAFE_SEGMENT_START_SECONDS = {
    "ep_02": 2.7,
    "ep_04": 0.0,
    "ep_06": 0.0,
    "ep_08": 0.0,
}
EXPECTED_WEATHER_BY_LAYOUT = {
    "layout_01": "CloudyNoon",
    "layout_02": "ClearSunset",
    "layout_03": "WetCloudyNoon",
    "layout_04": "SoftRainNoon",
}
EXPECTED_DYNAMIC_BLUEPRINTS = {
    "vehicle": "vehicle.audi.tt",
    "walker": "walker.pedestrian.0025",
    "motorcycle": "vehicle.kawasaki.ninja",
}
EXPECTED_DYNAMIC_PAIRS = [
    {
        "kind": "vehicle",
        "contact_episode": "ep_03",
        "safe_episode": "ep_04",
        "layout_id": "layout_02",
        "asset_key": "c7_vehicle_target",
        "target_blueprint": "vehicle.audi.tt",
        "pair_difference": "TARGET_LATERAL_OFFSET_ONLY",
    },
    {
        "kind": "walker",
        "contact_episode": "ep_05",
        "safe_episode": "ep_06",
        "layout_id": "layout_03",
        "asset_key": "c7_walker_target",
        "target_blueprint": "walker.pedestrian.0025",
        "pair_difference": "TARGET_LATERAL_OFFSET_ONLY",
    },
    {
        "kind": "motorcycle",
        "contact_episode": "ep_07",
        "safe_episode": "ep_08",
        "layout_id": "layout_04",
        "asset_key": "c7_motorcycle_target",
        "target_blueprint": "vehicle.kawasaki.ninja",
        "pair_difference": "TARGET_LATERAL_OFFSET_ONLY",
    },
]
EXPECTED_CAPTURE_SEED = 104729
TRUTH_TAIL_SECONDS = 3.0
MINIMUM_DYNAMIC_LEAD_SECONDS = 2.0
MINIMUM_DYNAMIC_CONTACT_RECALL = 0.80
MINIMUM_AGGREGATE_PRECISION = 0.95
MINIMUM_AGGREGATE_F1 = 0.80
ALLOWED_MOTION_AUTHORITIES = {
    "STATIC_SCENE",
    "RIGID_DYNAMIC",
    "EGO_CARRIED",
    "UNAUTHORIZED_MOTION",
}
RISK_INELIGIBLE_AUTHORITIES = {"EGO_CARRIED", "UNAUTHORIZED_MOTION"}
EPSILON = 1e-9


def prefix(
    rows: Sequence[Mapping[str, Any]], end_s: float
) -> list[Mapping[str, Any]]:
    return [value for value in rows if float(value["time_s"]) <= end_s + EPSILON]


def format_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def arm_frames_full(
    predictions: Mapping[str, Any], episode_id: str, arm: str
) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": int(frame["sample_index"]),
            "time_s": float(frame["time_s"]),
            "route_risk": bool(frame["arms"][arm]["route_risk"]),
        }
        for frame in predictions["episodes"][episode_id]["frames"]
    ]


def validate_protocol(protocol: Mapping[str, Any], protocol_path: Path) -> None:
    base.require(
        base.sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256,
        "c7_protocol_hash_not_frozen",
    )
    base.require(
        protocol.get("schema_version") == EXPECTED_PROTOCOL_SCHEMA,
        "c7_protocol_schema",
    )
    base.require(protocol.get("cohort_id") == EXPECTED_COHORT_ID, "c7_cohort")
    base.require(
        int(protocol["capture"]["seed"]) == EXPECTED_CAPTURE_SEED,
        "c7_capture_seed",
    )
    layouts = protocol["layouts"]
    base.require(tuple(layouts) == tuple(EXPECTED_WEATHER_BY_LAYOUT), "c7_layouts")
    base.require(
        {key: value["weather"] for key, value in layouts.items()}
        == EXPECTED_WEATHER_BY_LAYOUT,
        "c7_weather",
    )

    scenarios = {value["episode_id"]: value for value in protocol["scenarios"]}
    base.require(tuple(scenarios) == EPISODES, "c7_episode_order")
    base.require(
        tuple(key for key in EPISODES if scenarios[key]["expected_outcome"] == "CONTACT")
        == CONTACT_EPISODES,
        "c7_contact_scenarios",
    )
    base.require(
        tuple(key for key in EPISODES if scenarios[key]["expected_outcome"] == "SAFE")
        == SAFE_EPISODES,
        "c7_safe_scenarios",
    )

    contract = protocol["evaluation_contract"]
    base.require(
        float(contract["truth_tail_seconds"]) == TRUTH_TAIL_SECONDS,
        "c7_truth_tail",
    )
    base.require(
        contract["future_truth_rule"]
        == "SCORE_ONLY_FRAMES_WITH_FULL_CAPTURED_HORIZON",
        "c7_future_truth_rule",
    )
    base.require(
        contract["score_window_end_seconds"] == SCORE_WINDOW_END_SECONDS,
        "c7_score_windows",
    )
    base.require(tuple(contract["contact_episodes"]) == CONTACT_EPISODES, "c7_contacts")
    base.require(tuple(contract["safe_episodes"]) == SAFE_EPISODES, "c7_safe")
    base.require(
        tuple(contract["fresh_dynamic_contact_episodes"])
        == DYNAMIC_CONTACT_EPISODES,
        "c7_dynamic_contacts",
    )
    base.require(
        tuple(contract["fresh_dynamic_safe_episodes"]) == DYNAMIC_SAFE_EPISODES,
        "c7_dynamic_safe",
    )
    base.require(
        tuple(contract["retained_occlusion_episodes"])
        == RETAINED_OCCLUSION_EPISODES,
        "c7_retained_occlusion",
    )
    base.require(
        contract["safe_segment_start_seconds"] == SAFE_SEGMENT_START_SECONDS,
        "c7_safe_boundaries",
    )
    base.require(contract["dynamic_pairs"] == EXPECTED_DYNAMIC_PAIRS, "c7_dynamic_pairs")
    base.require(
        contract.get("all_physical_obstacles_are_truth_relevant") is True,
        "c7_truth_relevance",
    )

    source_disjoint = protocol["source_disjoint_contract"]
    base.require(
        source_disjoint["parent_cohort_id"] == EXPECTED_PARENT_COHORT_ID,
        "c7_parent_cohort",
    )
    base.require(
        source_disjoint["parent_protocol_canonical_sha256"]
        == EXPECTED_PARENT_PROTOCOL_CANONICAL_SHA256,
        "c7_parent_protocol_hash",
    )
    base.require(
        tuple(source_disjoint["retained_regression_episodes"])
        == RETAINED_OCCLUSION_EPISODES,
        "c7_regression_stratum",
    )
    base.require(
        tuple(source_disjoint["confirmation_episodes"]) == EPISODES[2:],
        "c7_confirmation_stratum",
    )
    base.require(
        int(source_disjoint["new_capture_seed"]) == EXPECTED_CAPTURE_SEED,
        "c7_disjoint_seed",
    )
    base.require(
        source_disjoint["weather_by_layout"] == EXPECTED_WEATHER_BY_LAYOUT,
        "c7_disjoint_weather",
    )
    base.require(
        source_disjoint["new_dynamic_target_blueprints"]
        == EXPECTED_DYNAMIC_BLUEPRINTS,
        "c7_disjoint_blueprints",
    )
    base.require(
        source_disjoint["dynamic_pairs"] == EXPECTED_DYNAMIC_PAIRS,
        "c7_disjoint_pairs",
    )
    base.require(
        set(source_disjoint["parent_dynamic_target_blueprints_excluded"])
        == {
            "vehicle.mercedes.sprinter",
            "walker.pedestrian.0024",
            "vehicle.lincoln.mkz_2020",
            "walker.pedestrian.0031",
            "vehicle.yamaha.yzf",
        },
        "c7_parent_targets_excluded",
    )

    occlusion = [
        value
        for value in protocol["occlusion_contracts"]
        if value.get("contract_id") == OCCLUSION_CONTRACT
    ]
    base.require(
        len(occlusion) == 1
        and tuple(occlusion[0]["episodes"]) == RETAINED_OCCLUSION_EPISODES,
        "c7_occlusion_contract",
    )
    horizon_s = float(protocol["route_contract"]["future_horizon_seconds"])
    base.require(
        TRUTH_TAIL_SECONDS + EPSILON >= horizon_s,
        "c7_truth_tail_shorter_than_horizon",
    )
    for episode_id, scenario in scenarios.items():
        duration_s = float(layouts[scenario["layout_id"]]["duration_seconds"])
        base.require(
            duration_s + EPSILON
            >= SCORE_WINDOW_END_SECONDS[episode_id] + TRUTH_TAIL_SECONDS,
            f"c7_declared_tail:{episode_id}",
        )


def validate_prediction_envelopes(
    x24: Mapping[str, Any],
    x30: Mapping[str, Any],
    freeze_x24: Mapping[str, Any],
    freeze_x30: Mapping[str, Any],
    run_root: Path,
) -> None:
    base.require(
        x24.get("schema") == X24_SCHEMA and x30.get("schema") == X30_SCHEMA,
        "prediction_schema",
    )
    base.require(
        x24.get("status") == PREDICTION_STATUS
        and x30.get("status") == PREDICTION_STATUS,
        "prediction_status",
    )
    base.require(
        x24.get("truth_blind") is True and x30.get("truth_blind") is True,
        "prediction_truth_blind",
    )
    base.require(
        x24.get("claim_boundary", {}).get("evaluator_opened") is False
        and x30.get("claim_boundary", {}).get("evaluator_opened") is False,
        "prediction_evaluator_opened",
    )
    base.require(
        x24.get("claim_boundary", {}).get("current_actor_oracle_used") is False
        and x30.get("claim_boundary", {}).get("current_actor_oracle_used") is False,
        "prediction_actor_oracle",
    )
    base.require(
        ARM_X24 in x24["arms"] and x30["arms"] == [ARM_X30],
        "prediction_arms",
    )
    base.require(
        tuple(x24["episodes"]) == EPISODES and tuple(x30["episodes"]) == EPISODES,
        "prediction_episodes",
    )

    base.require(freeze_x24.get("schema") == X24_FREEZE_SCHEMA, "x24_freeze_schema")
    base.require(freeze_x30.get("schema") == X30_FREEZE_SCHEMA, "x30_freeze_schema")
    base.require(
        freeze_x24.get("status") == FREEZE_STATUS
        and freeze_x30.get("status") == FREEZE_STATUS,
        "prediction_freeze_status",
    )
    base.require(
        freeze_x24.get("truth_blind") is True
        and freeze_x30.get("truth_blind") is True,
        "prediction_freeze_truth_blind",
    )
    x24_predictions_hash = base.sha256_file(run_root / "predictions-x24.json")
    x24_freeze_hash = base.sha256_file(run_root / "freeze-x24.json")
    x30_freeze_hash = base.sha256_file(run_root / "freeze-x30.json")
    base.require(
        x24["source"]["freeze_sha256"] == x24_freeze_hash,
        "x24_prediction_freeze_drift",
    )
    base.require(
        x30["source"]["freeze_sha256"] == x30_freeze_hash,
        "x30_prediction_freeze_drift",
    )
    base.require(
        freeze_x30["source"]["x24_freeze_sha256"] == x24_freeze_hash,
        "x30_x24_freeze_drift",
    )
    base.require(
        freeze_x30["source"]["x24_predictions_sha256"] == x24_predictions_hash
        and x30["source"]["x24_predictions_sha256"] == x24_predictions_hash,
        "x30_x24_prediction_drift",
    )
    base.require(
        freeze_x30["algorithm_files"]["x30_predictor"]["sha256"]
        == EXPECTED_X30_PREDICTOR_SHA256,
        "x30_predictor_not_frozen",
    )
    base.require(
        freeze_x30["fixed_constants"] == x30["fixed_constants"],
        "x30_constants_drift",
    )
    base.require(
        int(freeze_x24["episodes"]) == len(EPISODES)
        and int(freeze_x30["episodes"]) == len(EPISODES),
        "prediction_freeze_episode_count",
    )


def authority_invariants(
    predictions: Mapping[str, Any], score_end: Mapping[str, float]
) -> dict[str, Any]:
    risk_ineligible_authority_frames = 0
    non_dynamic_nonzero_velocity_frames = 0
    promotion_during_hold_frames = 0
    unknown_authority_frames = 0
    risk_eligible_track_count_mismatches = 0
    route_risk_without_confirmed_eligible_track_frames = 0
    dynamic = {
        key: {
            "route_risk_frames": 0,
            "rigid_dynamic_track_frames": 0,
            "rigid_dynamic_risk_track_frames": 0,
            "rigid_dynamic_confirmed_route_risk_frames": 0,
            "route_risk_without_confirmed_rigid_dynamic_frames": 0,
            "confirmed_non_rigid_risk_track_references": 0,
            "confirmed_missing_track_references": 0,
        }
        for key in DYNAMIC_CONTACT_EPISODES
    }

    for episode_id, episode in predictions["episodes"].items():
        previous: dict[str, str] = {}
        for frame in episode["frames"]:
            if float(frame["time_s"]) > float(score_end[episode_id]) + EPSILON:
                break
            tracks = {str(value["track_id"]): value for value in frame["tracks"]}
            eligible = {
                track_id for track_id, value in tracks.items() if bool(value["risk_eligible"])
            }
            risk_eligible_track_count_mismatches += int(
                int(frame["risk_eligible_tracks"]) != len(eligible)
            )
            for track_id, track in tracks.items():
                authority = str(track["motion_authority"])
                risk_eligible = bool(track["risk_eligible"])
                speed = abs(float(track["velocity_forward_mps"])) + abs(
                    float(track["velocity_right_mps"])
                )
                unknown_authority_frames += int(
                    authority not in ALLOWED_MOTION_AUTHORITIES
                )
                risk_ineligible_authority_frames += int(
                    authority in RISK_INELIGIBLE_AUTHORITIES and risk_eligible
                )
                non_dynamic_nonzero_velocity_frames += int(
                    authority != "RIGID_DYNAMIC" and speed > EPSILON
                )
                promotion_during_hold_frames += int(
                    str(track["disposition"]) == "HOLD"
                    and authority == "RIGID_DYNAMIC"
                    and previous.get(track_id) != "RIGID_DYNAMIC"
                )
                if episode_id in dynamic and authority == "RIGID_DYNAMIC":
                    dynamic[episode_id]["rigid_dynamic_track_frames"] += 1
                    dynamic[episode_id]["rigid_dynamic_risk_track_frames"] += int(
                        risk_eligible
                    )
                previous[track_id] = authority

            arm = frame["arms"][ARM_X30]
            if not bool(arm["route_risk"]):
                continue
            confirmed = {str(value) for value in arm["confirmed_risk_track_ids"]}
            confirmed_eligible = confirmed & eligible
            route_risk_without_confirmed_eligible_track_frames += int(
                not confirmed_eligible
            )
            if episode_id not in dynamic:
                continue
            dynamic[episode_id]["route_risk_frames"] += 1
            rigid_confirmed = {
                track_id
                for track_id in confirmed_eligible
                if str(tracks[track_id]["motion_authority"]) == "RIGID_DYNAMIC"
            }
            dynamic[episode_id][
                "rigid_dynamic_confirmed_route_risk_frames"
            ] += int(bool(rigid_confirmed))
            dynamic[episode_id][
                "route_risk_without_confirmed_rigid_dynamic_frames"
            ] += int(not rigid_confirmed)
            dynamic[episode_id][
                "confirmed_non_rigid_risk_track_references"
            ] += sum(
                track_id in tracks
                and bool(tracks[track_id]["risk_eligible"])
                and str(tracks[track_id]["motion_authority"]) != "RIGID_DYNAMIC"
                for track_id in confirmed
            )
            dynamic[episode_id]["confirmed_missing_track_references"] += sum(
                track_id not in tracks for track_id in confirmed
            )

    return {
        "risk_ineligible_authority_frames": risk_ineligible_authority_frames,
        "non_dynamic_nonzero_velocity_frames": non_dynamic_nonzero_velocity_frames,
        "promotion_during_hold_frames": promotion_during_hold_frames,
        "unknown_authority_frames": unknown_authority_frames,
        "risk_eligible_track_count_mismatches": risk_eligible_track_count_mismatches,
        "route_risk_without_confirmed_eligible_track_frames": (
            route_risk_without_confirmed_eligible_track_frames
        ),
        "dynamic_contacts": dynamic,
    }


def render_svg(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    x24 = aggregate[ARM_X24]
    x30 = aggregate[ARM_X30]
    contacts = result["contacts"]
    safe_segments = sum(
        result["safe"][key][ARM_X30]["false_alert_segment_count"]
        for key in SAFE_EPISODES
    )
    authority = result["authority_invariants"]["dynamic_contacts"]
    accent = "#22c55e" if result["gate_met"] else "#f97316"
    decision = html.escape(str(result["decision"]))
    rows = []
    for y, episode_id, label in (
        (465, "ep_03", "Lincoln vehicle"),
        (510, "ep_05", "Walker"),
        (555, "ep_07", "Yamaha motorcycle"),
    ):
        metric = contacts[episode_id][ARM_X30]
        rigid = authority[episode_id]["rigid_dynamic_confirmed_route_risk_frames"]
        rows.append(
            f'<text x="75" y="{y}" font-family="Segoe UI,sans-serif" '
            f'font-size="29" fill="#f8fafc">{label}: lead '
            f'{format_seconds(metric["first_alert_lead_seconds"])} s · recall '
            f'{metric["future_positive_recall"]:.3f} · confirmed RIGID frames '
            f'{rigid}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
<rect width="1400" height="900" fill="#07111f"/><rect x="35" y="35" width="1330" height="830" rx="28" fill="#101d30" stroke="#29435f" stroke-width="2"/>
<text x="75" y="105" font-family="Segoe UI,sans-serif" font-size="42" font-weight="700" fill="#f8fafc">X30 adaptive surface contact interval · CARLA C7</text>
<text x="75" y="158" font-family="Segoe UI,sans-serif" font-size="23" fill="{accent}">{decision}</text>
<text x="75" y="230" font-family="Segoe UI,sans-serif" font-size="24" fill="#93c5fd">Source-disjoint pre-truth recovery · uncensored scored-window metrics</text>
<text x="75" y="282" font-family="Segoe UI,sans-serif" font-size="35" fill="#cbd5e1">X24 F1 {x24['f1']:.3f} · precision {x24['precision']:.3f} · recall {x24['recall']:.3f}</text>
<text x="75" y="334" font-family="Segoe UI,sans-serif" font-size="35" font-weight="700" fill="{accent}">X30 F1 {x30['f1']:.3f} · precision {x30['precision']:.3f} · recall {x30['recall']:.3f}</text>
<text x="75" y="410" font-family="Segoe UI,sans-serif" font-size="24" fill="#93c5fd">Fresh dynamic CONTACT authority</text>
{''.join(rows)}
<text x="75" y="650" font-family="Segoe UI,sans-serif" font-size="24" fill="#93c5fd">SAFE separation and retained regression</text>
<text x="75" y="700" font-family="Segoe UI,sans-serif" font-size="31" font-weight="700" fill="{accent}">Adjudicated SAFE false-alert segments: {safe_segments}</text>
<text x="75" y="745" font-family="Segoe UI,sans-serif" font-size="31" fill="#f8fafc">Original occlusion coverage: X24 {result['occlusion'][ARM_X24]['coverage']:.3f} · X30 {result['occlusion'][ARM_X30]['coverage']:.3f}</text>
<text x="75" y="820" font-family="Segoe UI,sans-serif" font-size="21" fill="#94a3b8">8 episodes · 4 layouts · new seed/weather/targets · 3 s evaluator tail · scripted CARLA only</text>
</svg>'''


def score(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    result_path = run_root / "result-x30.json"
    svg_path = run_root / "result-x30.svg"
    base.require(
        not result_path.exists() and not svg_path.exists(),
        "x30_score_outputs_exist",
    )

    protocol = base.read_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    predictor_path = HERE / "dtr_carla_x30_adaptive_surface_interval_predictor.py"
    base.require(
        base.sha256_file(predictor_path) == EXPECTED_X30_PREDICTOR_SHA256,
        "x30_predictor_source_drift",
    )

    x24 = base.read_json(run_root / "predictions-x24.json")
    x30 = base.read_json(run_root / "predictions-x30.json")
    freeze_x24 = base.read_json(run_root / "freeze-x24.json")
    freeze_x30 = base.read_json(run_root / "freeze-x30.json")
    validate_prediction_envelopes(x24, x30, freeze_x24, freeze_x30, run_root)

    # The evaluator-bearing source is intentionally opened only after all
    # frozen protocol, predictor, prediction, and freeze identities validate.
    source_result = base.read_json(source_root / "result.json")
    base.require(
        source_result.get("status") == SOURCE_COMPLETE_STATUS,
        "source_incomplete",
    )
    base.require(
        bool(source_result.get("checks"))
        and all(bool(value) for value in source_result["checks"].values()),
        "source_gate_failed",
    )
    base.require(
        source_result["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256,
        "protocol_source_drift",
    )
    base.require(
        int(source_result["episode_count"]) == len(EPISODES)
        and int(source_result["layout_count"]) == len(EXPECTED_WEATHER_BY_LAYOUT),
        "source_cohort_count",
    )
    model_manifest_path = source_root / "model" / "manifest.json"
    base.require(
        base.sha256_file(model_manifest_path)
        == freeze_x24["model_manifest"]["sha256"]
        == freeze_x30["source"]["model_manifest_sha256"],
        "source_model_manifest_drift",
    )

    score_end = dict(SCORE_WINDOW_END_SECONDS)
    horizon_s = float(protocol["route_contract"]["future_horizon_seconds"])
    expected = set(EPISODES)
    evaluator_full = {
        episode_id: base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in EPISODES
    }
    predictions_full = {
        ARM_X24: {
            episode_id: arm_frames_full(x24, episode_id, ARM_X24)
            for episode_id in EPISODES
        },
        ARM_X30: {
            episode_id: arm_frames_full(x30, episode_id, ARM_X30)
            for episode_id in EPISODES
        },
    }
    for arm, episodes in predictions_full.items():
        for episode_id in expected:
            base.align(
                evaluator_full[episode_id],
                episodes[episode_id],
                f"{arm}:{episode_id}",
            )

    truth_tail_checks = {
        episode_id: (
            float(rows[-1]["time_s"]) + EPSILON
            >= score_end[episode_id] + horizon_s
        )
        for episode_id, rows in evaluator_full.items()
    }
    base.require(all(truth_tail_checks.values()), "right_censored_score_window")
    evaluator = {
        episode_id: prefix(rows, score_end[episode_id])
        for episode_id, rows in evaluator_full.items()
    }
    predictions = {
        arm: {
            episode_id: prefix(rows, score_end[episode_id])
            for episode_id, rows in episodes.items()
        }
        for arm, episodes in predictions_full.items()
    }

    aggregate = {
        arm: base.confusion(evaluator, episodes)
        for arm, episodes in predictions.items()
    }
    contacts = {
        episode_id: {
            arm: base.contact_metrics(evaluator[episode_id], episodes[episode_id])
            for arm, episodes in predictions.items()
        }
        for episode_id in CONTACT_EPISODES
    }
    safe = {
        episode_id: {
            arm: base.false_segments(
                episodes[episode_id], SAFE_SEGMENT_START_SECONDS[episode_id]
            )
            for arm, episodes in predictions.items()
        }
        for episode_id in SAFE_EPISODES
    }
    occlusion = base.occlusion_coverage(
        base.read_json(source_root / "evaluator" / "physical_occlusion_report.json"),
        predictions,
    )
    invariants = authority_invariants(x30, score_end)
    safe_segments = sum(
        safe[key][ARM_X30]["false_alert_segment_count"] for key in SAFE_EPISODES
    )
    dynamic_safe_segments = sum(
        safe[key][ARM_X30]["false_alert_segment_count"]
        for key in DYNAMIC_SAFE_EPISODES
    )
    dynamic_authority = invariants["dynamic_contacts"]
    gate_checks = {
        "all_scored_frames_have_full_realized_future": all(
            truth_tail_checks.values()
        ),
        "x30_detects_all_three_dynamic_contacts": all(
            contacts[key][ARM_X30]["event_detected_before_contact"]
            for key in DYNAMIC_CONTACT_EPISODES
        ),
        "each_dynamic_contact_has_at_least_2s_lead": all(
            contacts[key][ARM_X30]["first_alert_lead_seconds"] is not None
            and float(contacts[key][ARM_X30]["first_alert_lead_seconds"])
            + EPSILON
            >= MINIMUM_DYNAMIC_LEAD_SECONDS
            for key in DYNAMIC_CONTACT_EPISODES
        ),
        "each_dynamic_contact_future_positive_recall_at_least_0_80": all(
            contacts[key][ARM_X30]["future_positive_recall"] + EPSILON
            >= MINIMUM_DYNAMIC_CONTACT_RECALL
            for key in DYNAMIC_CONTACT_EPISODES
        ),
        "x30_has_zero_adjudicated_safe_risk_segments": safe_segments == 0,
        "x30_has_zero_dynamic_safe_risk_segments": dynamic_safe_segments == 0,
        "x30_aggregate_precision_at_least_0_95": (
            aggregate[ARM_X30]["precision"] + EPSILON
            >= MINIMUM_AGGREGATE_PRECISION
        ),
        "x30_aggregate_f1_at_least_0_80": (
            aggregate[ARM_X30]["f1"] + EPSILON >= MINIMUM_AGGREGATE_F1
        ),
        "x30_frame_f1_exceeds_x24": (
            aggregate[ARM_X30]["f1"] > aggregate[ARM_X24]["f1"] + EPSILON
        ),
        "x30_retains_original_occlusion_coverage": (
            occlusion[ARM_X30]["coverage"] + EPSILON
            >= occlusion[ARM_X24]["coverage"]
        ),
        "every_dynamic_contact_risk_is_exclusively_rigid_dynamic": all(
            dynamic_authority[key]["rigid_dynamic_confirmed_route_risk_frames"]
            > 0
            and dynamic_authority[key][
                "route_risk_without_confirmed_rigid_dynamic_frames"
            ]
            == 0
            and dynamic_authority[key][
                "confirmed_non_rigid_risk_track_references"
            ]
            == 0
            and dynamic_authority[key]["confirmed_missing_track_references"]
            == 0
            for key in DYNAMIC_CONTACT_EPISODES
        ),
        "ego_and_unauthorized_tracks_never_enter_risk": (
            invariants["risk_ineligible_authority_frames"] == 0
        ),
        "non_dynamic_authorities_have_zero_velocity": (
            invariants["non_dynamic_nonzero_velocity_frames"] == 0
        ),
        "hold_never_promotes_motion_authority": (
            invariants["promotion_during_hold_frames"] == 0
        ),
        "all_motion_authorities_are_known": (
            invariants["unknown_authority_frames"] == 0
        ),
        "risk_eligible_track_counts_are_consistent": (
            invariants["risk_eligible_track_count_mismatches"] == 0
        ),
        "route_risk_has_confirmed_eligible_track": (
            invariants["route_risk_without_confirmed_eligible_track_frames"] == 0
        ),
    }
    gate_met = all(gate_checks.values())
    result = {
        "schema": "blindassist-dtr-carla-c7-x30-source-disjoint-score-result-v1",
        "status": "COMPLETE",
        "decision": (
            "DTR_CARLA_X30_ADAPTIVE_SURFACE_CONTACT_INTERVAL_SOURCE_DISJOINT_PRETRUTH_DEVELOPMENT_GATE_MET"
            if gate_met
            else "DTR_CARLA_X30_ADAPTIVE_SURFACE_CONTACT_INTERVAL_SOURCE_DISJOINT_PRETRUTH_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "gate_checks": gate_checks,
        "thresholds": {
            "minimum_dynamic_contact_lead_seconds": MINIMUM_DYNAMIC_LEAD_SECONDS,
            "minimum_dynamic_contact_future_positive_recall": (
                MINIMUM_DYNAMIC_CONTACT_RECALL
            ),
            "maximum_safe_risk_segments": 0,
            "minimum_aggregate_precision": MINIMUM_AGGREGATE_PRECISION,
            "minimum_aggregate_f1": MINIMUM_AGGREGATE_F1,
            "x30_f1_must_strictly_exceed_x24": True,
            "dynamic_contact_risk_authority": "RIGID_DYNAMIC_ONLY",
        },
        "aggregate": aggregate,
        "contacts": contacts,
        "safe": safe,
        "occlusion": occlusion,
        "authority_invariants": invariants,
        "truth_tail_checks": truth_tail_checks,
        "score_window_end_seconds": score_end,
        "deltas": {
            "frame_f1": aggregate[ARM_X30]["f1"] - aggregate[ARM_X24]["f1"],
            "frame_precision": (
                aggregate[ARM_X30]["precision"]
                - aggregate[ARM_X24]["precision"]
            ),
            "frame_recall": (
                aggregate[ARM_X30]["recall"] - aggregate[ARM_X24]["recall"]
            ),
            "safe_risk_segments": safe_segments,
        },
        "source": {
            "source_result_sha256": base.sha256_file(source_root / "result.json"),
            "protocol_sha256": base.sha256_file(protocol_path),
            "x24_predictions_sha256": base.sha256_file(
                run_root / "predictions-x24.json"
            ),
            "x30_predictions_sha256": base.sha256_file(
                run_root / "predictions-x30.json"
            ),
            "x24_freeze_sha256": base.sha256_file(run_root / "freeze-x24.json"),
            "x30_freeze_sha256": base.sha256_file(run_root / "freeze-x30.json"),
            "x30_predictor_sha256": base.sha256_file(predictor_path),
            "scorer_sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "fresh_scripted_carla_source_disjoint_confirmation": False,
            "fresh_scripted_carla_source_disjoint_replay": False,
            "fresh_scripted_carla_source_disjoint_pretruth_development": True,
            "pristine_preopened_scorer_identity": False,
            "scorer_frozen_before_source": False,
            "predictor_frozen_before_source": False,
            "predictor_mechanical_schema_field_rename_after_source_before_truth": True,
            "evaluator_opened_before_correction": False,
            "retained_occlusion_is_regression_only": True,
            "full_horizon_truth_tail": True,
            "frozen_x30_predictor_sha256": EXPECTED_X30_PREDICTOR_SHA256,
            "real_world_confirmation": False,
            "product_default_authority": False,
            "deployment_or_safety_authority": False,
        },
    }
    base.write_json_exclusive(result_path, result)
    base.write_exclusive(svg_path, render_svg(result).encode("utf-8"))
    return {
        **result,
        "result_sha256": base.sha256_file(result_path),
        "svg_sha256": base.sha256_file(svg_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(
        json.dumps(
            score(parse_args()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"ERROR: {error}") from error
