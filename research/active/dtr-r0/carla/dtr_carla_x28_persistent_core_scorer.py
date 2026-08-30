"""Score the predeclared fresh C5 X24/X28 development replay."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
import sys

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x25_rigid_footprint_scorer as base  # noqa: E402


X24_SCHEMA = "blindassist-dtr-carla-x23-x24-predictions-v1"
X28_SCHEMA = "blindassist-dtr-carla-x28-persistent-core-predictions-v1"
ARM_X24 = "X24_ISSUED_PLAN_ADHERENCE"
ARM_X28 = "X28_ISSUED_PLAN_PERSISTENT_OCCUPANCY_CORE"
CONTACT_EPISODES = ("ep_01", "ep_03", "ep_05")
SAFE_EPISODES = ("ep_02", "ep_04", "ep_06")
FRESH_CONTACT_EPISODES = ("ep_03", "ep_05")
FRESH_SAFE_EPISODES = ("ep_04", "ep_06")
SAFE_START_SECONDS = {"ep_02": 2.70, "ep_04": 0.0, "ep_06": 0.0}
MINIMUM_DYNAMIC_LEAD_SECONDS = 2.0
MINIMUM_DYNAMIC_CONTACT_RECALL = 0.80
MINIMUM_AGGREGATE_PRECISION = 0.95
MINIMUM_AGGREGATE_F1 = 0.80
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


def authority_invariants(
    predictions: Mapping[str, Any], score_end: Mapping[str, float]
) -> dict[str, Any]:
    risk_ineligible_authority_frames = 0
    non_dynamic_nonzero_velocity_frames = 0
    promotion_during_hold_frames = 0
    rigid_track_frames = {key: 0 for key in FRESH_CONTACT_EPISODES}
    rigid_risk_track_frames = {key: 0 for key in FRESH_CONTACT_EPISODES}
    for episode_id, episode in predictions["episodes"].items():
        previous: dict[str, str] = {}
        for frame in episode["frames"]:
            if float(frame["time_s"]) > float(score_end[episode_id]) + EPSILON:
                break
            for track in frame["tracks"]:
                track_id = str(track["track_id"])
                authority = str(track["motion_authority"])
                risk_eligible = bool(track["risk_eligible"])
                speed = abs(float(track["velocity_forward_mps"])) + abs(
                    float(track["velocity_right_mps"])
                )
                if authority in {"EGO_CARRIED", "UNAUTHORIZED_MOTION"} and risk_eligible:
                    risk_ineligible_authority_frames += 1
                if authority != "RIGID_DYNAMIC" and speed > EPSILON:
                    non_dynamic_nonzero_velocity_frames += 1
                if (
                    str(track["disposition"]) == "HOLD"
                    and authority == "RIGID_DYNAMIC"
                    and previous.get(track_id) != "RIGID_DYNAMIC"
                ):
                    promotion_during_hold_frames += 1
                if episode_id in rigid_track_frames and authority == "RIGID_DYNAMIC":
                    rigid_track_frames[episode_id] += 1
                    rigid_risk_track_frames[episode_id] += int(risk_eligible)
                previous[track_id] = authority
    return {
        "risk_ineligible_authority_frames": risk_ineligible_authority_frames,
        "non_dynamic_nonzero_velocity_frames": non_dynamic_nonzero_velocity_frames,
        "promotion_during_hold_frames": promotion_during_hold_frames,
        "rigid_dynamic_track_frames": rigid_track_frames,
        "rigid_dynamic_risk_track_frames": rigid_risk_track_frames,
    }


def render_svg(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    x24 = aggregate[ARM_X24]
    x28 = aggregate[ARM_X28]
    contacts = result["contacts"]
    safe_segments = sum(
        result["safe"][key][ARM_X28]["false_alert_segment_count"]
        for key in SAFE_EPISODES
    )
    rigid = result["authority_invariants"]["rigid_dynamic_risk_track_frames"]
    accent = "#22c55e" if result["gate_met"] else "#f97316"
    decision = html.escape(str(result["decision"]))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="860" viewBox="0 0 1400 860">
<rect width="1400" height="860" fill="#07111f"/><rect x="35" y="35" width="1330" height="790" rx="28" fill="#101d30" stroke="#29435f" stroke-width="2"/>
<text x="75" y="105" font-family="Segoe UI,sans-serif" font-size="42" font-weight="700" fill="#f8fafc">X28 persistent occupancy core · CARLA C5</text>
<text x="75" y="158" font-family="Segoe UI,sans-serif" font-size="24" fill="{accent}">{decision}</text>
<text x="75" y="230" font-family="Segoe UI,sans-serif" font-size="24" fill="#93c5fd">Uncensored scored-window metrics</text>
<text x="75" y="282" font-family="Segoe UI,sans-serif" font-size="36" fill="#cbd5e1">X24 F1 {x24['f1']:.3f} · precision {x24['precision']:.3f} · recall {x24['recall']:.3f}</text>
<text x="75" y="334" font-family="Segoe UI,sans-serif" font-size="36" font-weight="700" fill="{accent}">X28 F1 {x28['f1']:.3f} · precision {x28['precision']:.3f} · recall {x28['recall']:.3f}</text>
<text x="75" y="415" font-family="Segoe UI,sans-serif" font-size="24" fill="#93c5fd">Fresh moving CONTACT arms</text>
<text x="75" y="465" font-family="Segoe UI,sans-serif" font-size="32" fill="#f8fafc">Sprinter lead {format_seconds(contacts['ep_03'][ARM_X28]['first_alert_lead_seconds'])} s · positive recall {contacts['ep_03'][ARM_X28]['future_positive_recall']:.3f}</text>
<text x="75" y="510" font-family="Segoe UI,sans-serif" font-size="32" fill="#f8fafc">Walker lead {format_seconds(contacts['ep_05'][ARM_X28]['first_alert_lead_seconds'])} s · positive recall {contacts['ep_05'][ARM_X28]['future_positive_recall']:.3f}</text>
<text x="75" y="590" font-family="Segoe UI,sans-serif" font-size="24" fill="#93c5fd">Authority and SAFE separation</text>
<text x="75" y="640" font-family="Segoe UI,sans-serif" font-size="32" fill="#f8fafc">RIGID_DYNAMIC risk frames: Sprinter {rigid['ep_03']} · Walker {rigid['ep_05']}</text>
<text x="75" y="685" font-family="Segoe UI,sans-serif" font-size="32" font-weight="700" fill="{accent}">SAFE false-alert segments: {safe_segments}</text>
<text x="75" y="760" font-family="Segoe UI,sans-serif" font-size="21" fill="#94a3b8">Fresh seed + moving vehicle + moving pedestrian + 3 s evaluator tail · 1280×720 RGB-D</text>
</svg>'''


def score(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    result_path = run_root / "result-x28.json"
    svg_path = run_root / "result-x28.svg"
    base.require(
        not result_path.exists() and not svg_path.exists(),
        "x28_score_outputs_exist",
    )
    protocol = base.read_json(protocol_path)
    source_result = base.read_json(source_root / "result.json")
    base.require(
        source_result.get("status") == "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE",
        "source_incomplete",
    )
    base.require(all(bool(value) for value in source_result["checks"].values()), "source_gate_failed")
    base.require(
        source_result["protocol_sha256"] == base.sha256_file(protocol_path),
        "protocol_source_drift",
    )
    contract = protocol["evaluation_contract"]
    score_end = {
        str(key): float(value)
        for key, value in contract["score_window_end_seconds"].items()
    }
    horizon_s = float(protocol["route_contract"]["future_horizon_seconds"])
    tail_s = float(contract["truth_tail_seconds"])
    base.require(tail_s + EPSILON >= horizon_s, "truth_tail_shorter_than_horizon")

    x24 = base.read_json(run_root / "predictions-x24.json")
    x28 = base.read_json(run_root / "predictions-x28.json")
    base.require(
        x24.get("schema") == X24_SCHEMA and x28.get("schema") == X28_SCHEMA,
        "prediction_schema",
    )
    expected = set(CONTACT_EPISODES + SAFE_EPISODES)
    base.require(
        set(x24["episodes"]) == expected
        and set(x28["episodes"]) == expected
        and set(score_end) == expected,
        "prediction_episodes",
    )

    evaluator_full = {
        episode_id: base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in sorted(expected)
    }
    predictions_full = {
        ARM_X24: {
            episode_id: arm_frames_full(x24, episode_id, ARM_X24)
            for episode_id in sorted(expected)
        },
        ARM_X28: {
            episode_id: arm_frames_full(x28, episode_id, ARM_X28)
            for episode_id in sorted(expected)
        },
    }
    truth_tail_checks: dict[str, bool] = {}
    for arm, episodes in predictions_full.items():
        for episode_id in expected:
            base.align(
                evaluator_full[episode_id],
                episodes[episode_id],
                f"{arm}:{episode_id}",
            )
    for episode_id, rows in evaluator_full.items():
        captured_end = float(rows[-1]["time_s"])
        truth_tail_checks[episode_id] = (
            captured_end + EPSILON >= score_end[episode_id] + horizon_s
        )
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
                episodes[episode_id], SAFE_START_SECONDS[episode_id]
            )
            for arm, episodes in predictions.items()
        }
        for episode_id in SAFE_EPISODES
    }
    occlusion = base.occlusion_coverage(
        base.read_json(source_root / "evaluator" / "physical_occlusion_report.json"),
        predictions,
    )
    invariants = authority_invariants(x28, score_end)
    safe_segments = sum(
        safe[key][ARM_X28]["false_alert_segment_count"] for key in SAFE_EPISODES
    )
    fresh_safe_segments = sum(
        safe[key][ARM_X28]["false_alert_segment_count"]
        for key in FRESH_SAFE_EPISODES
    )
    gate_checks = {
        "all_scored_frames_have_full_realized_future": all(truth_tail_checks.values()),
        "x28_detects_both_fresh_dynamic_contacts": all(
            contacts[key][ARM_X28]["event_detected_before_contact"]
            for key in FRESH_CONTACT_EPISODES
        ),
        "each_dynamic_contact_has_at_least_2s_lead": all(
            contacts[key][ARM_X28]["first_alert_lead_seconds"] is not None
            and float(contacts[key][ARM_X28]["first_alert_lead_seconds"])
            + EPSILON
            >= MINIMUM_DYNAMIC_LEAD_SECONDS
            for key in FRESH_CONTACT_EPISODES
        ),
        "each_dynamic_contact_future_positive_recall_at_least_0_80": all(
            contacts[key][ARM_X28]["future_positive_recall"] + EPSILON
            >= MINIMUM_DYNAMIC_CONTACT_RECALL
            for key in FRESH_CONTACT_EPISODES
        ),
        "x28_has_zero_safe_false_alert_segments": safe_segments == 0,
        "x28_has_zero_fresh_safe_false_alert_segments": fresh_safe_segments == 0,
        "x28_aggregate_precision_at_least_0_95": (
            aggregate[ARM_X28]["precision"] + EPSILON
            >= MINIMUM_AGGREGATE_PRECISION
        ),
        "x28_aggregate_f1_at_least_0_80": (
            aggregate[ARM_X28]["f1"] + EPSILON >= MINIMUM_AGGREGATE_F1
        ),
        "x28_frame_f1_exceeds_x24": (
            aggregate[ARM_X28]["f1"] > aggregate[ARM_X24]["f1"] + EPSILON
        ),
        "x28_retains_original_occlusion_coverage": (
            occlusion[ARM_X28]["coverage"] + EPSILON
            >= occlusion[ARM_X24]["coverage"]
        ),
        "both_fresh_contacts_obtain_rigid_dynamic_risk_authority": all(
            invariants["rigid_dynamic_risk_track_frames"][key] > 0
            for key in FRESH_CONTACT_EPISODES
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
    }
    gate_met = all(gate_checks.values())
    result = {
        "schema": "blindassist-dtr-carla-c5-x28-score-result-v1",
        "status": "COMPLETE",
        "decision": (
            "DTR_CARLA_X28_PERSISTENT_OCCUPANCY_CORE_DEVELOPMENT_GATE_MET"
            if gate_met
            else "DTR_CARLA_X28_PERSISTENT_OCCUPANCY_CORE_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "gate_checks": gate_checks,
        "aggregate": aggregate,
        "contacts": contacts,
        "safe": safe,
        "occlusion": occlusion,
        "authority_invariants": invariants,
        "truth_tail_checks": truth_tail_checks,
        "score_window_end_seconds": score_end,
        "deltas": {
            "frame_f1": aggregate[ARM_X28]["f1"] - aggregate[ARM_X24]["f1"],
            "frame_precision": (
                aggregate[ARM_X28]["precision"] - aggregate[ARM_X24]["precision"]
            ),
            "frame_recall": (
                aggregate[ARM_X28]["recall"] - aggregate[ARM_X24]["recall"]
            ),
            "safe_false_alert_segments": safe_segments,
        },
        "source": {
            "source_result_sha256": base.sha256_file(source_root / "result.json"),
            "protocol_sha256": base.sha256_file(protocol_path),
            "x24_predictions_sha256": base.sha256_file(
                run_root / "predictions-x24.json"
            ),
            "x28_predictions_sha256": base.sha256_file(
                run_root / "predictions-x28.json"
            ),
            "x28_freeze_sha256": base.sha256_file(run_root / "freeze-x28.json"),
            "scorer_sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "fresh_scripted_carla_development": True,
            "full_horizon_truth_tail": True,
            "source_disjoint_confirmation": False,
            "real_world_confirmation": False,
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
