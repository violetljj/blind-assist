"""Score the predeclared fresh C4 X24/X26 development replay."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
import sys

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x25_rigid_footprint_scorer as base  # noqa: E402


X24_SCHEMA = "blindassist-dtr-carla-x23-x24-predictions-v1"
X26_SCHEMA = "blindassist-dtr-carla-x26-support-consensus-predictions-v1"
ARM_X24 = "X24_ISSUED_PLAN_ADHERENCE"
ARM_X26 = "X26_ISSUED_PLAN_SUPPORT_CONSENSUS_FOOTPRINT"
CONTACT_EPISODES = ("ep_01", "ep_03", "ep_05")
SAFE_EPISODES = ("ep_02", "ep_04", "ep_06")
EDGE_CONTACT_EPISODES = ("ep_03", "ep_05")
EDGE_SAFE_EPISODES = ("ep_04", "ep_06")
MINIMUM_EDGE_LEAD_GAIN_SECONDS = 1.0
EPSILON = 1e-9


def lead_gain(x24: dict[str, Any], x26: dict[str, Any]) -> float | None:
    x26_lead = x26["first_alert_lead_seconds"]
    if x26_lead is None:
        return None
    x24_lead = x24["first_alert_lead_seconds"]
    baseline_lead = 0.0 if x24_lead is None else float(x24_lead)
    return float(x26_lead) - baseline_lead


def format_seconds(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def render_svg(result: dict[str, Any]) -> str:
    x24 = result["aggregate"][ARM_X24]
    x26 = result["aggregate"][ARM_X26]
    contacts = result["contacts"]
    safe = result["safe"]
    hgv_gain = result["edge_lead_gain_seconds"]["ep_03"]
    motorcycle_gain = result["edge_lead_gain_seconds"]["ep_05"]
    safe_segments = sum(safe[key][ARM_X26]["false_alert_segment_count"] for key in EDGE_SAFE_EPISODES)
    accent = "#22c55e" if result["gate_met"] else "#f97316"
    decision = html.escape(result["decision"])
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="820" viewBox="0 0 1400 820">
<rect width="1400" height="820" fill="#07111f"/><rect x="35" y="35" width="1330" height="750" rx="28" fill="#101d30" stroke="#29435f" stroke-width="2"/>
<text x="75" y="105" font-family="Segoe UI,sans-serif" font-size="42" font-weight="700" fill="#f8fafc">X26 support-consensus CARLA C4</text>
<text x="75" y="158" font-family="Segoe UI,sans-serif" font-size="25" fill="{accent}">{decision}</text>
<text x="75" y="235" font-family="Segoe UI,sans-serif" font-size="25" fill="#93c5fd">Fresh-cohort frame metrics</text>
<text x="75" y="285" font-family="Segoe UI,sans-serif" font-size="38" fill="#cbd5e1">X24 F1 {x24['f1']:.3f} · recall {x24['recall']:.3f}</text>
<text x="75" y="335" font-family="Segoe UI,sans-serif" font-size="38" font-weight="700" fill="{accent}">X26 F1 {x26['f1']:.3f} · recall {x26['recall']:.3f}</text>
<text x="75" y="420" font-family="Segoe UI,sans-serif" font-size="25" fill="#93c5fd">Edge CONTACT alert lead gain</text>
<text x="75" y="470" font-family="Segoe UI,sans-serif" font-size="32" fill="#f8fafc">HGV: {format_seconds(hgv_gain, signed=True)} s · X26 lead {format_seconds(contacts['ep_03'][ARM_X26]['first_alert_lead_seconds'])} s</text>
<text x="75" y="515" font-family="Segoe UI,sans-serif" font-size="32" fill="#f8fafc">Side-on motorcycle: {format_seconds(motorcycle_gain, signed=True)} s · X26 lead {format_seconds(contacts['ep_05'][ARM_X26]['first_alert_lead_seconds'])} s</text>
<text x="75" y="600" font-family="Segoe UI,sans-serif" font-size="25" fill="#93c5fd">Fresh SAFE false-alert segments</text>
<text x="75" y="650" font-family="Segoe UI,sans-serif" font-size="38" font-weight="700" fill="{accent}">{safe_segments}</text>
<text x="75" y="730" font-family="Segoe UI,sans-serif" font-size="21" fill="#94a3b8">New seed + HGV + motorcycle + changed weather | 1280×720 RGB-D | no class-size prior</text>
</svg>'''


def score(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    result_path = run_root / "result-x26.json"
    svg_path = run_root / "result-x26.svg"
    base.require(not result_path.exists() and not svg_path.exists(), "x26_score_outputs_exist")
    source_result = base.read_json(source_root / "result.json")
    base.require(source_result.get("status") == "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE", "source_incomplete")
    base.require(all(bool(value) for value in source_result["checks"].values()), "source_gate_failed")
    x24 = base.read_json(run_root / "predictions-x24.json")
    x26 = base.read_json(run_root / "predictions-x26.json")
    base.require(x24.get("schema") == X24_SCHEMA and x26.get("schema") == X26_SCHEMA, "prediction_schema")
    expected = set(CONTACT_EPISODES + SAFE_EPISODES)
    base.require(set(x24["episodes"]) == expected and set(x26["episodes"]) == expected, "prediction_episodes")

    evaluator = {
        episode_id: base.read_jsonl(source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl")
        for episode_id in sorted(expected)
    }
    predictions = {
        ARM_X24: {episode_id: base.arm_frames(x24, episode_id, ARM_X24) for episode_id in sorted(expected)},
        ARM_X26: {episode_id: base.arm_frames(x26, episode_id, ARM_X26) for episode_id in sorted(expected)},
    }
    for arm, episodes in predictions.items():
        for episode_id in expected:
            base.align(evaluator[episode_id], episodes[episode_id], f"{arm}:{episode_id}")

    aggregate = {arm: base.confusion(evaluator, episodes) for arm, episodes in predictions.items()}
    contacts = {
        episode_id: {
            arm: base.contact_metrics(evaluator[episode_id], episodes[episode_id])
            for arm, episodes in predictions.items()
        }
        for episode_id in CONTACT_EPISODES
    }
    safe = {
        episode_id: {
            arm: base.false_segments(episodes[episode_id], 2.70 if episode_id == "ep_02" else 0.0)
            for arm, episodes in predictions.items()
        }
        for episode_id in SAFE_EPISODES
    }
    occlusion = base.occlusion_coverage(
        base.read_json(source_root / "evaluator" / "physical_occlusion_report.json"),
        predictions,
    )
    gains = {
        episode_id: lead_gain(contacts[episode_id][ARM_X24], contacts[episode_id][ARM_X26])
        for episode_id in EDGE_CONTACT_EPISODES
    }
    fresh_safe_segments = sum(safe[key][ARM_X26]["false_alert_segment_count"] for key in EDGE_SAFE_EPISODES)
    checks = {
        "x26_detects_both_fresh_edge_contacts": all(
            contacts[key][ARM_X26]["event_detected_before_contact"] for key in EDGE_CONTACT_EPISODES
        ),
        "each_fresh_edge_contact_gains_at_least_1s_lead": all(
            gains[key] is not None and gains[key] + EPSILON >= MINIMUM_EDGE_LEAD_GAIN_SECONDS
            for key in EDGE_CONTACT_EPISODES
        ),
        "x26_has_zero_fresh_safe_false_alert_segments": fresh_safe_segments == 0,
        "x26_frame_f1_exceeds_x24": aggregate[ARM_X26]["f1"] > aggregate[ARM_X24]["f1"] + EPSILON,
        "x26_retains_original_occlusion_coverage": (
            occlusion[ARM_X26]["coverage"] + EPSILON >= occlusion[ARM_X24]["coverage"]
        ),
    }
    gate_met = all(checks.values())
    result = {
        "schema": "blindassist-dtr-carla-c4-x26-score-result-v1",
        "status": "COMPLETE",
        "decision": (
            "DTR_CARLA_X26_SUPPORT_CONSENSUS_DEVELOPMENT_GATE_MET"
            if gate_met
            else "DTR_CARLA_X26_SUPPORT_CONSENSUS_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "gate_checks": checks,
        "aggregate": aggregate,
        "contacts": contacts,
        "safe": safe,
        "occlusion": occlusion,
        "edge_lead_gain_seconds": gains,
        "deltas": {
            "frame_f1": aggregate[ARM_X26]["f1"] - aggregate[ARM_X24]["f1"],
            "frame_recall": aggregate[ARM_X26]["recall"] - aggregate[ARM_X24]["recall"],
            "fresh_safe_false_alert_segments": fresh_safe_segments,
        },
        "source": {
            "source_result_sha256": base.sha256_file(source_root / "result.json"),
            "x24_predictions_sha256": base.sha256_file(run_root / "predictions-x24.json"),
            "x26_predictions_sha256": base.sha256_file(run_root / "predictions-x26.json"),
            "x26_freeze_sha256": base.sha256_file(run_root / "freeze-x26.json"),
            "scorer_sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "fresh_scripted_carla_development": True,
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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(score(parse_args()), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
