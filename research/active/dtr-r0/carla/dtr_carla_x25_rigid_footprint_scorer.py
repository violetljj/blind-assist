"""Score the frozen C3 X24/X25 rigid-footprint development replay."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


X24_SCHEMA = "blindassist-dtr-carla-x23-x24-predictions-v1"
X25_SCHEMA = "blindassist-dtr-carla-x25-rigid-footprint-predictions-v1"
ARM_X24 = "X24_ISSUED_PLAN_ADHERENCE"
ARM_X25 = "X25_ISSUED_PLAN_RIGID_FOOTPRINT"
CONTACT_EPISODES = ("ep_01", "ep_03", "ep_05")
SAFE_EPISODES = ("ep_02", "ep_04", "ep_06")
EDGE_CONTACT_EPISODES = ("ep_03", "ep_05")
EDGE_SAFE_EPISODES = ("ep_04", "ep_06")
OCCLUSION_CONTRACT = "track_then_physical_loss_pair_01"
EPSILON = 1e-9


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    require(bool(rows), f"empty_jsonl:{path}")
    return rows


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_json_exclusive(path: Path, value: Any) -> None:
    write_exclusive(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def arm_frames(predictions: Mapping[str, Any], episode_id: str, arm: str) -> list[dict[str, Any]]:
    frames = predictions["episodes"][episode_id]["frames"]
    return [
        {
            "sample_index": int(frame["sample_index"]),
            "time_s": float(frame["time_s"]),
            "route_risk": bool(frame["arms"][arm]["route_risk"]),
        }
        for frame in frames
    ]


def uniform_period(rows: Sequence[Mapping[str, Any]]) -> float:
    deltas = [float(right["time_s"]) - float(left["time_s"]) for left, right in zip(rows, rows[1:])]
    require(bool(deltas) and min(deltas) > 0.0, "invalid_sample_period")
    period = sum(deltas) / len(deltas)
    require(max(abs(value - period) for value in deltas) <= 1e-6, "nonuniform_sample_period")
    return period


def align(
    evaluator: Sequence[Mapping[str, Any]],
    prediction: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    require(len(evaluator) == len(prediction), f"{label}:frame_count")
    for truth, frame in zip(evaluator, prediction):
        require(int(truth["sample_index"]) == int(frame["sample_index"]), f"{label}:sample")
        require(abs(float(truth["time_s"]) - float(frame["time_s"])) <= 1e-6, f"{label}:time")


def confusion(
    evaluator_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
    prediction_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for episode_id, truth_rows in evaluator_by_episode.items():
        for truth, prediction in zip(truth_rows, prediction_by_episode[episode_id]):
            target = bool(truth["truth"]["future_contact_within_horizon"])
            observed = bool(prediction["route_risk"])
            tp += int(target and observed)
            fp += int(not target and observed)
            tn += int(not target and not observed)
            fn += int(target and not observed)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def longest_false_run(frames: Sequence[Mapping[str, Any]], period_s: float) -> tuple[int, float]:
    longest = current = 0
    for frame in frames:
        if bool(frame["route_risk"]):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest, longest * period_s


def contact_metrics(
    evaluator: Sequence[Mapping[str, Any]],
    prediction: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contact_times = [float(row["time_s"]) for row in evaluator if bool(row["truth"]["current_contact"])]
    require(bool(contact_times), "contact_episode_without_contact")
    first_contact = contact_times[0]
    alert_times = [float(row["time_s"]) for row in prediction if bool(row["route_risk"])]
    first_alert = alert_times[0] if alert_times else None
    positive = [
        (truth, frame)
        for truth, frame in zip(evaluator, prediction)
        if bool(truth["truth"]["future_contact_within_horizon"])
    ]
    positive_alerts = sum(bool(frame["route_risk"]) for _truth, frame in positive)
    period = uniform_period(evaluator)
    continuity_rows = []
    if first_alert is not None and first_alert <= first_contact + EPSILON:
        continuity_rows = [
            frame
            for frame in prediction
            if first_alert - EPSILON <= float(frame["time_s"]) <= first_contact + EPSILON
        ]
    missed = sum(not bool(frame["route_risk"]) for frame in continuity_rows)
    longest_frames, longest_seconds = longest_false_run(continuity_rows, period) if continuity_rows else (0, 0.0)
    return {
        "first_contact_time_s": first_contact,
        "first_alert_time_s": first_alert,
        "first_alert_lead_seconds": None if first_alert is None else first_contact - first_alert,
        "event_detected_before_contact": first_alert is not None and first_alert <= first_contact + EPSILON,
        "future_positive_frames": len(positive),
        "future_positive_alert_frames": positive_alerts,
        "future_positive_recall": positive_alerts / max(1, len(positive)),
        "first_alert_to_contact_frames": len(continuity_rows),
        "first_alert_to_contact_alert_frames": len(continuity_rows) - missed,
        "first_alert_to_contact_coverage": (
            (len(continuity_rows) - missed) / len(continuity_rows) if continuity_rows else 0.0
        ),
        "first_alert_to_contact_missed_frames": missed,
        "longest_alert_gap_frames": longest_frames,
        "longest_alert_gap_seconds": longest_seconds,
    }


def false_segments(frames: Sequence[Mapping[str, Any]], start_s: float) -> dict[str, Any]:
    selected = [row for row in frames if float(row["time_s"]) + EPSILON >= start_s]
    segments: list[dict[str, float]] = []
    active: list[Mapping[str, Any]] = []
    for frame in selected + [{"route_risk": False, "time_s": math.inf}]:
        if bool(frame["route_risk"]):
            active.append(frame)
        elif active:
            segments.append(
                {
                    "start_s": float(active[0]["time_s"]),
                    "end_s": float(active[-1]["time_s"]),
                }
            )
            active = []
    return {
        "tail_start_s": start_s,
        "false_alert_frames": sum(bool(row["route_risk"]) for row in selected),
        "false_alert_segment_count": len(segments),
        "segments": segments,
    }


def occlusion_coverage(
    report: Sequence[Mapping[str, Any]],
    prediction_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    matches = [value for value in report if value.get("contract_id") == OCCLUSION_CONTRACT]
    require(len(matches) == 1 and matches[0].get("passed") is True, "occlusion_contract")
    indices = [int(value) for value in matches[0]["selected_indices"]["ep_01"]]
    output: dict[str, Any] = {"sample_indices": indices}
    for arm, episodes in prediction_by_arm.items():
        lookup = {int(row["sample_index"]): row for row in episodes["ep_01"]}
        covered = sum(bool(lookup[index]["route_risk"]) for index in indices)
        output[arm] = {
            "covered_frames": covered,
            "total_frames": len(indices),
            "coverage": covered / max(1, len(indices)),
        }
    return output


def svg_board(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    edge = result["edge_pairs"]
    occlusion = result["occlusion"]
    x24 = aggregate[ARM_X24]
    x25 = aggregate[ARM_X25]
    passed = result["gate_met"]
    accent = "#22c55e" if passed else "#f97316"
    decision = html.escape(str(result["decision"]))
    bus = edge["ep_03"]
    bike = edge["ep_05"]
    safe_segments = sum(edge[key][ARM_X25]["false_alert_segment_count"] for key in EDGE_SAFE_EPISODES)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="820" viewBox="0 0 1400 820">
<rect width="1400" height="820" fill="#08111f"/><rect x="35" y="35" width="1330" height="750" rx="28" fill="#101c2e" stroke="#29415f" stroke-width="2"/>
<text x="75" y="105" font-family="Segoe UI,sans-serif" font-size="42" font-weight="700" fill="#f8fafc">X25 rigid-footprint CARLA C3</text>
<text x="75" y="158" font-family="Segoe UI,sans-serif" font-size="25" fill="{accent}">{decision}</text>
<text x="75" y="235" font-family="Segoe UI,sans-serif" font-size="25" fill="#93c5fd">Frame F1</text>
<text x="75" y="285" font-family="Segoe UI,sans-serif" font-size="42" font-weight="700" fill="#cbd5e1">X24 {x24['f1']:.3f}</text>
<text x="340" y="285" font-family="Segoe UI,sans-serif" font-size="42" font-weight="700" fill="{accent}">X25 {x25['f1']:.3f}</text>
<text x="75" y="365" font-family="Segoe UI,sans-serif" font-size="25" fill="#93c5fd">Fresh edge CONTACT recovery</text>
<text x="75" y="415" font-family="Segoe UI,sans-serif" font-size="31" fill="#f8fafc">Bus: X24 {str(bus[ARM_X24]['event_detected_before_contact']).lower()} → X25 {str(bus[ARM_X25]['event_detected_before_contact']).lower()}</text>
<text x="75" y="460" font-family="Segoe UI,sans-serif" font-size="31" fill="#f8fafc">Side-on bicycle: X24 {str(bike[ARM_X24]['event_detected_before_contact']).lower()} → X25 {str(bike[ARM_X25]['event_detected_before_contact']).lower()}</text>
<text x="75" y="540" font-family="Segoe UI,sans-serif" font-size="25" fill="#93c5fd">No-regression surfaces</text>
<text x="75" y="590" font-family="Segoe UI,sans-serif" font-size="31" fill="#f8fafc">Original occlusion: X24 {occlusion[ARM_X24]['covered_frames']}/{occlusion[ARM_X24]['total_frames']} → X25 {occlusion[ARM_X25]['covered_frames']}/{occlusion[ARM_X25]['total_frames']}</text>
<text x="75" y="635" font-family="Segoe UI,sans-serif" font-size="31" fill="#f8fafc">Fresh SAFE false-alert segments (X25): {safe_segments}</text>
<text x="75" y="720" font-family="Segoe UI,sans-serif" font-size="21" fill="#94a3b8">YOLO11n-seg + aligned RGB-D | rigid quantile OBB | translation registration | issued-plan sweep</text>
</svg>'''


def score(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    result_path = run_root / "result-x25.json"
    svg_path = run_root / "result-x25.svg"
    require(not result_path.exists() and not svg_path.exists(), "x25_score_outputs_exist")
    source_result = read_json(source_root / "result.json")
    require(source_result.get("status") == "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE", "source_incomplete")
    require(all(bool(value) for value in source_result["checks"].values()), "source_gate_failed")
    x24 = read_json(run_root / "predictions-x24.json")
    x25 = read_json(run_root / "predictions-x25.json")
    require(x24.get("schema") == X24_SCHEMA and x25.get("schema") == X25_SCHEMA, "prediction_schema")
    expected = set(CONTACT_EPISODES + SAFE_EPISODES)
    require(set(x24["episodes"]) == expected and set(x25["episodes"]) == expected, "prediction_episodes")

    evaluator_by_episode = {
        episode_id: read_jsonl(source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl")
        for episode_id in sorted(expected)
    }
    prediction_by_arm = {
        ARM_X24: {episode_id: arm_frames(x24, episode_id, ARM_X24) for episode_id in sorted(expected)},
        ARM_X25: {episode_id: arm_frames(x25, episode_id, ARM_X25) for episode_id in sorted(expected)},
    }
    for arm, episodes in prediction_by_arm.items():
        for episode_id in expected:
            align(evaluator_by_episode[episode_id], episodes[episode_id], f"{arm}:{episode_id}")

    aggregate = {
        arm: confusion(evaluator_by_episode, episodes)
        for arm, episodes in prediction_by_arm.items()
    }
    contacts = {
        episode_id: {
            arm: contact_metrics(evaluator_by_episode[episode_id], episodes[episode_id])
            for arm, episodes in prediction_by_arm.items()
        }
        for episode_id in CONTACT_EPISODES
    }
    safe = {
        episode_id: {
            arm: false_segments(episodes[episode_id], 2.70 if episode_id == "ep_02" else 0.0)
            for arm, episodes in prediction_by_arm.items()
        }
        for episode_id in SAFE_EPISODES
    }
    edge_pairs = {**{key: contacts[key] for key in EDGE_CONTACT_EPISODES}, **{key: safe[key] for key in EDGE_SAFE_EPISODES}}
    occlusion = occlusion_coverage(
        read_json(source_root / "evaluator" / "physical_occlusion_report.json"),
        prediction_by_arm,
    )
    x24_edge = sum(contacts[key][ARM_X24]["event_detected_before_contact"] for key in EDGE_CONTACT_EPISODES)
    x25_edge = sum(contacts[key][ARM_X25]["event_detected_before_contact"] for key in EDGE_CONTACT_EPISODES)
    x25_edge_safe_segments = sum(safe[key][ARM_X25]["false_alert_segment_count"] for key in EDGE_SAFE_EPISODES)
    gate_checks = {
        "x25_recovers_both_fresh_edge_contacts": x25_edge == len(EDGE_CONTACT_EPISODES),
        "x25_recovers_more_fresh_edge_contacts_than_x24": x25_edge > x24_edge,
        "x25_has_zero_fresh_edge_safe_false_alert_segments": x25_edge_safe_segments == 0,
        "x25_frame_f1_exceeds_x24": aggregate[ARM_X25]["f1"] > aggregate[ARM_X24]["f1"] + EPSILON,
        "x25_retains_original_contact_occlusion_coverage": (
            occlusion[ARM_X25]["coverage"] + EPSILON >= occlusion[ARM_X24]["coverage"]
        ),
    }
    gate_met = all(gate_checks.values())
    decision = (
        "DTR_CARLA_X25_RIGID_FOOTPRINT_DEVELOPMENT_GATE_MET"
        if gate_met
        else "DTR_CARLA_X25_RIGID_FOOTPRINT_DEVELOPMENT_GATE_NOT_MET"
    )
    result = {
        "schema": "blindassist-dtr-carla-c3-x25-score-result-v1",
        "status": "COMPLETE",
        "decision": decision,
        "gate_met": gate_met,
        "gate_checks": gate_checks,
        "aggregate": aggregate,
        "contacts": contacts,
        "safe": safe,
        "edge_pairs": edge_pairs,
        "occlusion": occlusion,
        "deltas": {
            "frame_f1": aggregate[ARM_X25]["f1"] - aggregate[ARM_X24]["f1"],
            "frame_recall": aggregate[ARM_X25]["recall"] - aggregate[ARM_X24]["recall"],
            "fresh_edge_contacts_detected": x25_edge - x24_edge,
        },
        "source": {
            "source_result_sha256": sha256_file(source_root / "result.json"),
            "x24_predictions_sha256": sha256_file(run_root / "predictions-x24.json"),
            "x25_predictions_sha256": sha256_file(run_root / "predictions-x25.json"),
            "x25_freeze_sha256": sha256_file(run_root / "freeze-x25.json"),
            "scorer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "same_source_scripted_carla_development": True,
            "source_disjoint_confirmation": False,
            "real_world_confirmation": False,
        },
    }
    write_json_exclusive(result_path, result)
    write_exclusive(svg_path, svg_board(result).encode("utf-8"))
    return {
        **result,
        "result_sha256": sha256_file(result_path),
        "svg_sha256": sha256_file(svg_path),
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
