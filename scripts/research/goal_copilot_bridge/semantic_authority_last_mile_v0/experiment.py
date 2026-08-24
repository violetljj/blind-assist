"""Controlled SAGE-LM V0 mechanism experiment.

Identity is supplied by an exact semantic anchor and is never inferred by the
geometry policy.  The baseline servos to the anchor bbox centre/scale.  The
challenger uses a short active-parallax observation to recover the target
aperture and requires temporally consistent geometric arrival evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "semantic_authority_conditioned_last_mile_geometry_v0"
KINDS = ("ROOM_SIGN", "QR_ENTRANCE", "EXACT_SHELF_TARGET")


@dataclass(frozen=True)
class Episode:
    episode_id: str
    kind: str
    aperture_x_m: float
    aperture_width_m: float
    anchor_x_m: float
    start_x_m: float
    start_range_m: float
    anchor_height_m: float
    occlusion_steps: tuple[int, ...]
    observation_noise_m: float


@dataclass
class Pose:
    x_m: float
    range_m: float


def build_cohort(seed: int = 240824, per_kind: int = 12) -> list[Episode]:
    rng = random.Random(seed)
    episodes: list[Episode] = []
    for kind in KINDS:
        for index in range(per_kind):
            aperture_x = rng.uniform(-0.9, 0.9)
            width = rng.uniform(1.00, 1.45) if kind != "EXACT_SHELF_TARGET" else rng.uniform(0.65, 0.90)
            if index % 3 == 0:
                # Controls where the semantic carrier is already a reasonable
                # proxy for the physical approach region.
                offset_magnitude = rng.uniform(0.04, width * 0.28)
            else:
                offset_magnitude = rng.uniform(width * 0.52, width * 0.82)
            offset = rng.choice((-1.0, 1.0)) * offset_magnitude
            anchor_x = aperture_x + offset
            start_x = rng.uniform(-1.1, 1.1)
            occlusion = (5, 6) if index % 3 == 0 else ()
            episodes.append(
                Episode(
                    episode_id=f"{kind.lower()}-{index + 1:02d}",
                    kind=kind,
                    aperture_x_m=round(aperture_x, 4),
                    aperture_width_m=round(width, 4),
                    anchor_x_m=round(anchor_x, 4),
                    start_x_m=round(start_x, 4),
                    start_range_m=round(rng.uniform(6.0, 8.5), 4),
                    anchor_height_m=0.22 if kind != "QR_ENTRANCE" else 0.16,
                    occlusion_steps=occlusion,
                    observation_noise_m=round(rng.uniform(0.020, 0.075), 4),
                )
            )
    return episodes


def _direction(delta_x: float) -> str:
    if delta_x < -0.14:
        return "LEFT"
    if delta_x > 0.14:
        return "RIGHT"
    return "FORWARD"


def _truth_arrived(episode: Episode, pose: Pose) -> bool:
    usable_half_width = episode.aperture_width_m / 2.0 - 0.12
    return pose.range_m <= 0.82 and abs(pose.x_m - episode.aperture_x_m) <= usable_half_width


def _advance(pose: Pose, target_x: float) -> None:
    pose.x_m += max(-0.28, min(0.28, target_x - pose.x_m))
    pose.range_m = max(0.45, pose.range_m - 0.54)


def _run_baseline(episode: Episode, seed: int) -> dict:
    rng = random.Random(seed)
    pose = Pose(episode.start_x_m, episode.start_range_m)
    path = [[pose.x_m, pose.range_m]]
    correct_directions = 0
    direction_count = 0
    completion = False
    completion_step = None
    lost_events = 0
    reacquisitions = 0
    movement_during_lost = 0
    was_lost = False

    for step in range(22):
        visible = step not in episode.occlusion_steps
        if not visible:
            if not was_lost:
                lost_events += 1
            was_lost = True
            # Bbox servo has no current target and keeps its last command.
            target_x = episode.anchor_x_m
            movement_during_lost += 1
        else:
            if was_lost:
                reacquisitions += 1
            was_lost = False
            target_x = episode.anchor_x_m + rng.gauss(0.0, episode.observation_noise_m)

        command = _direction(target_x - pose.x_m)
        oracle = _direction(episode.aperture_x_m - pose.x_m)
        correct_directions += int(command == oracle)
        direction_count += 1
        _advance(pose, target_x)
        path.append([pose.x_m, pose.range_m])

        apparent_anchor_height = episode.anchor_height_m / max(pose.range_m, 0.1)
        if apparent_anchor_height >= 0.19:
            completion = True
            completion_step = step
            break

    arrived = _truth_arrived(episode, pose)
    return {
        "arm": "BBOX_CENTER_SCALE",
        "completion": completion,
        "completion_step": completion_step,
        "true_arrival": arrived,
        "premature_arrival": completion and not arrived,
        "endpoint_lateral_error_m": abs(pose.x_m - episode.aperture_x_m),
        "direction_correct": correct_directions,
        "direction_count": direction_count,
        "lost_events": lost_events,
        "reacquisitions": reacquisitions,
        "movement_during_lost": movement_during_lost,
        "path": path,
    }


def _active_aperture_estimate(episode: Episode, rng: random.Random) -> tuple[float, float]:
    """Return aperture centre and confidence from a controlled parallax pair.

    The simulator exposes noisy left/right boundary bearings from two camera
    positions separated by 0.24 m.  This function represents the triangulation
    output; it never receives or changes the semantic identity label.
    """
    baseline_m = 0.24
    camera_left = episode.start_x_m - baseline_m / 2.0
    camera_right = episode.start_x_m + baseline_m / 2.0
    boundary_xs = (
        episode.aperture_x_m - episode.aperture_width_m / 2.0,
        episode.aperture_x_m + episode.aperture_width_m / 2.0,
    )
    estimates = []
    confidence_terms = []
    angular_sigma = episode.observation_noise_m / episode.start_range_m
    for boundary_x in boundary_xs:
        tangent_left = (boundary_x - camera_left) / episode.start_range_m + rng.gauss(0.0, angular_sigma)
        tangent_right = (boundary_x - camera_right) / episode.start_range_m + rng.gauss(0.0, angular_sigma)
        disparity = tangent_left - tangent_right
        if disparity <= 1e-4:
            return episode.start_x_m, 0.0
        estimated_range = baseline_m / disparity
        left_estimate = camera_left + estimated_range * tangent_left
        right_estimate = camera_right + estimated_range * tangent_right
        estimates.append((left_estimate + right_estimate) / 2.0)
        expected_disparity = baseline_m / episode.start_range_m
        relative_error = abs(disparity - expected_disparity) / expected_disparity
        confidence_terms.append(max(0.0, 1.0 - relative_error))
    centre = sum(estimates) / 2.0
    width_estimate = estimates[1] - estimates[0]
    width_consistency = max(0.0, 1.0 - abs(width_estimate - episode.aperture_width_m) / episode.aperture_width_m)
    confidence = max(0.0, min(1.0, min(confidence_terms) * width_consistency))
    return centre, confidence


def _run_sage_lm(episode: Episode, seed: int) -> dict:
    rng = random.Random(seed)
    pose = Pose(episode.start_x_m, episode.start_range_m)
    path = [[pose.x_m, pose.range_m]]
    target_x, geometry_confidence = _active_aperture_estimate(episode, rng)
    correct_directions = 0
    direction_count = 0
    completion = False
    completion_step = None
    lost_events = 0
    reacquisitions = 0
    movement_during_lost = 0
    was_lost = False
    arrival_support = 0

    # The first action actively creates parallax instead of claiming identity.
    pose.x_m += 0.12 if pose.x_m <= target_x else -0.12
    path.append([pose.x_m, pose.range_m])

    for step in range(1, 28):
        visible = step not in episode.occlusion_steps
        if not visible:
            if not was_lost:
                lost_events += 1
            was_lost = True
            arrival_support = 0
            path.append([pose.x_m, pose.range_m])
            continue
        if was_lost:
            reacquisitions += 1
            # A fresh exact anchor re-authorizes the same aperture estimate.
            target_x, geometry_confidence = _active_aperture_estimate(episode, rng)
        was_lost = False

        command = _direction(target_x - pose.x_m)
        oracle = _direction(episode.aperture_x_m - pose.x_m)
        correct_directions += int(command == oracle)
        direction_count += 1
        _advance(pose, target_x)
        path.append([pose.x_m, pose.range_m])

        near = pose.range_m <= 0.82
        aligned = abs(pose.x_m - target_x) <= max(0.12, episode.aperture_width_m * 0.22)
        aperture_supported = geometry_confidence >= 0.35
        arrival_support = arrival_support + 1 if near and aligned and aperture_supported else 0
        if arrival_support >= 2:
            completion = True
            completion_step = step
            break

    arrived = _truth_arrived(episode, pose)
    return {
        "arm": "SAGE_LM_ACTIVE_APERTURE_PROGRESS",
        "completion": completion,
        "completion_step": completion_step,
        "true_arrival": arrived,
        "premature_arrival": completion and not arrived,
        "endpoint_lateral_error_m": abs(pose.x_m - episode.aperture_x_m),
        "direction_correct": correct_directions,
        "direction_count": direction_count,
        "lost_events": lost_events,
        "reacquisitions": reacquisitions,
        "movement_during_lost": movement_during_lost,
        "geometry_confidence": geometry_confidence,
        "path": path,
    }


def _aggregate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    completions = sum(int(row["completion"]) for row in rows)
    true_arrivals = sum(int(row["true_arrival"]) for row in rows)
    correct = sum(int(row["direction_correct"]) for row in rows)
    directions = sum(int(row["direction_count"]) for row in rows)
    lost = sum(int(row["lost_events"]) for row in rows)
    reacquired = sum(int(row["reacquisitions"]) for row in rows)
    errors = sorted(float(row["endpoint_lateral_error_m"]) for row in rows)
    return {
        "episode_count": len(rows),
        "direction_accuracy": correct / directions,
        "target_front_arrival_rate": true_arrivals / len(rows),
        "median_endpoint_lateral_error_m": (errors[len(errors) // 2 - 1] + errors[len(errors) // 2]) / 2,
        "completion_decisions": completions,
        "completion_precision": sum(int(row["completion"] and row["true_arrival"]) for row in rows) / completions if completions else None,
        "verified_completion_rate": sum(int(row["completion"] and row["true_arrival"]) for row in rows) / len(rows),
        "premature_arrival_count": sum(int(row["premature_arrival"]) for row in rows),
        "lost_event_count": lost,
        "lost_recovery_rate": reacquired / lost if lost else None,
        "movement_steps_while_lost": sum(int(row["movement_during_lost"]) for row in rows),
    }


def run_experiment(seed: int = 240824, per_kind: int = 12) -> dict:
    cohort = build_cohort(seed=seed, per_kind=per_kind)
    rows = []
    for index, episode in enumerate(cohort):
        baseline = _run_baseline(episode, seed + index * 17 + 1)
        challenger = _run_sage_lm(episode, seed + index * 17 + 2)
        rows.append({"episode": asdict(episode), "baseline": baseline, "sage_lm": challenger})
    baseline_rows = [row["baseline"] for row in rows]
    challenger_rows = [row["sage_lm"] for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "REVERSIBLE_EXPLORATION_CONTROLLED_SYNTHETIC_GEOMETRY",
        "seed": seed,
        "identity_contract": "EXACT_SEMANTIC_AUTHORITY_FIXED_GEOMETRY_CANNOT_REBIND",
        "cohort": {"episode_count": len(cohort), "kinds": {kind: per_kind for kind in KINDS}},
        "metrics": {
            "bbox_center_scale": _aggregate(baseline_rows),
            "sage_lm": _aggregate(challenger_rows),
        },
        "rows": rows,
        "claim_ceiling": "CONTROLLED_GEOMETRY_MECHANISM_EFFECT_ONLY",
    }


def _render_demo(report: dict, output_path: Path) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return

    width, height = 1500, 920
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    title = "SAGE-LM V0: semantic anchor bbox vs authority-conditioned aperture geometry"
    cv2.putText(canvas, title, (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2, cv2.LINE_AA)
    selected = [report["rows"][0], report["rows"][12], report["rows"][24]]
    panel_w, panel_h = 455, 345
    for column, row in enumerate(selected):
        episode = row["episode"]
        for arm_index, key in enumerate(("baseline", "sage_lm")):
            x0 = 28 + column * 490
            y0 = 90 + arm_index * 395
            arm = row[key]
            cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (210, 210, 210), 1)
            label = "bbox center + scale" if key == "baseline" else "SAGE-LM aperture + progress"
            color = (55, 75, 220) if key == "baseline" else (65, 155, 45)
            cv2.putText(canvas, f"{episode['kind']} | {label}", (x0 + 12, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            scale_x, scale_z = 105.0, 36.0
            origin_x, wall_y = x0 + panel_w // 2, y0 + panel_h - 32
            aperture_left = int(origin_x + (episode["aperture_x_m"] - episode["aperture_width_m"] / 2) * scale_x)
            aperture_right = int(origin_x + (episode["aperture_x_m"] + episode["aperture_width_m"] / 2) * scale_x)
            cv2.line(canvas, (x0 + 15, wall_y), (aperture_left, wall_y), (80, 80, 80), 5)
            cv2.line(canvas, (aperture_right, wall_y), (x0 + panel_w - 15, wall_y), (80, 80, 80), 5)
            cv2.line(canvas, (aperture_left, wall_y), (aperture_right, wall_y), (55, 180, 80), 2)
            anchor_px = int(origin_x + episode["anchor_x_m"] * scale_x)
            cv2.circle(canvas, (anchor_px, wall_y - 10), 6, (0, 150, 240), -1)
            points = []
            for px, pz in arm["path"]:
                points.append((int(origin_x + px * scale_x), int(wall_y - pz * scale_z)))
            for a, b in zip(points, points[1:]):
                cv2.line(canvas, a, b, color, 3, cv2.LINE_AA)
            cv2.circle(canvas, points[-1], 6, color, -1)
            result = "TRUE ARRIVAL" if arm["true_arrival"] else "FALSE / MISSED"
            cv2.putText(canvas, f"{result}; lateral error={arm['endpoint_lateral_error_m']:.2f}m", (x0 + 12, y0 + panel_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=240824)
    parser.add_argument("--per-kind", type=int, default=12)
    args = parser.parse_args()
    report = run_experiment(seed=args.seed, per_kind=args.per_kind)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _render_demo(report, args.output_dir / "trajectory_demo.png")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
