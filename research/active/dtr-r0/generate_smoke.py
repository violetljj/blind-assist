"""Generate deterministic DTR-R0 mechanism-smoke episodes.

The generator derives evaluator truth from complete clean trajectories first.
It then emits only noisy current/past observations for the algorithm.  These
episodes are deliberately synthetic and cannot adjudicate the research gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Callable

from dtr_r0 import EgoPose, Vec2


SCENE_TYPES = (
    "crossing_enters_route",
    "oncoming",
    "parallel_outside_route",
    "static_roadside",
    "ego_turn_pseudo_motion",
    "enter_then_exit",
)


def _rounded(value: float) -> float:
    return round(value, 6)


def _timeline(duration_s: float, step_s: float) -> list[float]:
    count = int(round(duration_s / step_s))
    return [_rounded(index * step_s) for index in range(count + 1)]


def _scene_functions(
    scene_type: str, episode_index: int
) -> tuple[Callable[[float], Vec2], Callable[[float], Vec2], Callable[[float], float]]:
    phase = (episode_index % 4) - 1.5
    ego_speed = 1.00 + 0.025 * phase
    encounter_s = 3.70 + 0.10 * phase

    def ego_position(time_s: float) -> Vec2:
        return Vec2(ego_speed * time_s, 0.0)

    if scene_type == "crossing_enters_route":
        crossing_speed = 0.72 + 0.02 * phase
        encounter_x = ego_speed * encounter_s

        def target_position(time_s: float) -> Vec2:
            return Vec2(encounter_x, crossing_speed * (encounter_s - time_s))

    elif scene_type == "oncoming":
        target_speed = -0.90 - 0.025 * phase
        initial_x = (ego_speed - target_speed) * encounter_s

        def target_position(time_s: float) -> Vec2:
            return Vec2(initial_x + target_speed * time_s, 0.0)

    elif scene_type == "parallel_outside_route":
        offset = 2.10 + 0.05 * (episode_index % 3)

        def target_position(time_s: float) -> Vec2:
            return Vec2(0.60 + ego_speed * time_s, offset)

    elif scene_type == "static_roadside":
        roadside_x = ego_speed * encounter_s
        roadside_y = 2.10 + 0.05 * (episode_index % 3)

        def target_position(time_s: float) -> Vec2:
            del time_s
            return Vec2(roadside_x, roadside_y)

    elif scene_type == "ego_turn_pseudo_motion":
        roadside_x = ego_speed * encounter_s
        roadside_y = 2.10 + 0.05 * (episode_index % 3)

        def target_position(time_s: float) -> Vec2:
            del time_s
            return Vec2(roadside_x, roadside_y)

    elif scene_type == "enter_then_exit":
        crossing_speed = 1.05 + 0.025 * phase
        encounter_x = ego_speed * encounter_s

        def target_position(time_s: float) -> Vec2:
            return Vec2(encounter_x, crossing_speed * (encounter_s - time_s))

    else:
        raise ValueError(f"unknown scene type: {scene_type}")

    if scene_type == "ego_turn_pseudo_motion":

        def sensor_yaw(time_s: float) -> float:
            # A large camera sweep while body route yaw remains exactly zero.
            return 0.90 * math.sin((time_s - 2.5) * 1.15)

    else:

        def sensor_yaw(time_s: float) -> float:
            del time_s
            return 0.0

    return ego_position, target_position, sensor_yaw


def _truth_from_complete_clean_trajectory(
    ego_position: Callable[[float], Vec2],
    target_position: Callable[[float], Vec2],
    *,
    duration_s: float,
    truth_contact_radius_m: float = 0.75,
) -> dict[str, object]:
    contact_times: list[float] = []
    dense_step_s = 0.01
    for time_s in _timeline(duration_s, dense_step_s):
        if (target_position(time_s) - ego_position(time_s)).norm() <= truth_contact_radius_m:
            contact_times.append(time_s)
    if not contact_times:
        return {
            "critical_event": False,
            "event_start_s": None,
            "event_end_s": None,
            "warning_start_s": None,
            "warning_end_s": None,
            "exit_time_s": None,
            "truth_source": "complete_clean_trajectory_before_observation_noise",
            "truth_contact_radius_m": truth_contact_radius_m,
        }
    event_start_s = min(contact_times)
    event_end_s = max(contact_times)
    return {
        "critical_event": True,
        "event_start_s": event_start_s,
        "event_end_s": event_end_s,
        "warning_start_s": max(0.0, event_start_s - 3.0),
        "warning_end_s": event_start_s,
        "exit_time_s": event_end_s,
        "truth_source": "complete_clean_trajectory_before_observation_noise",
        "truth_contact_radius_m": truth_contact_radius_m,
    }


def generate_episode(
    scene_type: str,
    episode_index: int,
    *,
    seed: int,
    duration_s: float = 7.0,
    step_s: float = 0.25,
) -> dict[str, object]:
    if scene_type not in SCENE_TYPES:
        raise ValueError(f"unknown scene type: {scene_type}")
    rng = random.Random(seed + SCENE_TYPES.index(scene_type) * 10_000 + episode_index)
    ego_position, target_position, sensor_yaw = _scene_functions(
        scene_type, episode_index
    )
    truth = _truth_from_complete_clean_trajectory(
        ego_position, target_position, duration_s=duration_s
    )

    frames: list[dict[str, object]] = []
    for time_s in _timeline(duration_s, step_s):
        clean_ego = ego_position(time_s)
        clean_pose = EgoPose(
            x_m=clean_ego.x,
            y_m=clean_ego.y,
            body_yaw_rad=0.0,
            sensor_yaw_rad=sensor_yaw(time_s),
        )
        forward_m, left_m = clean_pose.world_to_local(target_position(time_s))

        # Observation and pose noise are independent draws.  The algorithm is
        # never given the clean coordinates or a future target sample.
        observed_pose = {
            "x_m": _rounded(clean_ego.x + rng.gauss(0.0, 0.012)),
            "y_m": _rounded(clean_ego.y + rng.gauss(0.0, 0.012)),
            "body_yaw_rad": _rounded(rng.gauss(0.0, 0.004)),
            "sensor_yaw_rad": _rounded(
                sensor_yaw(time_s) + rng.gauss(0.0, 0.004)
            ),
        }
        frames.append(
            {
                "time_s": time_s,
                "ego_pose": observed_pose,
                "observations": [
                    {
                        "track_id": "target-0",
                        "forward_m": _rounded(forward_m + rng.gauss(0.0, 0.025)),
                        "left_m": _rounded(left_m + rng.gauss(0.0, 0.025)),
                        "radius_m": 0.30,
                    }
                ],
            }
        )

    return {
        "schema_version": "dtr-r0-episode-v1",
        "episode_id": f"{scene_type}-{episode_index:03d}",
        "scene_type": scene_type,
        "mechanism_smoke_only": True,
        "frames": frames,
        "truth": truth,
    }


def generate_episodes(
    *, episodes_per_class: int = 4, seed: int = 1701
) -> list[dict[str, object]]:
    if episodes_per_class < 1:
        raise ValueError("episodes_per_class must be at least one")
    return [
        generate_episode(scene_type, index, seed=seed)
        for scene_type in SCENE_TYPES
        for index in range(episodes_per_class)
    ]


def write_jsonl(path: Path, episodes: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    episodes = generate_episodes(
        episodes_per_class=args.episodes_per_class, seed=args.seed
    )
    write_jsonl(args.output, episodes)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episode_count": len(episodes),
                "scene_types": list(SCENE_TYPES),
                "claim_ceiling": "CONTROLLED_SYNTHETIC_MECHANICS_ONLY",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
