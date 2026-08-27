from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from l10_r0 import Action, Candidate, CONTROLLERS, Decision, State


GOALS = (
    "ROOM 203",
    "EXIT",
    "INFORMATION",
    "LIFT 4",
    "SERVICE DESK",
)


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    goal: str
    target_present: bool
    target_bearing: float
    bearing_jump: float
    start_distance: float
    occlusion_start: int
    occlusion_end: int
    distractor_bearings: tuple[float, ...]


class GoalLockWorld:
    fov_half_angle = 0.63

    def __init__(self, spec: EpisodeSpec):
        self.spec = spec
        self.camera_yaw = 0.0
        self.distance = spec.start_distance
        self.step_index = 0

    def _rng(self) -> random.Random:
        return random.Random(self.spec.seed * 1_000_003 + self.step_index * 9_176 + 41)

    def _target_bearing(self) -> float:
        jump = self.spec.bearing_jump if self.step_index >= self.spec.occlusion_end else 0.0
        return self.spec.target_bearing + jump

    @staticmethod
    def _score(rng: random.Random, mean: float, spread: float = 0.10) -> float:
        return max(0.02, min(0.98, rng.gauss(mean, spread)))

    def observe(self) -> tuple[list[Candidate], str | None, float | None]:
        rng = self._rng()
        candidates: list[Candidate] = []
        truth_id: str | None = None
        relative_target: float | None = None
        target_occluded = self.spec.occlusion_start <= self.step_index < self.spec.occlusion_end
        if self.spec.target_present:
            relative_target = self._target_bearing() - self.camera_yaw
            target_visible = abs(relative_target) <= self.fov_half_angle and not target_occluded
            # A real spotter is intermittent under blur/perspective; easy text is not perfect text.
            target_detected = target_visible and rng.random() >= (0.14 if self.distance > 3.0 else 0.08)
            if target_detected:
                segment = 0 if self.step_index < self.spec.occlusion_start else 1
                truth_id = f"p{segment}-{self.spec.seed % 97}"
                normalized_x = max(-1.0, min(1.0, relative_target / self.fov_half_angle))
                scale = max(0.10, min(0.93, 1.18 / self.distance + 0.05 + rng.gauss(0.0, 0.025)))
                text_mean = 0.84 if rng.random() > 0.18 else 0.43
                candidates.append(
                    Candidate(
                        proposal_id=truth_id,
                        center_x=normalized_x,
                        scale=scale,
                        text_score=self._score(rng, text_mean, 0.10),
                        appearance_score=self._score(rng, 0.78, 0.09),
                        structure_score=self._score(rng, 0.80, 0.08),
                        completion_score=self._score(rng, 0.82 if self.distance <= 1.65 else 0.25, 0.10),
                    )
                )

        for index, world_bearing in enumerate(self.spec.distractor_bearings):
            relative = world_bearing - self.camera_yaw
            if abs(relative) > self.fov_half_angle or rng.random() < 0.18:
                continue
            kind = index % 3
            if kind == 0:  # text lookalike: strong OCR fragment, weak instance appearance
                text, appearance, structure = 0.70, 0.42, 0.58
            elif kind == 1:  # appearance lookalike: similar sign design, wrong wording
                text, appearance, structure = 0.35, 0.72, 0.70
            else:  # transient joint ambiguity, deliberately hard
                text, appearance, structure = 0.59, 0.60, 0.58
            false_near = (self.spec.seed + index) % 11 == 0
            candidates.append(
                Candidate(
                    proposal_id=f"d{index}-{self.spec.seed % 31}",
                    center_x=max(-1.0, min(1.0, relative / self.fov_half_angle)),
                    scale=self._score(rng, 0.76 if false_near else 0.36, 0.06),
                    text_score=self._score(rng, text, 0.10),
                    appearance_score=self._score(rng, appearance, 0.09),
                    structure_score=self._score(rng, structure, 0.08),
                    completion_score=self._score(rng, 0.68 if false_near else 0.18, 0.10),
                )
            )
        rng.shuffle(candidates)
        return candidates, truth_id, relative_target

    def advance(self, action: Action) -> None:
        if action is Action.LEFT:
            self.camera_yaw -= 0.115
        elif action is Action.RIGHT:
            self.camera_yaw += 0.115
        elif action is Action.FORWARD:
            self.distance = max(0.85, self.distance - 0.34)
        self.step_index += 1

    def completion_is_true(self, decision: Decision, truth_id: str | None) -> bool:
        if not self.spec.target_present or decision.selected_id != truth_id:
            return False
        relative = self._target_bearing() - self.camera_yaw
        return self.distance <= 1.65 and abs(relative) <= 0.14


def make_specs(episodes: int, seed: int) -> list[EpisodeSpec]:
    specs = []
    for index in range(episodes):
        rng = random.Random(seed + index * 7_919)
        target_present = index % 5 != 0
        bearing = rng.uniform(-0.50, 0.50)
        distractors = tuple(rng.uniform(-0.95, 0.95) for _ in range(3 + index % 3))
        specs.append(
            EpisodeSpec(
                seed=seed + index * 7_919,
                goal=GOALS[index % len(GOALS)],
                target_present=target_present,
                target_bearing=bearing,
                bearing_jump=rng.choice((-1.0, 1.0)) * rng.uniform(0.20, 0.46),
                start_distance=rng.uniform(6.6, 9.2),
                occlusion_start=rng.randint(17, 21),
                occlusion_end=rng.randint(25, 30),
                distractor_bearings=distractors,
            )
        )
    return specs


def correct_direction(relative_target: float | None, action: Action) -> bool | None:
    if relative_target is None:
        return None
    if relative_target < -0.10:
        return action is Action.LEFT
    if relative_target > 0.10:
        return action is Action.RIGHT
    return action is Action.FORWARD


def run_episode(controller: Any, spec: EpisodeSpec, max_steps: int) -> dict[str, Any]:
    controller.reset()
    world = GoalLockWorld(spec)
    completed = False
    false_complete = False
    completion_step: int | None = None
    wrong_lock_frames = 0
    true_lock_before_occlusion = False
    reacquired_step: int | None = None
    direction_correct = 0
    direction_total = 0
    lock_frames = 0
    state_switches = 0
    previous_state = State.SEARCH

    for step in range(max_steps):
        candidates, truth_id, relative_target = world.observe()
        decision = controller.step(candidates)
        if decision.state != previous_state:
            state_switches += 1
            previous_state = decision.state
        selected_true = truth_id is not None and decision.selected_id == truth_id
        locked = decision.state in {State.LOCKED, State.NEAR, State.TASK_COMPLETE}
        if locked:
            lock_frames += 1
            if decision.selected_id is not None and not selected_true:
                wrong_lock_frames += 1
        if step < spec.occlusion_start and selected_true and locked:
            true_lock_before_occlusion = True
        if (
            step >= spec.occlusion_end
            and true_lock_before_occlusion
            and reacquired_step is None
            and selected_true
            and locked
        ):
            reacquired_step = step
        direction_ok = correct_direction(relative_target if truth_id else None, decision.action)
        if direction_ok is not None and decision.action is not Action.COMPLETE:
            direction_total += 1
            direction_correct += int(direction_ok)
        if decision.action is Action.COMPLETE:
            completed = world.completion_is_true(decision, truth_id)
            false_complete = not completed
            completion_step = step
            break
        world.advance(decision.action)

    reacquire_eligible = spec.target_present and true_lock_before_occlusion
    reacquired = reacquired_step is not None and reacquired_step - spec.occlusion_end <= 12
    return {
        "target_present": spec.target_present,
        "completed": completed,
        "false_complete": false_complete,
        "completion_step": completion_step,
        "wrong_lock_frames": wrong_lock_frames,
        "lock_frames": lock_frames,
        "reacquire_eligible": reacquire_eligible,
        "reacquired": reacquired,
        "reacquire_latency": None if reacquired_step is None else reacquired_step - spec.occlusion_end,
        "direction_correct": direction_correct,
        "direction_total": direction_total,
        "state_switches": state_switches,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in rows if row["target_present"]]
    absent = [row for row in rows if not row["target_present"]]
    completed_steps = [row["completion_step"] for row in present if row["completed"]]
    reacquire = [row for row in rows if row["reacquire_eligible"]]
    latencies = [row["reacquire_latency"] for row in reacquire if row["reacquired"]]
    total_lock_frames = sum(row["lock_frames"] for row in rows)
    completed_count = sum(row["completed"] for row in present)
    false_complete_count = sum(row["false_complete"] for row in absent)
    reacquired_count = sum(row["reacquired"] for row in reacquire)
    wrong_lock_frames = sum(row["wrong_lock_frames"] for row in rows)
    return {
        "present_episodes": len(present),
        "absent_episodes": len(absent),
        "completed_episodes": completed_count,
        "task_success_rate": _rate(completed_count, len(present)),
        "absent_false_completions": false_complete_count,
        "absent_false_complete_rate": _rate(false_complete_count, len(absent)),
        "median_steps_to_complete": round(statistics.median(completed_steps), 1) if completed_steps else None,
        "reacquire_eligible_episodes": len(reacquire),
        "reacquired_episodes": reacquired_count,
        "reacquire_success_rate": _rate(reacquired_count, len(reacquire)),
        "median_reacquire_frames": round(statistics.median(latencies), 1) if latencies else None,
        "direction_accuracy": _rate(
            sum(row["direction_correct"] for row in rows),
            sum(row["direction_total"] for row in rows),
        ),
        "wrong_lock_frames": wrong_lock_frames,
        "lock_frames": total_lock_frames,
        "wrong_lock_frame_rate": _rate(wrong_lock_frames, total_lock_frames),
        "mean_state_switches": round(statistics.mean(row["state_switches"] for row in rows), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the L10-R0 controlled closed-loop benchmark.")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 25:
        parser.error("--episodes must be at least 25 so present/absent strata are represented")

    specs = make_specs(args.episodes, args.seed)
    arms: dict[str, Any] = {}
    for controller_type in CONTROLLERS:
        controller = controller_type()
        rows = [run_episode(controller, spec, args.max_steps) for spec in specs]
        arms[controller.name] = summarize(rows)
    result = {
        "experiment": "L10-R0-GOAL-LOCK-CONTROLLED-V1",
        "claim_ceiling": "CONTROLLED_SYNTHETIC_CLOSED_LOOP_MECHANICS_ONLY",
        "result_status": "DEVELOPMENT_MECHANISM_RESULT",
        "episode_count": args.episodes,
        "seed": args.seed,
        "goal_classes": list(GOALS),
        "arms": arms,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
