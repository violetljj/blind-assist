"""Typed public API for GOAL-COPILOT-1 candidates.

Observations contain only candidate-visible evidence. Scenario truth, transition
graphs, expected actions, completion truth, and evaluator state are intentionally
absent from this module and from the exported SearchTaskBundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskFamily(str, Enum):
    FIND_AND_REACH = "FIND_AND_REACH"
    TRACK_AND_REACQUIRE = "TRACK_AND_REACQUIRE"
    FIND_ALIGN_INTERACT = "FIND_ALIGN_INTERACT"


class Action(str, Enum):
    SCAN_LEFT = "SCAN_LEFT"
    SCAN_RIGHT = "SCAN_RIGHT"
    ALIGN_LEFT = "ALIGN_LEFT"
    ALIGN_RIGHT = "ALIGN_RIGHT"
    FORWARD = "FORWARD"
    INTERACT = "INTERACT"
    STOP = "STOP"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class Observation:
    target_visible: bool
    target_bearing: float | None
    target_relative_scale: float | None
    target_confidence: float
    forward_free: bool
    left_free: bool
    right_free: bool
    relative_nearness: float | None
    approach_rate: float | None
    tracking_quality: float
    observation_quality: float
    interaction_ready: bool


CANDIDATE_FUNCTIONS = (
    "update_task_belief",
    "propose_actions",
    "select_action",
    "detect_progress",
    "recover_target",
    "decide_completion",
)
