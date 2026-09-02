from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Any, Mapping


FORBIDDEN_POLICY_FIELDS = frozenset(
    {
        "target_position",
        "distance_to_goal",
        "goal_world",
        "occ_map",
        "height_map",
        "extra",
        "reference_trajectory",
        "endpoint",
        "end_point",
        "facade_id",
        "entrance_id",
        "target_present",
        "handoff_truth",
    }
)


@dataclass(frozen=True)
class TruthFreePoiObservation:
    poi_name: str
    images: Mapping[str, Any]
    position: Any
    rotation: Any
    heading: float
    step_count: int
    history_images: Any = None
    history_poses: Any = None


@dataclass(frozen=True)
class ActionReceipt:
    episode_id: str
    step_count: int
    arm: str
    action_label: str
    requested_waypoint: tuple[float, float]
    before_pose_digest: str
    after_pose_digest: str
    before_image_digest: str
    after_image_digest: str
    receipt_digest: str


def _read(observation: Any, field: str, default: Any = None) -> Any:
    if isinstance(observation, Mapping):
        return observation.get(field, default)
    return getattr(observation, field, default)


def strip_policy_truth(observation: Any) -> TruthFreePoiObservation:
    """Project an ABotN observation onto the frozen truth-free policy view."""

    return TruthFreePoiObservation(
        poi_name=str(_read(observation, "poi_name", "")),
        images=_read(observation, "images", {}),
        position=_read(observation, "position"),
        rotation=_read(observation, "rotation"),
        heading=float(_read(observation, "heading", 0.0)),
        step_count=int(_read(observation, "step_count", 0)),
        history_images=_read(observation, "history_images"),
        history_poses=_read(observation, "history_poses"),
    )


def policy_field_names(observation: TruthFreePoiObservation) -> frozenset[str]:
    return frozenset(field.name for field in fields(observation))


def assert_truth_free(observation: TruthFreePoiObservation) -> None:
    leaked = policy_field_names(observation) & FORBIDDEN_POLICY_FIELDS
    if leaked:
        raise ValueError(f"truth-bearing policy fields: {sorted(leaked)}")


def _digest(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif hasattr(value, "tobytes"):
        payload = value.tobytes()
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def make_action_receipt(
    *,
    episode_id: str,
    step_count: int,
    arm: str,
    action_label: str,
    requested_waypoint: tuple[float, float],
    before_pose: Any,
    after_pose: Any,
    before_image: Any,
    after_image: Any,
) -> ActionReceipt:
    core = {
        "episode_id": episode_id,
        "step_count": step_count,
        "arm": arm,
        "action_label": action_label,
        "requested_waypoint": requested_waypoint,
        "before_pose_digest": _digest(before_pose),
        "after_pose_digest": _digest(after_pose),
        "before_image_digest": _digest(before_image),
        "after_image_digest": _digest(after_image),
    }
    return ActionReceipt(
        **core,
        receipt_digest=_digest(core),
    )
