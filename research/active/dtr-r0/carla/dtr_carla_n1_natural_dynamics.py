"""Compile a deterministic, CARLA-independent N1 natural-dynamics world plan.

The compiler consumes a small JSON registry, derives independent subsystem
seeds from one master seed, and emits actor intents plus scheduled interactions.
It deliberately does not import CARLA or contact a running simulator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REGISTRY_SCHEMA_VERSION = "dtr-carla-n1-natural-dynamics-registry-v1"
PLAN_SCHEMA_VERSION = "dtr-carla-n1-natural-dynamics-plan-v1"
CARLA_MAP = "Carla/Maps/Town10HD_Opt"
SUPPORTED_CARLA_MAPS = (
    CARLA_MAP,
    "Carla/Maps/Town01",
    "Carla/Maps/Town04",
    "Carla/Maps/Town05",
)
SCENARIO_CLASS_PARAMETERS = {
    "construction_zone": {"chicane_width_m", "barrier_count"},
    "crowded_pedestrians": {
        "minimum_dynamic_pedestrians",
        "minimum_dynamic_targets",
    },
    "bus_stop": {"stop_length_m", "boarding_zone_width_m"},
    "parking_lot": {"parking_aisle_width_m", "parked_vehicle_rows"},
}
REQUIRED_DRIVING_PROFILES = ("cautious", "nominal", "assertive")
REQUIRED_EVENT_TYPES = (
    "occluded_jaywalk",
    "sudden_brake",
    "reverse_pullout",
    "door_open",
)
SUBSYSTEMS = ("traffic", "crowd", "grouping", "rerouting", "events")


class RegistrySchemaError(ValueError):
    """Raised when an N1 registry violates the supported schema."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistrySchemaError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    _require(
        actual_set == expected_set,
        f"{label} keys differ: missing={sorted(expected_set - actual_set)}, "
        f"unexpected={sorted(actual_set - expected_set)}",
    )


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be a finite number",
    )
    result = float(value)
    if minimum is not None:
        _require(result >= minimum, f"{label} must be >= {minimum}")
    return result


def _probability(value: Any, label: str) -> float:
    result = _number(value, label)
    _require(0.0 <= result <= 1.0, f"{label} must be in [0, 1]")
    return result


def _numeric_range(
    value: Any, label: str, *, minimum: float | None = None
) -> tuple[float, float]:
    items = _array(value, label)
    _require(len(items) == 2, f"{label} must contain exactly two numbers")
    low = _number(items[0], f"{label}[0]", minimum=minimum)
    high = _number(items[1], f"{label}[1]", minimum=minimum)
    _require(low <= high, f"{label} must be ordered low-to-high")
    return low, high


def _string_array(value: Any, label: str) -> list[str]:
    result = [_text(item, f"{label} item") for item in _array(value, label)]
    _require(bool(result), f"{label} must not be empty")
    _require(len(result) == len(set(result)), f"{label} contains duplicates")
    return result


def _vector2(value: Any, label: str) -> tuple[float, float]:
    items = _array(value, label)
    _require(len(items) == 2, f"{label} must contain two numbers")
    return (_number(items[0], f"{label}[0]"), _number(items[1], f"{label}[1]"))


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object, translating parse failures into schema errors."""

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistrySchemaError(f"registry is not readable JSON: {path}: {exc}") from exc
    return _object(value, "registry")


def validate_registry(registry: Mapping[str, Any]) -> None:
    """Validate the complete, intentionally narrow N1 registry schema."""

    registry = _object(registry, "registry")
    _exact_keys(
        registry,
        {
            "schema_version",
            "registry_id",
            "purpose",
            "carla",
            "focus",
            "duration_seconds",
            "seed_contract",
            "traffic_profiles",
            "crowd",
            "grouping",
            "rerouting",
            "long_tail_events",
            "claim_boundary",
        },
        "registry",
    )
    _require(
        registry["schema_version"] == REGISTRY_SCHEMA_VERSION,
        f"registry.schema_version must be {REGISTRY_SCHEMA_VERSION!r}",
    )
    _text(registry["registry_id"], "registry.registry_id")
    _require(
        registry["purpose"] == "seeded_synthetic_development_world_planning",
        "registry.purpose is unsupported",
    )

    carla = _object(registry["carla"], "registry.carla")
    _exact_keys(carla, {"version", "map", "requires_running_server"}, "registry.carla")
    _text(carla["version"], "registry.carla.version")
    _require(
        carla["map"] in SUPPORTED_CARLA_MAPS,
        f"registry.carla.map must be one of {list(SUPPORTED_CARLA_MAPS)}",
    )
    _require(
        carla["requires_running_server"] is False,
        "registry.carla.requires_running_server must be false",
    )

    focus = _object(registry["focus"], "registry.focus")
    _exact_keys(
        focus,
        {"scenario_class", "source_registry", "source_scene_id", "anchor", "class_parameters"},
        "registry.focus",
    )
    scenario_class = _text(focus["scenario_class"], "registry.focus.scenario_class")
    _require(
        scenario_class in SCENARIO_CLASS_PARAMETERS,
        f"unsupported focus scenario_class: {scenario_class}",
    )
    _text(focus["source_scene_id"], "registry.focus.source_scene_id")
    _text(focus["source_registry"], "registry.focus.source_registry")
    anchor = _object(focus["anchor"], "registry.focus.anchor")
    _exact_keys(anchor, {"center_xy_m", "forward_xy", "right_xy", "source"}, "registry.focus.anchor")
    _vector2(anchor["center_xy_m"], "registry.focus.anchor.center_xy_m")
    _vector2(anchor["forward_xy"], "registry.focus.anchor.forward_xy")
    _vector2(anchor["right_xy"], "registry.focus.anchor.right_xy")
    source = _object(anchor["source"], "registry.focus.anchor.source")
    _exact_keys(
        source,
        {"kind", "spawn_point_index", "runtime_validation_required"},
        "registry.focus.anchor.source",
    )
    _require(
        source["kind"] in {"c3_captured_anchor", "c4_multimap_captured_anchor"},
        "focus anchor source kind differs",
    )
    _integer(source["spawn_point_index"], "focus anchor spawn_point_index")
    _require(
        source["runtime_validation_required"] is False,
        "captured focus anchor must not claim runtime validation is required",
    )
    class_parameters = _object(focus["class_parameters"], "registry.focus.class_parameters")
    _exact_keys(
        class_parameters,
        SCENARIO_CLASS_PARAMETERS[scenario_class],
        "focus class_parameters",
    )
    for name, value in class_parameters.items():
        _number(value, f"focus class_parameters.{name}", minimum=0.1)

    duration_seconds = _number(registry["duration_seconds"], "duration_seconds", minimum=1.0)
    seed_contract = _object(registry["seed_contract"], "registry.seed_contract")
    _exact_keys(seed_contract, {"derivation", "subsystems"}, "registry.seed_contract")
    _require(seed_contract["derivation"] == "sha256_64bit", "unsupported seed derivation")
    _require(
        _string_array(seed_contract["subsystems"], "seed_contract.subsystems") == list(SUBSYSTEMS),
        f"seed_contract.subsystems must be {list(SUBSYSTEMS)}",
    )

    profiles = _object(registry["traffic_profiles"], "registry.traffic_profiles")
    _require(
        set(profiles) == set(REQUIRED_DRIVING_PROFILES),
        f"traffic_profiles must be exactly {list(REQUIRED_DRIVING_PROFILES)}",
    )
    profile_keys = {
        "actor_count",
        "blueprint_ids",
        "target_speed_mps",
        "following_distance_s",
        "max_acceleration_mps2",
        "comfort_deceleration_mps2",
        "lane_change_probability",
        "intent_modes",
        "destination_ids",
    }
    for profile_name in REQUIRED_DRIVING_PROFILES:
        profile = _object(profiles[profile_name], f"traffic_profiles.{profile_name}")
        _exact_keys(profile, profile_keys, f"traffic_profiles.{profile_name}")
        _integer(profile["actor_count"], f"{profile_name}.actor_count", minimum=1)
        _string_array(profile["blueprint_ids"], f"{profile_name}.blueprint_ids")
        for range_name in (
            "target_speed_mps",
            "following_distance_s",
            "max_acceleration_mps2",
            "comfort_deceleration_mps2",
        ):
            _numeric_range(profile[range_name], f"{profile_name}.{range_name}", minimum=0.0)
        _probability(profile["lane_change_probability"], f"{profile_name}.lane_change_probability")
        _string_array(profile["intent_modes"], f"{profile_name}.intent_modes")
        _string_array(profile["destination_ids"], f"{profile_name}.destination_ids")

    crowd = _object(registry["crowd"], "registry.crowd")
    crowd_keys = {
        "count_range",
        "blueprint_ids",
        "walking_speed_mps",
        "personal_space_m",
        "crossing_propensity",
        "distraction_probability",
        "origin_ids",
        "destination_ids",
    }
    _exact_keys(crowd, crowd_keys, "registry.crowd")
    count_range = _array(crowd["count_range"], "crowd.count_range")
    _require(len(count_range) == 2, "crowd.count_range must contain two integers")
    count_low = _integer(count_range[0], "crowd.count_range[0]", minimum=2)
    count_high = _integer(count_range[1], "crowd.count_range[1]", minimum=2)
    _require(count_low <= count_high, "crowd.count_range must be ordered")
    _string_array(crowd["blueprint_ids"], "crowd.blueprint_ids")
    _numeric_range(crowd["walking_speed_mps"], "crowd.walking_speed_mps", minimum=0.0)
    _numeric_range(crowd["personal_space_m"], "crowd.personal_space_m", minimum=0.0)
    _numeric_range(crowd["crossing_propensity"], "crowd.crossing_propensity", minimum=0.0)
    _probability(crowd["distraction_probability"], "crowd.distraction_probability")
    _string_array(crowd["origin_ids"], "crowd.origin_ids")
    _string_array(crowd["destination_ids"], "crowd.destination_ids")

    grouping = _object(registry["grouping"], "registry.grouping")
    grouping_keys = {
        "membership_probability",
        "group_size_range",
        "cohesion_range",
        "formations",
        "shared_destination_ids",
    }
    _exact_keys(grouping, grouping_keys, "registry.grouping")
    _probability(grouping["membership_probability"], "grouping.membership_probability")
    group_size = _array(grouping["group_size_range"], "grouping.group_size_range")
    _require(len(group_size) == 2, "grouping.group_size_range must contain two integers")
    group_low = _integer(group_size[0], "grouping.group_size_range[0]", minimum=2)
    group_high = _integer(group_size[1], "grouping.group_size_range[1]", minimum=2)
    _require(group_low <= group_high <= count_high, "invalid grouping.group_size_range")
    _numeric_range(grouping["cohesion_range"], "grouping.cohesion_range", minimum=0.0)
    _string_array(grouping["formations"], "grouping.formations")
    _string_array(grouping["shared_destination_ids"], "grouping.shared_destination_ids")

    rerouting = _object(registry["rerouting"], "registry.rerouting")
    rerouting_keys = {
        "group_reroute_probability",
        "trigger_time_range_s",
        "decision_duration_range_s",
        "blocked_destination_id",
        "alternative_destination_ids",
        "selection_rule",
    }
    _exact_keys(rerouting, rerouting_keys, "registry.rerouting")
    _probability(rerouting["group_reroute_probability"], "rerouting.group_reroute_probability")
    reroute_trigger = _numeric_range(
        rerouting["trigger_time_range_s"], "rerouting.trigger_time_range_s", minimum=0.0
    )
    reroute_duration = _numeric_range(
        rerouting["decision_duration_range_s"],
        "rerouting.decision_duration_range_s",
        minimum=0.01,
    )
    _require(reroute_trigger[1] + reroute_duration[1] <= duration_seconds, "rerouting exceeds duration")
    _text(rerouting["blocked_destination_id"], "rerouting.blocked_destination_id")
    _string_array(rerouting["alternative_destination_ids"], "rerouting.alternative_destination_ids")
    _require(
        rerouting["selection_rule"] == "seeded_bernoulli_with_first_group_guarantee",
        "unsupported rerouting.selection_rule",
    )

    events = _array(registry["long_tail_events"], "registry.long_tail_events")
    _require(len(events) >= len(REQUIRED_EVENT_TYPES), "registry needs all four long-tail event types")
    event_types: list[str] = []
    event_keys = {"type", "trigger_time_range_s", "duration_range_s", "selection_rule"}
    selection_keys = {
        "selector",
        "eligible_actor_kind",
        "eligible_profiles",
        "distinct_primary_actor_when_possible",
    }
    for index, event_value in enumerate(events):
        event = _object(event_value, f"long_tail_events[{index}]")
        _exact_keys(event, event_keys, f"long_tail_events[{index}]")
        event_type = _text(event["type"], f"long_tail_events[{index}].type")
        _require(event_type in REQUIRED_EVENT_TYPES, f"unsupported long-tail event type: {event_type}")
        event_types.append(event_type)
        trigger_range = _numeric_range(
            event["trigger_time_range_s"],
            f"long_tail_events[{index}].trigger_time_range_s",
            minimum=0.0,
        )
        event_duration = _numeric_range(
            event["duration_range_s"],
            f"long_tail_events[{index}].duration_range_s",
            minimum=0.01,
        )
        _require(trigger_range[1] + event_duration[1] <= duration_seconds, f"{event_type} exceeds duration")
        selection = _object(event["selection_rule"], f"{event_type}.selection_rule")
        _exact_keys(selection, selection_keys, f"{event_type}.selection_rule")
        _require(
            selection["selector"] == "seeded_choice_from_sorted_eligible_actor_ids",
            f"unsupported selector for {event_type}",
        )
        expected_kind = "pedestrian" if event_type == "occluded_jaywalk" else "vehicle"
        _require(selection["eligible_actor_kind"] == expected_kind, f"{event_type} actor kind differs")
        eligible_profiles = _array(selection["eligible_profiles"], f"{event_type}.eligible_profiles")
        _require(all(item in REQUIRED_DRIVING_PROFILES for item in eligible_profiles), f"{event_type} profile differs")
        if expected_kind == "vehicle":
            _require(bool(eligible_profiles), f"{event_type} needs eligible vehicle profiles")
        else:
            _require(not eligible_profiles, f"{event_type} pedestrian rule cannot name vehicle profiles")
        _require(
            isinstance(selection["distinct_primary_actor_when_possible"], bool),
            f"{event_type}.distinct_primary_actor_when_possible must be boolean",
        )
    _require(len(event_types) == len(set(event_types)), "long-tail event types contain duplicates")
    _require(set(REQUIRED_EVENT_TYPES).issubset(event_types), "long-tail event coverage is incomplete")

    claim_boundary = _string_array(registry["claim_boundary"], "registry.claim_boundary")
    _require(len(claim_boundary) >= 2, "claim_boundary must state at least two limits")


def _derive_seed(master_seed: int, subsystem: str) -> int:
    material = f"{PLAN_SCHEMA_VERSION}|{master_seed}|{subsystem}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _uniform(rng: random.Random, value: Sequence[Any], digits: int = 3) -> float:
    return round(rng.uniform(float(value[0]), float(value[1])), digits)


def _compile_traffic(
    registry: Mapping[str, Any], rng: random.Random
) -> list[dict[str, Any]]:
    actors: list[dict[str, Any]] = []
    scenario_class = str(registry["focus"]["scenario_class"])
    lane_ids = (f"{scenario_class}_approach_left", f"{scenario_class}_approach_right")
    for profile_name in REQUIRED_DRIVING_PROFILES:
        profile = registry["traffic_profiles"][profile_name]
        for profile_index in range(1, int(profile["actor_count"]) + 1):
            actor_id = f"n1_vehicle_{profile_name}_{profile_index:02d}"
            actors.append(
                {
                    "actor_id": actor_id,
                    "actor_kind": "vehicle",
                    "blueprint_id": rng.choice(profile["blueprint_ids"]),
                    "behavior_profile": profile_name,
                    "spawn_intent": {
                        "lane_id": rng.choice(lane_ids),
                        "longitudinal_offset_m": round(rng.uniform(-18.0, 18.0), 3),
                        "lateral_offset_m": round(rng.uniform(-0.25, 0.25), 3),
                    },
                    "intent": {
                        "maneuver": rng.choice(profile["intent_modes"]),
                        "destination_id": rng.choice(profile["destination_ids"]),
                        "target_speed_mps": _uniform(rng, profile["target_speed_mps"]),
                        "following_distance_s": _uniform(rng, profile["following_distance_s"]),
                        "max_acceleration_mps2": _uniform(rng, profile["max_acceleration_mps2"]),
                        "comfort_deceleration_mps2": _uniform(
                            rng, profile["comfort_deceleration_mps2"]
                        ),
                        "lane_change_probability": float(profile["lane_change_probability"]),
                        "scheduled_event_ids": [],
                    },
                }
            )
    return actors


def _compile_crowd(
    registry: Mapping[str, Any], rng: random.Random
) -> list[dict[str, Any]]:
    crowd = registry["crowd"]
    count = rng.randint(int(crowd["count_range"][0]), int(crowd["count_range"][1]))
    actors: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        actor_id = f"n1_pedestrian_{index:02d}"
        crossing_propensity = _uniform(rng, crowd["crossing_propensity"])
        actors.append(
            {
                "actor_id": actor_id,
                "actor_kind": "pedestrian",
                "blueprint_id": rng.choice(crowd["blueprint_ids"]),
                "behavior_profile": "crowd_member",
                "spawn_intent": {
                    "origin_id": rng.choice(crowd["origin_ids"]),
                    "forward_offset_m": round(rng.uniform(-8.0, 12.0), 3),
                    "right_offset_m": round(rng.uniform(-7.0, 7.0), 3),
                },
                "intent": {
                    "initial_destination_id": rng.choice(crowd["destination_ids"]),
                    "effective_destination_id": None,
                    "walking_speed_mps": _uniform(rng, crowd["walking_speed_mps"]),
                    "personal_space_m": _uniform(rng, crowd["personal_space_m"]),
                    "crossing_propensity": crossing_propensity,
                    "distracted": rng.random() < float(crowd["distraction_probability"]),
                    "crossing_mode": (
                        "opportunistic" if rng.random() < crossing_propensity else "marked_crossing"
                    ),
                    "group_id": None,
                    "reroute_interaction_id": None,
                    "scheduled_event_ids": [],
                },
            }
        )
    return actors


def _compile_groups(
    registry: Mapping[str, Any],
    pedestrians: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    config = registry["grouping"]
    shuffled_ids = [actor["actor_id"] for actor in pedestrians]
    rng.shuffle(shuffled_ids)
    low, high = (int(value) for value in config["group_size_range"])
    groups: list[dict[str, Any]] = []
    cursor = 0
    while len(shuffled_ids) - cursor >= low:
        must_create = not groups
        if not must_create and rng.random() >= float(config["membership_probability"]):
            cursor += 1
            continue
        remaining = len(shuffled_ids) - cursor
        size = min(rng.randint(low, high), remaining)
        members = sorted(shuffled_ids[cursor : cursor + size])
        cursor += size
        group_id = f"n1_crowd_group_{len(groups) + 1:02d}"
        groups.append(
            {
                "group_id": group_id,
                "leader_actor_id": members[0],
                "member_actor_ids": members,
                "formation": rng.choice(config["formations"]),
                "cohesion": _uniform(rng, config["cohesion_range"]),
                "shared_destination_id": rng.choice(config["shared_destination_ids"]),
                "reroute_interaction_id": None,
            }
        )
    actor_by_id = {actor["actor_id"]: actor for actor in pedestrians}
    for group in groups:
        for actor_id in group["member_actor_ids"]:
            intent = actor_by_id[actor_id]["intent"]
            intent["group_id"] = group["group_id"]
            intent["initial_destination_id"] = group["shared_destination_id"]
    return groups


def _compile_reroutes(
    registry: Mapping[str, Any],
    pedestrians: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    config = registry["rerouting"]
    actor_by_id = {actor["actor_id"]: actor for actor in pedestrians}
    interactions: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        selected = group_index == 0 or rng.random() < float(config["group_reroute_probability"])
        if not selected:
            continue
        interaction_id = f"n1_group_reroute_{len(interactions) + 1:02d}"
        target = rng.choice(config["alternative_destination_ids"])
        interaction = {
            "interaction_id": interaction_id,
            "type": "group_reroute",
            "trigger_time_s": _uniform(rng, config["trigger_time_range_s"]),
            "duration_s": _uniform(rng, config["decision_duration_range_s"]),
            "participant_actor_ids": list(group["member_actor_ids"]),
            "leader_actor_id": group["leader_actor_id"],
            "from_destination_id": group["shared_destination_id"],
            "to_destination_id": target,
            "selection_rule": {
                "rule": config["selection_rule"],
                "configured_probability": float(config["group_reroute_probability"]),
                "first_group_guarantee_applied": group_index == 0,
                "obstruction": (
                    f"{registry['focus']['source_scene_id']}_"
                    f"{registry['focus']['scenario_class']}_interaction_zone"
                ),
                "blocked_waypoint_id": config["blocked_destination_id"],
            },
        }
        interactions.append(interaction)
        group["reroute_interaction_id"] = interaction_id
        for actor_id in group["member_actor_ids"]:
            intent = actor_by_id[actor_id]["intent"]
            intent["effective_destination_id"] = target
            intent["reroute_interaction_id"] = interaction_id
    for actor in pedestrians:
        intent = actor["intent"]
        if intent["effective_destination_id"] is None:
            intent["effective_destination_id"] = intent["initial_destination_id"]
    return interactions


def _event_effect(event_type: str, rng: random.Random) -> dict[str, Any]:
    if event_type == "occluded_jaywalk":
        return {
            "intent_override": "cross_from_behind_construction_occluder",
            "crossing_speed_mps": round(rng.uniform(1.2, 2.1), 3),
            "crossing_direction": rng.choice(("left_to_right", "right_to_left")),
        }
    if event_type == "sudden_brake":
        return {
            "intent_override": "hard_deceleration_then_hold",
            "deceleration_mps2": round(rng.uniform(4.0, 7.0), 3),
            "hold_speed_mps": 0.0,
        }
    if event_type == "reverse_pullout":
        return {
            "intent_override": "reverse_from_curb_into_approach_lane",
            "reverse_speed_mps": round(rng.uniform(1.0, 2.2), 3),
            "steering_degrees": round(rng.uniform(12.0, 28.0), 3),
        }
    if event_type == "door_open":
        return {
            "intent_override": "open_roadside_door_into_corridor",
            "door_side": rng.choice(("left", "right")),
            "opening_angle_degrees": round(rng.uniform(45.0, 75.0), 3),
        }
    raise AssertionError(f"validated event type was lost: {event_type}")


def _compile_events(
    registry: Mapping[str, Any],
    actors: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    actor_by_id = {actor["actor_id"]: actor for actor in actors}
    used_actor_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    for index, event_config in enumerate(registry["long_tail_events"], start=1):
        event_type = event_config["type"]
        configured_rule = event_config["selection_rule"]
        candidates = [
            actor["actor_id"]
            for actor in actors
            if actor["actor_kind"] == configured_rule["eligible_actor_kind"]
            and (
                not configured_rule["eligible_profiles"]
                or actor["behavior_profile"] in configured_rule["eligible_profiles"]
            )
        ]
        candidates.sort()
        _require(bool(candidates), f"{event_type} has no eligible actors")
        available = [actor_id for actor_id in candidates if actor_id not in used_actor_ids]
        distinct_applied = bool(
            configured_rule["distinct_primary_actor_when_possible"] and available
        )
        chosen_pool = available if distinct_applied else candidates
        chosen_actor_id = rng.choice(chosen_pool)
        used_actor_ids.add(chosen_actor_id)
        event_id = f"n1_event_{index:02d}_{event_type}"
        event = {
            "event_id": event_id,
            "type": event_type,
            "trigger_time_s": _uniform(rng, event_config["trigger_time_range_s"]),
            "duration_s": _uniform(rng, event_config["duration_range_s"]),
            "primary_actor_id": chosen_actor_id,
            "selection_rule": {
                "selector": configured_rule["selector"],
                "eligible_actor_kind": configured_rule["eligible_actor_kind"],
                "eligible_profiles": list(configured_rule["eligible_profiles"]),
                "distinct_primary_actor_when_possible": bool(
                    configured_rule["distinct_primary_actor_when_possible"]
                ),
                "candidate_actor_ids": candidates,
                "unused_candidate_rule_applied": distinct_applied,
                "chosen_actor_id": chosen_actor_id,
            },
            "intent_effect": _event_effect(event_type, rng),
        }
        events.append(event)
        actor_by_id[chosen_actor_id]["intent"]["scheduled_event_ids"].append(event_id)
    return events


def compile_plan(registry: Mapping[str, Any], master_seed: int) -> dict[str, Any]:
    """Compile one deterministic world plan from a validated registry and seed."""

    validate_registry(registry)
    master_seed = _integer(master_seed, "master_seed")
    _require(master_seed <= (2**63 - 1), "master_seed must be <= 2**63 - 1")
    source = copy.deepcopy(dict(registry))
    subsystem_seeds = {
        subsystem: _derive_seed(master_seed, subsystem) for subsystem in SUBSYSTEMS
    }
    traffic = _compile_traffic(source, random.Random(subsystem_seeds["traffic"]))
    pedestrians = _compile_crowd(source, random.Random(subsystem_seeds["crowd"]))
    groups = _compile_groups(
        source, pedestrians, random.Random(subsystem_seeds["grouping"])
    )
    interactions = _compile_reroutes(
        source,
        pedestrians,
        groups,
        random.Random(subsystem_seeds["rerouting"]),
    )
    traffic.sort(key=lambda actor: actor["actor_id"])
    pedestrians.sort(key=lambda actor: actor["actor_id"])
    actors = traffic + pedestrians
    events = _compile_events(source, actors, random.Random(subsystem_seeds["events"]))
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": f"dtr-carla-n1-seed-{master_seed}",
        "registry_id": source["registry_id"],
        "registry_schema_version": source["schema_version"],
        "evidence_role": "synthetic_development_input_plan",
        "master_seed": master_seed,
        "subsystem_seeds": subsystem_seeds,
        "environment": {
            "carla_version": source["carla"]["version"],
            "map": source["carla"]["map"],
            "requires_running_server_to_compile": False,
        },
        "focus": copy.deepcopy(source["focus"]),
        "duration_seconds": float(source["duration_seconds"]),
        "vehicle_intents": traffic,
        "walker_intents": pedestrians,
        "crowd_groups": groups,
        "group_reroute_interactions": interactions,
        "tail_events": events,
        "coverage": {
            "driving_profiles": list(REQUIRED_DRIVING_PROFILES),
            "event_types": [event["type"] for event in events],
            "actor_count": len(actors),
            "crowd_group_count": len(groups),
            "reroute_interaction_count": len(interactions),
        },
        "claim_boundary": copy.deepcopy(source["claim_boundary"]),
    }
    canonical = json.dumps(
        plan, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    plan["plan_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest().upper()
    return plan


def write_json_atomic(path: Path, value: Mapping[str, Any], *, indent: int = 2) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).with_name("dtr_carla_n1_natural_dynamics_registry.json"),
        help="N1 registry JSON (defaults to the registry beside this script)",
    )
    parser.add_argument("--seed", type=int, required=True, help="non-negative master seed")
    parser.add_argument("--output", type=Path, required=True, help="compiled plan JSON path")
    parser.add_argument("--indent", type=int, default=2, choices=(2, 4))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = load_json(args.registry.resolve(strict=True))
    plan = compile_plan(registry, args.seed)
    write_json_atomic(args.output, plan, indent=args.indent)
    summary = {
        "actor_count": plan["coverage"]["actor_count"],
        "event_types": plan["coverage"]["event_types"],
        "output": os.fspath(args.output.resolve()),
        "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
        "plan_id": plan["plan_id"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
