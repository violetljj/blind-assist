#!/usr/bin/env python3
"""Fail-closed validator for externally supplied route-intent episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_explicit_route_intent_episode_v1"
RUNTIME_PROVIDERS = {"navigation", "explicit_user_choice", "navigation_or_explicit_user_choice"}
REQUIRED_HORIZONS = [1000, 2000, 3000]


def require_identifier(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("REQUIRED_"):
        raise ValueError(f"missing concrete {field}")
    return text


def validate_episode(value: dict[str, Any], *, runtime: bool) -> dict[str, Any]:
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported route-intent schema")
    episode_id = require_identifier(value.get("episode_id"), "episode_id")
    source_id = require_identifier(value.get("parent_source_id"), "parent_source_id")
    provider = value.get("provider", {})
    provider_type = str(provider.get("type", ""))
    require_identifier(provider.get("provider_id"), "provider_id")
    if provider.get("inferred_by_risk_model") is not False:
        raise ValueError("route intent must not be inferred by the risk model")
    input_space = provider.get("input_space")
    coordinate = value.get("coordinate_contract", {})
    if coordinate.get("space") != "normalized_current_camera_frame_xy":
        raise ValueError("route intent must be projected into the current camera frame")
    require_identifier(coordinate.get("projection_receipt_id"), "projection_receipt_id")
    if input_space == "world_waypoints":
        require_identifier(coordinate.get("device_to_world_alignment_receipt_id"),
                           "device_to_world_alignment_receipt_id")
    if runtime and provider_type not in RUNTIME_PROVIDERS:
        raise ValueError("provider is not allowed at runtime")
    isolation = value.get("training_isolation", {})
    if runtime and isolation.get("future_video_teacher_allowed_in_eval_or_runtime") is not False:
        raise ValueError("future-video teacher must be disabled in runtime/eval")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("route-intent samples must be non-empty")
    previous_timestamp: int | None = None
    valid_count = 0
    for index, sample in enumerate(samples):
        timestamp = int(sample["timestamp_ms"])
        valid_until = int(sample["valid_until_timestamp_ms"])
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("sample timestamps must be strictly increasing")
        previous_timestamp = timestamp
        if valid_until < timestamp or valid_until - timestamp > 1000:
            raise ValueError("route intent validity must be within zero to one second")
        confidence = float(sample["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        route_valid = sample.get("route_valid") is True
        waypoints = sample.get("horizon_waypoints", [])
        if not route_valid:
            if waypoints:
                raise ValueError("invalid route sample must not contain waypoints")
            continue
        valid_count += 1
        horizons = [int(row["horizon_ms"]) for row in waypoints]
        if horizons != REQUIRED_HORIZONS:
            raise ValueError(f"sample {index} must contain exact 1/2/3 second horizons")
        for waypoint in waypoints:
            xy = waypoint.get("xy_norm")
            if not isinstance(xy, list) or len(xy) != 2 or not all(0.0 <= float(axis) <= 1.0 for axis in xy):
                raise ValueError("normalized waypoint must lie inside [0,1]^2")
    fallback = value.get("fallback", {})
    if fallback.get("missing_stale_or_low_confidence_route") != "context_attention_only":
        raise ValueError("unknown-route fallback must be context_attention_only")
    if fallback.get("directional_instruction_allowed") is not False:
        raise ValueError("unknown-route fallback must forbid directional instructions")
    if fallback.get("intervention_upgrade_allowed") is not False:
        raise ValueError("unknown-route fallback must forbid intervention upgrades")
    return {
        "episode_id": episode_id, "parent_source_id": source_id,
        "sample_count": len(samples), "valid_route_sample_count": valid_count,
        "runtime_contract_valid": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    value = json.loads(args.episode.read_text(encoding="utf-8"))
    print(json.dumps(validate_episode(value, runtime=args.runtime), ensure_ascii=False))


if __name__ == "__main__":
    main()
