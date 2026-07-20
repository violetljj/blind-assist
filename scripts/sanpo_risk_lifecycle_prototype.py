#!/usr/bin/env python3
"""Temporal prototype for attested human and provisional model SANPO supervision.

This module intentionally contains no trainer, image loader, pseudo-labeler, or
deployment path.  It turns only hash-attested, fully validated human episode
targets into deterministic temporal supervision and can build a small Keras
head over *externally supplied* per-frame features.  Public RGB/masks and
GPT/VLM labels can be admitted only through an explicitly provisional,
hash-attested report; they never become human truth or production evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


TARGET_FORMAT = "blindassist_risk_lifecycle_target_v1"
REPORT_FORMAT = "blindassist_risk_lifecycle_target_report_v1"
CORRIDOR_RELATIONS = ("outside_or_nonblocking", "enters_or_blocks")
LIFECYCLE_STATES = ("non_alert", "approach", "alertable", "post_event")


class TargetContractError(ValueError):
    """The supplied targets cannot safely supervise this prototype."""


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TargetContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TargetContractError(f"JSON root must be an object: {path}")
    return payload


def _read_targets(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TargetContractError(f"cannot read target file {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise TargetContractError(f"target file must not contain blank rows: line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise TargetContractError(f"invalid target JSON at line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise TargetContractError(f"target line {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise TargetContractError("target file must contain at least one reviewed episode")
    return rows


def _require_string(row: dict[str, Any], name: str, *, where: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise TargetContractError(f"{where}.{name} must be a non-empty string")
    return value


def validate_target(target: dict[str, Any], *, allowed_hazard_types: Iterable[str]) -> None:
    """Verify a builder output before it can be transformed into time labels."""
    where = f"target[{target.get('episode_id', '?')}]"
    if target.get("format") != TARGET_FORMAT:
        raise TargetContractError(f"{where}.format is not {TARGET_FORMAT}")
    if target.get("pixel_supervision_role") != "auxiliary_only":
        raise TargetContractError(f"{where}.pixel_supervision_role must be auxiliary_only")
    duration_ms = target.get("duration_ms")
    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms <= 0:
        raise TargetContractError(f"{where}.duration_ms must be a positive integer")
    if not isinstance(target.get("expected_should_alert"), bool):
        raise TargetContractError(f"{where}.expected_should_alert must be boolean")
    profile = target.get("risk_profile")
    intervals = target.get("lifecycle_intervals_ms")
    if not isinstance(profile, dict) or not isinstance(intervals, dict):
        raise TargetContractError(f"{where} must include risk_profile and lifecycle_intervals_ms objects")
    hazard = _require_string(profile, "primary_hazard_type", where=f"{where}.risk_profile")
    if hazard not in set(allowed_hazard_types):
        raise TargetContractError(f"{where}.risk_profile.primary_hazard_type is outside the frozen scene vocabulary")
    relation = profile.get("corridor_relation")
    if relation not in CORRIDOR_RELATIONS:
        raise TargetContractError(f"{where}.risk_profile.corridor_relation is invalid")
    lifecycle = profile.get("lifecycle")
    positive = target["expected_should_alert"]
    if positive:
        if relation != "enters_or_blocks" or lifecycle != "approach_alertable_clear":
            raise TargetContractError(f"{where} positive risk profile is inconsistent")
        expected_keys = {"approach", "alertable", "post_event"}
    else:
        if relation != "outside_or_nonblocking" or lifecycle != "no_alert":
            raise TargetContractError(f"{where} matched-negative risk profile is inconsistent")
        expected_keys = {"non_alert"}
    if set(intervals) != expected_keys:
        raise TargetContractError(f"{where}.lifecycle_intervals_ms keys are inconsistent with risk profile")
    expected_cursor = 0
    ordered_states = ("approach", "alertable", "post_event") if positive else ("non_alert",)
    for state in ordered_states:
        interval = intervals[state]
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in interval)
        ):
            raise TargetContractError(f"{where}.lifecycle_intervals_ms.{state} must be a two-integer interval")
        start_ms, end_ms = interval
        if start_ms != expected_cursor or not start_ms <= end_ms <= duration_ms:
            raise TargetContractError(f"{where}.lifecycle_intervals_ms must be contiguous half-open intervals")
        expected_cursor = end_ms
    if expected_cursor != duration_ms:
        raise TargetContractError(f"{where}.lifecycle_intervals_ms must cover the full episode duration")


def load_attested_targets(
    *, targets_path: Path, report_path: Path, allowed_hazard_types: Iterable[str],
) -> list[dict[str, Any]]:
    """Load human or explicitly-provisional targets with matching attestation."""
    report = _read_json(report_path)
    if report.get("format") != REPORT_FORMAT:
        raise TargetContractError("target report format is invalid")
    supervision_tier = report.get("supervision_tier", "attested_human_reviewed")
    if supervision_tier == "attested_human_reviewed":
        validation = report.get("validated_collection")
        if not isinstance(validation, dict) or validation.get("training_eligible") is not True:
            raise TargetContractError("target report is not attested as a complete human-reviewed collection")
        if report.get("training_execution_authorized") is not False:
            raise TargetContractError("human target report must not grant training execution authorization")
    elif supervision_tier == "hash_bound_model_silver_provisional":
        if report.get("training_execution_authorized") is not True:
            raise TargetContractError("provisional model target report must explicitly authorize training")
        if report.get("provisional_training_only") is not True:
            raise TargetContractError("provisional model target report must remain provisional_training_only")
        _require_string(report, "source_manifest_sha256", where="target report")
        labeler = report.get("labeler")
        if not isinstance(labeler, dict):
            raise TargetContractError("provisional model target report requires labeler attestation")
        for field in ("provider", "model", "prompt_sha256"):
            _require_string(labeler, field, where="target report.labeler")
    else:
        raise TargetContractError("target report supervision_tier is invalid")
    if report.get("production_model_replacement_authorized") is not False:
        raise TargetContractError("target report must not authorize production-model replacement")
    if report.get("pixel_supervision_role") != "auxiliary_only":
        raise TargetContractError("target report must preserve auxiliary-only pixel supervision")
    targets = _read_targets(targets_path)
    if report.get("target_sha256") != canonical_sha256(targets):
        raise TargetContractError("target SHA256 does not match the attested report")
    for target in targets:
        validate_target(target, allowed_hazard_types=allowed_hazard_types)
    return targets


def lifecycle_labels_for_timestamps(target: dict[str, Any], timestamps_ms: Iterable[int]) -> list[int]:
    """Map frame timestamps to lifecycle IDs using the reviewed half-open intervals."""
    profile = target.get("risk_profile")
    if not isinstance(profile, dict):
        raise TargetContractError("target risk_profile must be an object")
    duration_ms = target.get("duration_ms")
    intervals = target.get("lifecycle_intervals_ms")
    if not isinstance(duration_ms, int) or not isinstance(intervals, dict):
        raise TargetContractError("target has no validated duration or lifecycle intervals")
    state_intervals = [(state, bounds) for state, bounds in intervals.items()]
    labels: list[int] = []
    for timestamp in timestamps_ms:
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or not 0 <= timestamp < duration_ms:
            raise TargetContractError("timestamps must be integer milliseconds within the half-open episode duration")
        matches = [state for state, bounds in state_intervals if bounds[0] <= timestamp < bounds[1]]
        if len(matches) != 1 or matches[0] not in LIFECYCLE_STATES:
            raise TargetContractError("timestamp does not map to exactly one validated lifecycle state")
        labels.append(LIFECYCLE_STATES.index(matches[0]))
    return labels


def risk_profile_labels(target: dict[str, Any], *, hazard_types: tuple[str, ...]) -> dict[str, int]:
    """Return small integer targets directly from reviewed episode truth."""
    profile = target["risk_profile"]
    hazard = profile["primary_hazard_type"]
    if hazard not in hazard_types:
        raise TargetContractError("hazard type is outside the prototype vocabulary")
    return {
        "hazard": hazard_types.index(hazard),
        "corridor_relation": CORRIDOR_RELATIONS.index(profile["corridor_relation"]),
        "episode_should_alert": int(target["expected_should_alert"]),
    }


def build_temporal_risk_lifecycle_head(
    keras: Any, *, feature_dim: int, hazard_types: tuple[str, ...], channels: int = 96,
) -> Any:
    """Build a prototype head over precomputed frame features, without a trainer.

    The caller owns feature extraction.  Pixel segmentation is deliberately not
    an input to any primary event target.  The head emits logits so any future
    trainer must explicitly choose losses and preserve LOSO isolation.
    """
    if feature_dim <= 0 or channels <= 0:
        raise ValueError("feature_dim and channels must be positive")
    if not hazard_types or len(set(hazard_types)) != len(hazard_types):
        raise ValueError("hazard_types must be a non-empty tuple of unique values")
    features = keras.Input(shape=(None, feature_dim), dtype="float32", name="frame_features")
    temporal = keras.layers.LayerNormalization(name="risk_lifecycle_feature_norm")(features)
    temporal = keras.layers.Conv1D(channels, 3, padding="causal", activation="relu", name="risk_lifecycle_causal_conv1")(temporal)
    temporal = keras.layers.Conv1D(channels, 3, padding="causal", activation="relu", name="risk_lifecycle_causal_conv2")(temporal)
    lifecycle_logits = keras.layers.Dense(len(LIFECYCLE_STATES), name="lifecycle_logits")(temporal)
    episode = keras.layers.GlobalMaxPooling1D(name="risk_lifecycle_episode_pool")(temporal)
    hazard_logits = keras.layers.Dense(len(hazard_types), name="hazard_logits")(episode)
    corridor_logits = keras.layers.Dense(len(CORRIDOR_RELATIONS), name="corridor_relation_logits")(episode)
    should_alert_logit = keras.layers.Dense(1, name="episode_should_alert_logit")(episode)
    model = keras.Model(
        inputs=features,
        outputs={
            "hazard_logits": hazard_logits,
            "corridor_relation_logits": corridor_logits,
            "lifecycle_logits": lifecycle_logits,
            "episode_should_alert_logit": should_alert_logit,
        },
        name="sanpo_risk_lifecycle_prototype",
    )
    model.risk_lifecycle_contract = {
        "format": "blindassist_sanpo_risk_lifecycle_prototype_v1",
        "feature_input": "externally_supplied_frame_features_only",
        "primary_supervision": "attested_human_reviewed_or_hash_bound_model_silver_provisional",
        "pixel_supervision_role": "auxiliary_only",
        "training_execution_authorized": "depends_on_target_report_supervision_tier",
        "production_model_replacement_authorized": False,
        "required_split": "leave_one_session_out",
        "lifecycle_states": list(LIFECYCLE_STATES),
        "hazard_types": list(hazard_types),
    }
    return model
