"""Validation helpers for the frozen r7.23 multi-expert risk-profile contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_public_video_multiexpert_risk_profile_contract_v1"
EXPECTED_STATES = ["clear", "present", "uncertain"]
REQUIRED_FALSE_AUTHORIZATIONS = {
    "training",
    "calibration",
    "blind",
    "android_runtime_change",
    "production_model_replacement",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        raise ValueError(f"{label} SHA256 is invalid")
    return value.lower()


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unexpected multi-expert risk-profile contract schema")
    if contract.get("contract_id") != "public-video-multiexpert-risk-profile-r723":
        raise ValueError("unexpected contract_id")

    bound = contract.get("bound_inputs")
    channels = contract.get("risk_profile_channels")
    fusion = contract.get("fusion")
    roles = contract.get("supervision_roles")
    prospective = contract.get("prospective_requirements")
    authorization = contract.get("authorizations")
    if not all(isinstance(value, dict) for value in (
        bound, channels, fusion, roles, prospective, authorization
    )):
        raise ValueError("multi-expert contract sections are incomplete")

    for key in (
        "dinov2_pair_contract_sha256",
        "chromatic_marker_contract_sha256",
        "multiexpert_prototype_sha256",
        "japan_video_sha256",
    ):
        _sha256(bound.get(key), key)

    if set(channels) != {"general_static_dinov2", "chromatic_construction_marker"}:
        raise ValueError("risk-profile channel set drifted")
    if any(channel.get("absence_is_clear") is not False for channel in channels.values()):
        raise ValueError("channel absence must never be interpreted as clear")

    expected_fusion = {
        "open_rule": "OR over independent positive evidence channels",
        "close_rule": "every channel that opened the event must independently confirm close",
        "conflict_rule": "keep present or uncertain; never clear on conflict",
        "states": EXPECTED_STATES,
        "event_level_reminder_once": True,
        "frame_level_recall_is_diagnostic_only": True,
    }
    if any(fusion.get(key) != expected for key, expected in expected_fusion.items()):
        raise ValueError("multi-expert fusion semantics drifted")

    if roles != {
        "risk_profile_and_lifecycle": "primary",
        "pixel_segmentation": "auxiliary_only",
        "distance_field": "auxiliary_only",
    }:
        raise ValueError("primary or auxiliary supervision roles drifted")

    required_true = {
        "source_video_sha256_not_in_derivation",
        "item_level_reuse_license_required",
        "continuous_original_order_required",
        "all_channel_features_frozen_before_visual_review",
        "hard_cut_or_montage_forbidden",
        "no_post_review_threshold_window_or_channel_changes",
        "event_open_and_close_must_match_original_order_review",
    }
    if any(prospective.get(key) is not True for key in required_true):
        raise ValueError("prospective chronology or source requirements were weakened")
    if authorization.get("prospective_evaluation") is not True:
        raise ValueError("prospective evaluation must remain authorized")
    if any(authorization.get(key) is not False for key in REQUIRED_FALSE_AUTHORIZATIONS):
        raise ValueError("contract contains an unauthorized promotion flag")
    return contract


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve()
    sidecar = Path(str(resolved) + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"contract or sidecar is missing: {resolved}")
    actual = sha256_file(resolved)
    if sidecar.read_text(encoding="ascii").strip().lower() != actual:
        raise ValueError(f"contract sidecar mismatch: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    return validate_contract(value), {"path": str(resolved), "sha256": actual}


def verify_bound_inputs(
    contract: dict[str, Any],
    *,
    dinov2_contract: Path,
    chromatic_contract: Path,
    prototype_report: Path,
) -> None:
    bound = contract["bound_inputs"]
    checks = (
        (dinov2_contract, "dinov2_pair_contract_sha256"),
        (chromatic_contract, "chromatic_marker_contract_sha256"),
        (prototype_report, "multiexpert_prototype_sha256"),
    )
    for path, key in checks:
        if sha256_file(path.resolve()) != bound[key]:
            raise ValueError(f"bound input hash mismatch: {key}")
