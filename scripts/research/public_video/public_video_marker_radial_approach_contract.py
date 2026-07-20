"""Validation helpers for the frozen r7.25 radial marker-approach contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_public_video_marker_radial_approach_contract_v1"
FALSE_AUTHORIZATIONS = {"training", "calibration", "blind", "android_runtime_change", "production_model_replacement"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError(f"invalid {label} SHA256")
    return value.lower()


def validate_contract(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SCHEMA or value.get("contract_id") != "public-video-marker-radial-approach-r725":
        raise ValueError("unexpected radial-approach contract identity")
    bound = value.get("bound_inputs", {})
    for key in ("chromatic_marker_contract_sha256", "radial_approach_probe_sha256", "r724_negative_result_sha256", "japan_video_sha256", "matoaka_video_sha256"):
        _sha(bound.get(key), key)
    feature = value.get("feature_contract", {})
    if feature.get("accepted_detection_per_timestamp") != "accepted detection with maximum bottom_y_norm" or feature.get("track_identity_claimed") is not False:
        raise ValueError("detection selection or tracking claim drifted")
    if feature.get("dinov2_direct_event_open_allowed") is not False or feature.get("dinov2_role") != "support_only":
        raise ValueError("DINO direct-open prohibition drifted")
    gate = value.get("radial_approach_gate", {})
    expected_gate = {
        "minimum_accepted_samples": 5,
        "endpoint_samples": 3,
        "early_and_late_statistic": "median",
        "minimum_bottom_y_progress": 0.05,
        "bottom_y_progress_must_exceed_absolute_horizontal_shift": True,
        "positive_log_area_growth_required": True,
        "all_conditions_required": True,
    }
    if gate != expected_gate:
        raise ValueError("radial approach gate drifted")
    lifecycle = value.get("lifecycle", {})
    if lifecycle.get("states") != ["clear", "present", "uncertain"] or lifecycle.get("absence_alone_can_open") is not False or lifecycle.get("conflict_can_clear") is not False:
        raise ValueError("lifecycle fail-closed semantics drifted")
    prospective = value.get("prospective_requirements", {})
    if any(prospective.get(key) is not True for key in (
        "source_video_sha256_not_in_bound_inputs", "item_level_reuse_license_required",
        "continuous_ego_pedestrian_capture_required", "original_temporal_order_required",
        "hard_cut_or_montage_forbidden", "full_chromatic_features_frozen_before_visual_review",
        "no_post_review_threshold_window_or_rule_changes", "positive_requires_visual_corridor_risk_and_ordered_clear",
        "negative_control_opens_zero_events",
    )):
        raise ValueError("prospective requirements were weakened")
    roles = value.get("supervision_roles", {})
    if roles != {"risk_profile_and_lifecycle": "primary", "pixel_segmentation": "auxiliary_only", "distance_field": "auxiliary_only"}:
        raise ValueError("supervision roles drifted")
    auth = value.get("authorizations", {})
    if auth.get("prospective_evaluation") is not True or any(auth.get(key) is not False for key in FALSE_AUTHORIZATIONS):
        raise ValueError("unauthorized promotion flag")
    return value


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve()
    sidecar = Path(str(resolved) + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(resolved)
    actual = sha256_file(resolved)
    if sidecar.read_text(encoding="ascii").strip().lower() != actual:
        raise ValueError("radial-approach contract sidecar mismatch")
    return validate_contract(json.loads(resolved.read_text(encoding="utf-8"))), {"path": str(resolved), "sha256": actual}
