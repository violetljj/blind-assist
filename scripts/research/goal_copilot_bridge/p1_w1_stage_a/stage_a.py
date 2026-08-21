"""Fail-closed C0/T0 world-referent mechanics for the frozen P1-W1 Stage A.

This module deliberately does not read images, pose truth, evaluator labels, or
future frames.  Geometry and observation providers must emit the small causal
evidence record below.  The same observation and identity fields are consumed
by both arms; only the spatial evidence field differs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Arm = Literal["C0", "W1-T0"]
TriState = Literal["SUPPORTED", "REJECTED", "INSUFFICIENT"]


@dataclass(frozen=True)
class SpatialEvidence:
    reference_frame: Literal["CAMERA_RELATIVE", "KEYFRAME_RELATIVE"]
    geometry_supported: bool
    motion_observable: bool
    translation_overreach: bool
    geometry_degenerate: bool
    bearing_estimate_deg: float | None
    bearing_uncertainty_deg: float | None
    compatibility: TriState


@dataclass(frozen=True)
class FrameEvidence:
    frame_id: str
    observation_supported: bool
    candidate_region_xyxy: tuple[float, float, float, float] | None
    independent_identity_confirmation: TriState
    observability_reason: Literal[
        "IN_VIEW_CANDIDATE", "OUT_OF_VIEW", "OCCLUDED_EVIDENCED", "NO_OBSERVATION", "UNKNOWN"
    ]
    c0_spatial: SpatialEvidence
    t0_spatial: SpatialEvidence


def _validate_spatial(arm: Arm, value: SpatialEvidence) -> None:
    expected = "CAMERA_RELATIVE" if arm == "C0" else "KEYFRAME_RELATIVE"
    if value.reference_frame != expected:
        raise ValueError(f"{arm}: expected {expected} evidence")
    if (value.bearing_estimate_deg is None) != (value.bearing_uncertainty_deg is None):
        raise ValueError(f"{arm}: bearing and uncertainty must be present together")
    if value.bearing_uncertainty_deg is not None and value.bearing_uncertainty_deg < 0:
        raise ValueError(f"{arm}: negative bearing uncertainty")


def step(
    arm: Arm,
    referent_id: str,
    evidence: FrameEvidence,
    *,
    previous_observation_state: Literal["NONE", "SUPPORTED"],
) -> dict:
    """Produce one ReferentSnapshot without retaining or fabricating a bbox."""
    if not referent_id:
        raise ValueError("an established referent_id is required")
    spatial = evidence.c0_spatial if arm == "C0" else evidence.t0_spatial
    _validate_spatial(arm, spatial)
    if evidence.observation_supported != (evidence.candidate_region_xyxy is not None):
        raise ValueError("candidate_region must be present iff observation is supported")

    stale = (
        not spatial.motion_observable
        or not spatial.geometry_supported
        or spatial.translation_overreach
        or spatial.geometry_degenerate
        or spatial.bearing_estimate_deg is None
    )
    anchor_state = "STALE" if stale else "GOOD"
    compatibility: TriState = "INSUFFICIENT" if stale else spatial.compatibility
    identity = evidence.independent_identity_confirmation
    reacquired = (
        previous_observation_state == "NONE"
        and
        evidence.observation_supported
        and compatibility == "SUPPORTED"
        and identity == "SUPPORTED"
    )

    return {
        "referent_id": referent_id,
        "identity_state": "DEGRADED" if identity == "REJECTED" else "VALID",
        "spatial_anchor_state": anchor_state,
        "observation_state": "SUPPORTED" if evidence.observation_supported else "NONE",
        "observability_reason": evidence.observability_reason,
        "reference_frame": spatial.reference_frame,
        "candidate_region": list(evidence.candidate_region_xyxy) if evidence.observation_supported else None,
        "bearing_estimate": None if stale else spatial.bearing_estimate_deg,
        "bearing_uncertainty": None if stale else spatial.bearing_uncertainty_deg,
        "spatial_compatibility": compatibility,
        "independent_identity_confirmation": identity,
        "reacquisition_status": "REACQUIRED" if reacquired else "NOT_REACQUIRED",
        "directional_guidance_authorized": not stale,
    }


def evidence_to_dict(value: FrameEvidence) -> dict:
    """Stable serialization helper used by future public traces."""
    return asdict(value)


HARD_ZERO_METRICS = (
    "fabricated_observation",
    "single_channel_reacquisition",
    "stale_anchor_guidance_use",
    "post_initialization_truth_leakage",
    "future_frame_access",
)


def adjudicate_stage_a(c0: dict, t0: dict, support_counts: dict[str, int]) -> str:
    """Apply the frozen Stage-A verdict order to already-derived metrics."""
    required_support = (
        "ROTATION_DOMINANT", "SMALL_TRANSLATION", "TRANSLATION_BEYOND_TIER0",
        "OCCLUSION_OR_REAPPEARANCE", "IDENTITY_CONFUSER", "OBSERVATION_LOSS",
        "GEOMETRY_DEGENERATE",
    )
    if any(support_counts.get(bucket, 0) == 0 for bucket in required_support):
        return "W1_T0_NOT_EVALUABLE_DATA_OR_INTERFACE"
    for arm in (c0, t0):
        if any(arm.get(metric, 0) != 0 for metric in HARD_ZERO_METRICS):
            return "W1_T0_NOT_SUPPORTED"
    if not t0.get("translation_overreach_timely_stale", False) or not t0.get("geometry_degenerate_timely_stale", False):
        return "W1_T0_NOT_SUPPORTED"
    if t0["false_reacquisition"] > c0["false_reacquisition"]:
        return "W1_T0_NOT_SUPPORTED"
    if (
        t0["abstention_count"] > c0["abstention_count"]
        and t0["identity_confirmed_reacquisition"] < c0["identity_confirmed_reacquisition"]
        and t0["usable_anchor_coverage"] < c0["usable_anchor_coverage"]
    ):
        return "W1_T0_HONESTY_GAIN_ONLY_BY_ABSTENTION"
    if t0["identity_confirmed_reacquisition"] < c0["identity_confirmed_reacquisition"]:
        return "W1_T0_NOT_SUPPORTED"
    if t0["bearing_compatibility_rate"] < c0["bearing_compatibility_rate"]:
        return "W1_T0_NOT_SUPPORTED"
    safety_or_coverage_gain = (
        t0["false_continuity"] < c0["false_continuity"]
        or (
            t0["false_continuity"] == c0["false_continuity"] == 0
            and t0["usable_anchor_coverage"] > c0["usable_anchor_coverage"]
        )
    )
    if not safety_or_coverage_gain:
        return "W1_T0_NOT_SUPPORTED"
    return "W1_T0_WORLD_REFERENT_SIGNAL_ESTABLISHED"
