#!/usr/bin/env python3
"""Pre-activation-corrected P3 temporal supervision primitives."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Collection

import torch
from torch import nn
from torch.nn import functional


STATES = ("CLEAR", "OCCUPIED", "UNKNOWN_GROUND")
STATE_TO_INDEX = {state: index for index, state in enumerate(STATES)}
TRANSITIONS = tuple(f"{left}_TO_{right}" for left in STATES for right in STATES)
EXPECTED_MANIFEST_SCHEMA = (
    "blindassist_dav2_temporal_392_student_p3_r0_1_clip_manifest"
)
EXPECTED_COVERAGE_SCHEMA = (
    "blindassist_dav2_temporal_392_student_p3_r0_1_sealed_coverage_receipt"
)
ROLES = ("train", "validation", "sealed_holdout")
EVIDENCE_FIELDS = (
    "teacher_age_s",
    "tof_valid",
    "teacher_valid",
    "frozen_a2_mean_abs_log_depth_disagreement",
)
SHA256_RE = re.compile(r"^[0-9A-F]{64}$")

TOP_LEVEL_FIELDS = frozenset(
    {"schema", "protocol_sha256", "clip_length", "holdout", "clips"}
)
HOLDOUT_FIELDS = frozenset({"status", "outcomes_opened"})
CLIP_FIELDS = frozenset({"clip_id", "role", "video_id", "parent_id", "frames"})
HOLDOUT_FRAME_FIELDS = frozenset(
    {
        "frame_id",
        "video_id",
        "parent_id",
        "timestamp_ns",
        "sealed_target_id",
        "rgb_identity",
        "rgb_sha256",
    }
)
TRAIN_FRAME_FIELDS = frozenset(
    {
        "frame_id",
        "video_id",
        "parent_id",
        "timestamp_ns",
        "rgb_identity",
        "rgb_sha256",
        "teacher_depth_ref",
        "teacher_depth_sha256",
        "teacher_timestamp_ns",
        "teacher_valid",
        "tof_valid",
        "frozen_a2_mean_abs_log_depth_disagreement",
        "clearance_m",
        "geometry_state",
        "geometry_target_valid",
    }
)


@dataclass(frozen=True)
class TemporalEvidence:
    """The sole evidence object consumed by both the head and all loss masks."""

    head_features: torch.Tensor
    teacher_age_s: torch.Tensor
    teacher_frame_usable: torch.Tensor
    clearance_frame_usable: torch.Tensor
    external_abstain_target: torch.Tensor
    frozen_a2_disagreement: torch.Tensor


@dataclass(frozen=True)
class ClipManifestSummary:
    clips_by_role: dict[str, int]
    parents_by_role: dict[str, int]
    transitions_by_role: dict[str, dict[str, int]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_fields(value: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    _require(observed == allowed, f"{label} fields must be exact: {sorted(observed ^ allowed)}")


def _sha(value: Any, label: str) -> str:
    normalized = str(value).upper()
    _require(bool(SHA256_RE.fullmatch(normalized)), f"invalid {label} SHA-256")
    return normalized


def build_temporal_evidence(
    sample_timestamp_ns: torch.Tensor,
    teacher_timestamp_ns: torch.Tensor,
    teacher_valid: torch.Tensor,
    tof_valid: torch.Tensor,
    frozen_a2_disagreement: torch.Tensor,
    *,
    stale_after_s: float = 0.5,
    disagreement_threshold: float = 0.20,
) -> TemporalEvidence:
    """Build head features, abstention targets, and masks exactly once.

    `frozen_a2_disagreement` must be precomputed by the hash-bound parent A2
    checkpoint. The current trainable student is intentionally absent.
    """

    shapes = {
        tuple(sample_timestamp_ns.shape),
        tuple(teacher_timestamp_ns.shape),
        tuple(teacher_valid.shape),
        tuple(tof_valid.shape),
        tuple(frozen_a2_disagreement.shape),
    }
    if len(shapes) != 1 or sample_timestamp_ns.ndim != 2:
        raise ValueError("all temporal evidence inputs must share [batch, time]")
    if sample_timestamp_ns.shape[1] != 4:
        raise ValueError("P3 R0.1 requires four-frame evidence")
    age_s = (
        sample_timestamp_ns.to(torch.float64)
        - teacher_timestamp_ns.to(torch.float64)
    ) / 1_000_000_000.0
    if torch.any(~torch.isfinite(age_s)) or torch.any(age_s < 0.0):
        raise ValueError("teacher age must be finite and causal")
    disagreement = frozen_a2_disagreement.detach().float()
    if torch.any(~torch.isfinite(disagreement)) or torch.any(disagreement < 0.0):
        raise ValueError("frozen A2 disagreement must be finite and non-negative")
    teacher_ok = teacher_valid.bool() & (age_s <= stale_after_s)
    tof_ok = tof_valid.bool()
    abstain = (
        ~teacher_ok
        | ~tof_ok
        | (disagreement > disagreement_threshold)
    )
    head_features = torch.stack(
        (
            torch.clamp(age_s.float() / stale_after_s, 0.0, 2.0),
            tof_ok.float(),
            teacher_valid.bool().float(),
            disagreement,
        ),
        dim=-1,
    ).detach()
    return TemporalEvidence(
        head_features=head_features,
        teacher_age_s=age_s.float().detach(),
        teacher_frame_usable=teacher_ok.detach(),
        clearance_frame_usable=(teacher_ok & tof_ok).detach(),
        external_abstain_target=abstain.detach(),
        frozen_a2_disagreement=disagreement,
    )


def effective_number_transition_weights(
    transition_counts: dict[str, int], *, beta: float = 0.999
) -> torch.Tensor:
    """Derive the sole allowed nine-class weights from frozen train counts."""

    if set(transition_counts) != set(TRANSITIONS):
        raise ValueError("all nine transition counts are required")
    counts = torch.tensor(
        [int(transition_counts[name]) for name in TRANSITIONS], dtype=torch.float64
    )
    if torch.any(counts <= 0):
        raise ValueError("every transition class needs positive train support")
    if not 0.0 < beta < 1.0:
        raise ValueError("effective-number beta must be in (0,1)")
    weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))
    return (weights / weights.mean()).float()


def activation_transition_weights(summary: ClipManifestSummary) -> torch.Tensor:
    """Require nine-class support in train and validation before activation."""

    for role in ("train", "validation"):
        missing = [
            name for name, count in summary.transitions_by_role[role].items() if count <= 0
        ]
        if missing:
            raise ValueError(f"{role} lacks transition support: {missing}")
    return effective_number_transition_weights(summary.transitions_by_role["train"])


class DecoupledTemporalStateHead(nn.Module):
    """Geometry transition head plus an evidence-aware external abstention head."""

    def __init__(self, hidden_width: int = 64) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.pair_projection = nn.Sequential(
            nn.Linear(16 * 3 + 1, hidden_width), nn.GELU()
        )
        self.clearance_delta = nn.Linear(hidden_width, 3)
        self.geometry_transition_logits = nn.Linear(hidden_width, 3 * 9)
        self.abstention_projection = nn.Sequential(
            nn.Linear(16 + len(EVIDENCE_FIELDS), hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 3),
        )

    def forward(
        self,
        depth_m: torch.Tensor,
        evidence: TemporalEvidence,
        delta_seconds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if depth_m.ndim != 4:
            raise ValueError("depth_m must have [batch, time, height, width]")
        batch, time, height, width = depth_m.shape
        if time != 4 or min(height, width) < 4:
            raise ValueError("P3 R0.1 head requires four non-trivial frames")
        if evidence.head_features.shape != (batch, time, len(EVIDENCE_FIELDS)):
            raise ValueError("evidence was not built for this clip batch")
        if delta_seconds.shape != (batch, time - 1):
            raise ValueError("delta_seconds must have [batch, 3]")
        log_depth = torch.log(torch.clamp(depth_m.float(), 0.1, 20.0))
        frame = self.pool(log_depth.reshape(batch * time, 1, height, width))
        frame = frame.reshape(batch, time, -1)
        pair = torch.cat(
            (
                frame[:, :-1],
                frame[:, 1:],
                frame[:, 1:] - frame[:, :-1],
                delta_seconds[..., None],
            ),
            dim=-1,
        )
        hidden = self.pair_projection(pair)
        return {
            "clearance_delta_m": self.clearance_delta(hidden),
            "geometry_transition_logits": self.geometry_transition_logits(hidden).reshape(
                batch, time - 1, 3, 9
            ),
            "external_abstention_logits": self.abstention_projection(
                torch.cat((frame, evidence.head_features), dim=-1)
            ),
        }


def masked_a2_depth_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    teacher_pixel_valid: torch.Tensor,
    evidence: TemporalEvidence,
    *,
    beta: float = 0.05,
    gradient_weight: float = 0.5,
    scale_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A2 loss with frame and pixel validity applied to every component."""

    if prediction.shape != teacher.shape or prediction.ndim != 4:
        raise ValueError("depth clips must share [batch, 4, height, width]")
    if teacher_pixel_valid.shape != teacher.shape:
        raise ValueError("teacher_pixel_valid must match depth clips")
    batch, time, height, width = prediction.shape
    frame_mask = evidence.teacher_frame_usable[..., None, None]
    valid = (
        teacher_pixel_valid.bool()
        & frame_mask
        & torch.isfinite(teacher)
        & (teacher >= 0.1)
        & (teacher <= 20.0)
        & torch.isfinite(prediction)
        & (prediction > 0.0)
    )
    prediction = torch.clamp(prediction.float(), 0.1, 20.0)
    teacher = torch.clamp(teacher.float(), 0.1, 20.0)
    log_prediction = torch.log(prediction)
    log_teacher = torch.log(teacher)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if torch.any(mask):
            return values[mask].mean()
        return prediction.sum() * 0.0

    depth_values = functional.smooth_l1_loss(
        log_prediction, log_teacher, beta=beta, reduction="none"
    )
    depth = masked_mean(depth_values, valid)
    dx_mask = valid[..., 1:] & valid[..., :-1]
    dy_mask = valid[..., 1:, :] & valid[..., :-1, :]
    dx = (
        log_prediction[..., 1:] - log_prediction[..., :-1]
        - log_teacher[..., 1:]
        + log_teacher[..., :-1]
    ).abs()
    dy = (
        log_prediction[..., 1:, :] - log_prediction[..., :-1, :]
        - log_teacher[..., 1:, :]
        + log_teacher[..., :-1, :]
    ).abs()
    gradient = 0.5 * (masked_mean(dx, dx_mask) + masked_mean(dy, dy_mask))
    scale_values: list[torch.Tensor] = []
    log_ratio = log_prediction - log_teacher
    for batch_index in range(batch):
        for time_index in range(time):
            mask = valid[batch_index, time_index]
            if torch.any(mask):
                scale_values.append(
                    torch.median(log_ratio[batch_index, time_index][mask]).abs()
                )
    scale = (
        torch.stack(scale_values).mean()
        if scale_values
        else prediction.sum() * 0.0
    )
    total = depth + gradient_weight * gradient + scale_weight * scale
    return total, {
        "log_depth": depth,
        "gradient": gradient,
        "scale": scale,
        "valid_depth_pixels": valid.sum().detach(),
    }


def temporal_distillation_loss(
    student_depth_m: torch.Tensor,
    teacher_depth_m: torch.Tensor,
    teacher_pixel_valid: torch.Tensor,
    evidence: TemporalEvidence,
    head_output: dict[str, torch.Tensor],
    target_clearance_m: torch.Tensor,
    geometry_state_target: torch.Tensor,
    geometry_target_valid: torch.Tensor,
    geometry_transition_class_weights: torch.Tensor,
    *,
    clearance_delta_weight: float = 1.0,
    geometry_transition_weight: float = 0.5,
    external_abstention_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Four supervision families with fixed, mutually consistent masks."""

    batch, time, _height, _width = student_depth_m.shape
    if time != 4 or target_clearance_m.shape != (batch, time, 3):
        raise ValueError("invalid clearance targets")
    if geometry_state_target.shape != (batch, time, 3):
        raise ValueError("invalid geometry state targets")
    if geometry_target_valid.shape != (batch, time, 3):
        raise ValueError("invalid geometry target validity")
    if geometry_transition_class_weights.shape != (9,):
        raise ValueError("nine geometry transition class weights are required")
    depth, depth_parts = masked_a2_depth_loss(
        student_depth_m,
        teacher_depth_m,
        teacher_pixel_valid,
        evidence,
    )

    target_delta = target_clearance_m[:, 1:] - target_clearance_m[:, :-1]
    clearance_mask = (
        torch.isfinite(target_delta)
        & evidence.clearance_frame_usable[:, :-1, None]
        & evidence.clearance_frame_usable[:, 1:, None]
    )
    if torch.any(clearance_mask):
        clearance_delta = functional.smooth_l1_loss(
            head_output["clearance_delta_m"][clearance_mask],
            target_delta[clearance_mask],
            beta=0.05,
        )
    else:
        clearance_delta = head_output["clearance_delta_m"].sum() * 0.0

    transition_target = (
        geometry_state_target[:, :-1].long() * len(STATES)
        + geometry_state_target[:, 1:].long()
    )
    transition_mask = (
        geometry_target_valid[:, :-1].bool()
        & geometry_target_valid[:, 1:].bool()
        & evidence.teacher_frame_usable[:, :-1, None]
        & evidence.teacher_frame_usable[:, 1:, None]
    )
    logits = head_output["geometry_transition_logits"]
    if torch.any(transition_mask):
        geometry_transition = functional.cross_entropy(
            logits[transition_mask],
            transition_target[transition_mask],
            weight=geometry_transition_class_weights.to(logits.device, logits.dtype),
        )
    else:
        geometry_transition = logits.sum() * 0.0

    abstention_target = evidence.external_abstain_target[..., None].expand(-1, -1, 3)
    external_abstention = functional.binary_cross_entropy_with_logits(
        head_output["external_abstention_logits"], abstention_target.float()
    )
    total = (
        depth
        + clearance_delta_weight * clearance_delta
        + geometry_transition_weight * geometry_transition
        + external_abstention_weight * external_abstention
    )
    return total, {
        **depth_parts,
        "clearance_delta": clearance_delta,
        "clearance_delta_pairs": clearance_mask.sum().detach(),
        "geometry_transition": geometry_transition,
        "geometry_transition_pairs": transition_mask.sum().detach(),
        "external_abstention": external_abstention,
        "total": total,
    }


def validate_clip_manifest(
    manifest: dict[str, Any], consumed_parent_ids: Collection[str]
) -> ClipManifestSummary:
    """Validate exact schemas; sealed holdout accepts no renamed label fields."""

    _exact_fields(manifest, TOP_LEVEL_FIELDS, "manifest")
    _require(manifest["schema"] == EXPECTED_MANIFEST_SCHEMA, "manifest schema drift")
    _sha(manifest["protocol_sha256"], "protocol")
    _require(manifest["clip_length"] == 4, "clip length must be exactly four")
    holdout = manifest["holdout"]
    _require(isinstance(holdout, dict), "holdout block missing")
    _exact_fields(holdout, HOLDOUT_FIELDS, "holdout")
    _require(
        holdout["status"] == "SEALED_NOT_OPENED"
        and holdout["outcomes_opened"] is False,
        "holdout must remain sealed",
    )
    clips = manifest["clips"]
    _require(isinstance(clips, list) and clips, "clip list is empty")

    consumed = {str(value) for value in consumed_parent_ids}
    role_parents = {role: set() for role in ROLES}
    role_counts = {role: 0 for role in ROLES}
    transition_counts = {
        role: {transition: 0 for transition in TRANSITIONS} for role in ROLES
    }
    seen_clips: set[str] = set()
    seen_frames: set[str] = set()
    for clip in clips:
        _require(isinstance(clip, dict), "clip must be an object")
        _exact_fields(clip, CLIP_FIELDS, "clip")
        clip_id = str(clip["clip_id"])
        role = str(clip["role"])
        video_id = str(clip["video_id"])
        parent_id = str(clip["parent_id"])
        _require(clip_id and clip_id not in seen_clips, "duplicate clip id")
        _require(role in ROLES, "invalid clip role")
        _require(parent_id and parent_id not in consumed, "consumed parent reused")
        seen_clips.add(clip_id)
        role_parents[role].add(parent_id)
        role_counts[role] += 1
        frames = clip["frames"]
        _require(isinstance(frames, list) and len(frames) == 4, "bad clip length")
        timestamps: list[int] = []
        states: list[list[str]] = []
        validities: list[list[bool]] = []
        for frame in frames:
            _require(isinstance(frame, dict), "frame must be an object")
            allowed = HOLDOUT_FRAME_FIELDS if role == "sealed_holdout" else TRAIN_FRAME_FIELDS
            _exact_fields(frame, allowed, f"{role} frame")
            frame_id = str(frame["frame_id"])
            timestamp = frame["timestamp_ns"]
            _require(frame_id and frame_id not in seen_frames, "frame reused")
            _require(isinstance(timestamp, int) and timestamp > 0, "invalid timestamp")
            _require(frame["video_id"] == video_id, "video identity drift")
            _require(frame["parent_id"] == parent_id, "parent identity drift")
            _sha(frame["rgb_sha256"], "RGB")
            seen_frames.add(frame_id)
            timestamps.append(timestamp)
            if role == "sealed_holdout":
                _require(str(frame["sealed_target_id"]), "sealed target id missing")
                continue
            _sha(frame["teacher_depth_sha256"], "teacher depth")
            teacher_timestamp = frame["teacher_timestamp_ns"]
            _require(
                isinstance(teacher_timestamp, int)
                and 0 < teacher_timestamp <= timestamp,
                "teacher timestamp must be causal",
            )
            _require(isinstance(frame["teacher_valid"], bool), "bad teacher_valid")
            _require(isinstance(frame["tof_valid"], bool), "bad tof_valid")
            disagreement = frame["frozen_a2_mean_abs_log_depth_disagreement"]
            _require(
                isinstance(disagreement, (int, float))
                and math.isfinite(float(disagreement))
                and float(disagreement) >= 0.0,
                "bad frozen A2 disagreement",
            )
            clearance = frame["clearance_m"]
            geometry_state = frame["geometry_state"]
            target_valid = frame["geometry_target_valid"]
            _require(isinstance(clearance, list) and len(clearance) == 3, "bad clearance")
            _require(
                isinstance(geometry_state, list)
                and len(geometry_state) == 3
                and all(value in STATES for value in geometry_state),
                "bad geometry state",
            )
            _require(
                isinstance(target_valid, list)
                and len(target_valid) == 3
                and all(isinstance(value, bool) for value in target_valid),
                "bad geometry target validity",
            )
            for value in clearance:
                _require(
                    value is None
                    or (
                        isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        and float(value) >= 0.0
                    ),
                    "clearance must be null or finite and non-negative",
                )
            states.append(geometry_state)
            validities.append(target_valid)
        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        _require(all(0 < gap <= 500_000_000 for gap in gaps), "invalid clip cadence")
        if role != "sealed_holdout":
            for pair_index, (previous, current) in enumerate(zip(states, states[1:])):
                for band, (left, right) in enumerate(zip(previous, current)):
                    if validities[pair_index][band] and validities[pair_index + 1][band]:
                        transition_counts[role][f"{left}_TO_{right}"] += 1

    for index, left in enumerate(ROLES):
        for right in ROLES[index + 1 :]:
            _require(role_parents[left].isdisjoint(role_parents[right]), "parent leakage")
    _require(all(role_counts[role] > 0 for role in ROLES), "all roles are required")
    return ClipManifestSummary(
        clips_by_role=role_counts,
        parents_by_role={role: len(role_parents[role]) for role in ROLES},
        transitions_by_role=transition_counts,
    )


def validate_sealed_coverage_receipt(
    receipt: dict[str, Any],
    *,
    expected_identity_manifest_sha256: str,
    expected_protocol_sha256: str,
) -> None:
    allowed = frozenset(
        {
            "schema",
            "status",
            "protocol_sha256",
            "identity_manifest_sha256",
            "sealed_bundle_sha256",
            "coverage_producer_sha256",
            "created_before_training_activation",
            "label_rows_disclosed",
            "evaluable_clip_count",
            "video_parent_count",
            "key_transition_counts",
            "geometry_transition_counts",
        }
    )
    _exact_fields(receipt, allowed, "coverage receipt")
    _require(receipt["schema"] == EXPECTED_COVERAGE_SCHEMA, "coverage schema drift")
    _require(receipt["status"] == "SEALED_COVERAGE_VERIFIED", "coverage not verified")
    _require(
        _sha(receipt["protocol_sha256"], "coverage protocol")
        == expected_protocol_sha256.upper(),
        "coverage protocol mismatch",
    )
    _require(
        _sha(receipt["identity_manifest_sha256"], "coverage identity")
        == expected_identity_manifest_sha256.upper(),
        "coverage identity mismatch",
    )
    _sha(receipt["sealed_bundle_sha256"], "sealed bundle")
    _sha(receipt["coverage_producer_sha256"], "coverage producer")
    _require(receipt["created_before_training_activation"] is True, "late coverage receipt")
    _require(receipt["label_rows_disclosed"] is False, "coverage receipt leaks labels")
    _require(int(receipt["evaluable_clip_count"]) >= 32, "insufficient holdout clips")
    _require(int(receipt["video_parent_count"]) >= 8, "insufficient holdout parents")
    required = {
        "CLEAR_TO_OCCUPIED",
        "OCCUPIED_TO_CLEAR",
        "KNOWN_TO_UNKNOWN_GROUND",
        "UNKNOWN_GROUND_TO_KNOWN",
    }
    counts = receipt["key_transition_counts"]
    _require(isinstance(counts, dict) and set(counts) == required, "key coverage drift")
    _require(all(int(counts[key]) >= 8 for key in required), "insufficient key transitions")
    geometry = receipt["geometry_transition_counts"]
    _require(
        isinstance(geometry, dict) and set(geometry) == set(TRANSITIONS),
        "nine-class transition distribution missing",
    )
    _require(all(int(value) >= 0 for value in geometry.values()), "negative transition count")
