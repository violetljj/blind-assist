#!/usr/bin/env python3
"""Frozen building blocks for the P3 A2-392 temporal distillation route."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Collection

import torch
from torch import nn
from torch.nn import functional


STATES = ("CLEAR", "OCCUPIED", "UNKNOWN_GROUND")
STATE_TO_INDEX = {state: index for index, state in enumerate(STATES)}
EXPECTED_SCHEMA = "blindassist_dav2_temporal_392_student_p3_r0_clip_manifest"
ROLE_NAMES = ("train", "validation", "sealed_holdout")
EVIDENCE_WIDTH = 4


@dataclass(frozen=True)
class ClipManifestSummary:
    clips_by_role: dict[str, int]
    parents_by_role: dict[str, int]
    transitions_by_role: dict[str, int]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_clip_manifest(
    manifest: dict[str, Any],
    consumed_parent_ids: Collection[str] = (),
) -> ClipManifestSummary:
    """Validate the public training/identity manifest without opening holdout labels."""

    _require(manifest.get("schema") == EXPECTED_SCHEMA, "clip manifest schema drift")
    _require(manifest.get("clip_length") == 4, "P3 requires exactly four-frame clips")
    _require(
        manifest.get("holdout", {}).get("status") == "SEALED_NOT_OPENED"
        and manifest.get("holdout", {}).get("outcomes_opened") is False,
        "holdout must remain sealed and unopened",
    )
    clips = manifest.get("clips")
    _require(isinstance(clips, list) and clips, "clip manifest is empty")

    consumed = {str(value) for value in consumed_parent_ids}
    seen_clips: set[str] = set()
    seen_frames: set[str] = set()
    role_parents = {role: set() for role in ROLE_NAMES}
    counts = {role: 0 for role in ROLE_NAMES}

    for clip in clips:
        clip_id = str(clip.get("clip_id", ""))
        role = str(clip.get("role", ""))
        parent_id = str(clip.get("parent_id", ""))
        video_id = str(clip.get("video_id", ""))
        _require(clip_id and clip_id not in seen_clips, "duplicate or empty clip_id")
        _require(role in ROLE_NAMES, f"invalid clip role: {role}")
        _require(parent_id and video_id, f"missing parent/video identity: {clip_id}")
        _require(parent_id not in consumed, f"consumed parent reused: {parent_id}")
        seen_clips.add(clip_id)
        role_parents[role].add(parent_id)
        counts[role] += 1

        frames = clip.get("frames")
        _require(isinstance(frames, list) and len(frames) == 4, f"bad clip length: {clip_id}")
        timestamps: list[int] = []
        for frame in frames:
            frame_id = str(frame.get("frame_id", ""))
            timestamp_ns = frame.get("timestamp_ns")
            _require(frame_id and frame_id not in seen_frames, "frame reuse across clips")
            _require(
                isinstance(timestamp_ns, int) and timestamp_ns > 0,
                f"invalid real timestamp: {frame_id}",
            )
            _require(str(frame.get("video_id", "")) == video_id, "video identity drift")
            _require(str(frame.get("parent_id", "")) == parent_id, "parent identity drift")
            seen_frames.add(frame_id)
            timestamps.append(timestamp_ns)

            if role == "sealed_holdout":
                forbidden = {
                    "teacher_depth_ref",
                    "clearance_m",
                    "state",
                    "teacher_timestamp_ns",
                    "teacher_valid",
                    "tof_valid",
                }
                _require(
                    forbidden.isdisjoint(frame),
                    f"holdout label leaked into public manifest: {frame_id}",
                )
                _require("sealed_target_id" in frame, f"missing sealed target id: {frame_id}")
                continue

            _require("teacher_depth_ref" in frame, f"missing teacher depth: {frame_id}")
            clearances = frame.get("clearance_m")
            states = frame.get("state")
            _require(isinstance(clearances, list) and len(clearances) == 3, "bad clearance target")
            _require(isinstance(states, list) and len(states) == 3, "bad state target")
            _require(all(state in STATES for state in states), "unknown state vocabulary")
            teacher_timestamp_ns = frame.get("teacher_timestamp_ns")
            _require(
                isinstance(teacher_timestamp_ns, int) and teacher_timestamp_ns > 0,
                f"invalid teacher timestamp: {frame_id}",
            )
            teacher_valid = frame.get("teacher_valid")
            tof_valid = frame.get("tof_valid")
            _require(isinstance(teacher_valid, bool), "teacher_valid must be boolean")
            _require(isinstance(tof_valid, bool), "tof_valid must be boolean")
            stale = timestamp_ns - teacher_timestamp_ns > 500_000_000
            fail_closed = stale or not teacher_valid or not tof_valid
            if fail_closed:
                _require(
                    all(state == "UNKNOWN_GROUND" for state in states),
                    f"invalid evidence must target UNKNOWN_GROUND: {frame_id}",
                )
            for clearance, state in zip(clearances, states):
                if state == "UNKNOWN_GROUND":
                    _require(clearance is None, "UNKNOWN_GROUND clearance must be null")
                else:
                    _require(
                        isinstance(clearance, (int, float))
                        and math.isfinite(float(clearance))
                        and float(clearance) >= 0.0,
                        "known clearance must be finite and non-negative",
                    )

        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        _require(all(0 < gap <= 500_000_000 for gap in gaps), f"invalid clip cadence: {clip_id}")

    for index, left_role in enumerate(ROLE_NAMES):
        for right_role in ROLE_NAMES[index + 1 :]:
            _require(
                role_parents[left_role].isdisjoint(role_parents[right_role]),
                f"video-parent leakage: {left_role}/{right_role}",
            )
    _require(all(counts[role] > 0 for role in ROLE_NAMES), "all three roles are required")
    return ClipManifestSummary(
        clips_by_role=counts,
        parents_by_role={role: len(role_parents[role]) for role in ROLE_NAMES},
        transitions_by_role={role: counts[role] * 3 for role in ROLE_NAMES},
    )


class LightweightTemporalStateHead(nn.Module):
    """Non-recurrent head over independently produced per-frame depth maps."""

    def __init__(self, hidden_width: int = 64) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        frame_width = 16
        pair_width = frame_width * 3 + 1
        self.pair_projection = nn.Sequential(
            nn.Linear(pair_width, hidden_width),
            nn.GELU(),
        )
        self.clearance_delta = nn.Linear(hidden_width, 3)
        self.transition_logits = nn.Linear(hidden_width, 3 * 9)
        self.unknown_projection = nn.Sequential(
            nn.Linear(frame_width + EVIDENCE_WIDTH, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 3),
        )

    def forward(
        self,
        depth_m: torch.Tensor,
        evidence: torch.Tensor,
        delta_seconds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if depth_m.ndim != 4:
            raise ValueError("depth_m must have shape [batch, time, height, width]")
        batch, time, height, width = depth_m.shape
        if time != 4 or height < 4 or width < 4:
            raise ValueError("P3 head requires four non-trivial frames")
        if evidence.shape != (batch, time, EVIDENCE_WIDTH):
            raise ValueError("evidence must have shape [batch, 4, 4]")
        if delta_seconds.shape != (batch, time - 1):
            raise ValueError("delta_seconds must have shape [batch, 3]")
        log_depth = torch.log(torch.clamp(depth_m.float(), 0.1, 20.0))
        frame_features = self.pool(log_depth.reshape(batch * time, 1, height, width))
        frame_features = frame_features.reshape(batch, time, -1)
        pair_features = torch.cat(
            (
                frame_features[:, :-1],
                frame_features[:, 1:],
                frame_features[:, 1:] - frame_features[:, :-1],
                delta_seconds[..., None],
            ),
            dim=-1,
        )
        pair_hidden = self.pair_projection(pair_features)
        transition = self.transition_logits(pair_hidden).reshape(batch, time - 1, 3, 9)
        unknown = self.unknown_projection(torch.cat((frame_features, evidence.float()), dim=-1))
        return {
            "clearance_delta_m": self.clearance_delta(pair_hidden),
            "transition_logits": transition,
            "unknown_logits": unknown,
        }


def a2_single_frame_depth_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    beta: float = 0.05,
    gradient_weight: float = 0.5,
    scale_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The unchanged A2 log-depth, gradient, and scale supervision."""

    prediction = torch.clamp(prediction.float(), 0.1, 20.0)
    teacher = torch.clamp(teacher.float(), 0.1, 20.0)
    log_prediction = torch.log(prediction)
    log_teacher = torch.log(teacher)
    depth = functional.smooth_l1_loss(log_prediction, log_teacher, beta=beta)
    gradient = 0.5 * (
        functional.l1_loss(
            log_prediction[..., 1:] - log_prediction[..., :-1],
            log_teacher[..., 1:] - log_teacher[..., :-1],
        )
        + functional.l1_loss(
            log_prediction[..., 1:, :] - log_prediction[..., :-1, :],
            log_teacher[..., 1:, :] - log_teacher[..., :-1, :],
        )
    )
    scale = torch.median((log_prediction - log_teacher).flatten(1), dim=1).values.abs().mean()
    total = depth + gradient_weight * gradient + scale_weight * scale
    return total, {"log_depth": depth, "gradient": gradient, "scale": scale}


def temporal_distillation_loss(
    student_depth_m: torch.Tensor,
    teacher_depth_m: torch.Tensor,
    head_output: dict[str, torch.Tensor],
    target_clearance_m: torch.Tensor,
    target_state: torch.Tensor,
    teacher_age_s: torch.Tensor,
    tof_valid: torch.Tensor,
    teacher_valid: torch.Tensor,
    clearance_delta_weight: float = 1.0,
    transition_weight: float = 0.5,
    unknown_weight: float = 0.5,
    disagreement_threshold: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Four frozen supervision families; no recurrence or cadence objective."""

    if student_depth_m.shape != teacher_depth_m.shape or student_depth_m.ndim != 4:
        raise ValueError("student/teacher clip depth shapes must match [batch, 4, h, w]")
    batch, time, height, width = student_depth_m.shape
    if time != 4 or target_clearance_m.shape != (batch, time, 3):
        raise ValueError("invalid four-frame clearance target")
    if target_state.shape != (batch, time, 3):
        raise ValueError("invalid state target")

    per_frame, depth_parts = a2_single_frame_depth_loss(
        student_depth_m.reshape(batch * time, height, width),
        teacher_depth_m.reshape(batch * time, height, width),
    )
    log_student = torch.log(torch.clamp(student_depth_m.float(), 0.1, 20.0))
    log_teacher = torch.log(torch.clamp(teacher_depth_m.float(), 0.1, 20.0))
    disagreement = (log_student - log_teacher).abs().flatten(2).mean(dim=2).detach()
    frame_forced_unknown = (
        (teacher_age_s > 0.5)
        | (~tof_valid.bool())
        | (~teacher_valid.bool())
        | (disagreement > disagreement_threshold)
    )
    effective_state = target_state.long().clone()
    effective_state[
        frame_forced_unknown[..., None].expand_as(effective_state)
    ] = STATE_TO_INDEX["UNKNOWN_GROUND"]
    unknown_by_band = effective_state == STATE_TO_INDEX["UNKNOWN_GROUND"]

    target_delta = target_clearance_m[:, 1:] - target_clearance_m[:, :-1]
    delta_mask = (
        torch.isfinite(target_delta)
        & ~unknown_by_band[:, :-1]
        & ~unknown_by_band[:, 1:]
    )
    if torch.any(delta_mask):
        clearance_delta = functional.smooth_l1_loss(
            head_output["clearance_delta_m"][delta_mask],
            target_delta[delta_mask],
            beta=0.05,
        )
    else:
        # An all-UNKNOWN clip still trains transition and abstention. Keep a
        # differentiable zero so it cannot fabricate clearance evidence.
        clearance_delta = head_output["clearance_delta_m"].sum() * 0.0

    transition_target = effective_state[:, :-1] * len(STATES) + effective_state[:, 1:]
    transition = functional.cross_entropy(
        head_output["transition_logits"].reshape(-1, 9),
        transition_target.reshape(-1),
    )
    unknown_target = unknown_by_band.float()
    unknown = functional.binary_cross_entropy_with_logits(
        head_output["unknown_logits"], unknown_target
    )
    total = (
        per_frame
        + clearance_delta_weight * clearance_delta
        + transition_weight * transition
        + unknown_weight * unknown
    )
    components = {
        **depth_parts,
        "clearance_delta": clearance_delta,
        "state_transition": transition,
        "unknown": unknown,
        "total": total,
    }
    return total, components
