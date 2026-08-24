"""Observation contracts shared by synthetic and real-RGB SAGE-LM providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ApertureObservation:
    visible: bool
    center_x_m: float | None
    width_m: float | None
    range_m: float | None
    boundary_confidence: float
    flow_confidence: float
    depth_consistency: float
    geometry_confidence: float


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ExactAnchorObservation:
    frame_index: int
    referent_id: str
    bbox_xyxy: tuple[int, int, int, int] | None

    @property
    def visible(self) -> bool:
        return self.bbox_xyxy is not None


@dataclass(frozen=True)
class RgbEpisodeInput:
    """The complete input surface available to ``RgbObservationProvider``.

    Paths point only to composited RGB frames.  Source depth, pose, mesh and
    evaluator aperture labels deliberately have no field in this type.
    """

    episode_id: str
    kind: str
    rgb_frames: tuple[Path, ...]
    intrinsics: CameraIntrinsics
    commanded_baseline_m: float
    active_parallax_frame_index: int
    exact_anchor_observations: tuple[ExactAnchorObservation, ...]


@dataclass(frozen=True)
class RgbEpisodeTruth:
    """Evaluator-only truth; never accepted by an observation provider."""

    episode_id: str
    aperture_center_x_m: float
    aperture_width_m: float
    start_range_m: float
    camera_positions_m: tuple[tuple[float, float, float], ...]
    endpoint_center_x_m: float
    source_boundary_x_px: tuple[float, float] | None = None


class ObservationProvider(Protocol):
    input: RgbEpisodeInput
    diagnostics: dict

    def observe(self) -> ApertureObservation: ...
