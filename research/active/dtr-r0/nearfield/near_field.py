"""Near-field depth evidence, independent of classes, tracks and motion commands.

Development component: requires forward metric depth and calibrated camera height
and pitch. These inputs are not provided by an arbitrary monocular RGB frame.
NO_NEAR_OBSERVED is an observation result, never a traversability certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F


HEIGHT_EDGES = (.065, .65, 1.4, 1.85)
DIRECTIONS = ("left", "center", "right")
HEIGHTS = ("low", "body", "head")


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    hfov_deg: float
    camera_height_m: float
    pitch_deg: float = 0.

    def __post_init__(self):
        if self.width < 3 or self.height < 3:
            raise ValueError("image dimensions must be >= 3")
        if not all(math.isfinite(v) for v in (
            self.hfov_deg, self.camera_height_m, self.pitch_deg
        )) or not 0 < self.hfov_deg < 180 or self.camera_height_m <= 0:
            raise ValueError("finite calibrated camera parameters required")
        if abs(self.pitch_deg) >= 80:
            raise ValueError("near-horizontal camera required by this prototype")


@dataclass
class Evidence:
    # Nearest supported visible surface per height band and image tile.
    distance_m: np.ndarray
    pixel_index: np.ndarray
    support_count: np.ndarray
    # Raw near candidates rejected by spatial support are retained as uncertainty.
    weak_count: np.ndarray
    valid_fraction: np.ndarray
    region_distance_m: np.ndarray
    region_state: list[list[str]]
    attention_m: float
    device: str

    @property
    def token_bytes(self):
        return sum(a.nbytes for a in (
            self.distance_m, self.pixel_index, self.support_count,
            self.weak_count, self.valid_fraction, self.region_distance_m,
        ))

    def alerts(self):
        return np.isfinite(self.region_distance_m) & (
            self.region_distance_m <= self.attention_m
        )

    def summary(self):
        return {
            "coordinate_frame": "camera_heading_gravity_aligned",
            "attention_m": self.attention_m,
            "regions": [{
                "direction": direction, "height": height,
                "state": self.region_state[d][h],
                "distance_m": float(self.region_distance_m[d, h])
                    if np.isfinite(self.region_distance_m[d, h]) else None,
            } for d, direction in enumerate(DIRECTIONS)
                for h, height in enumerate(HEIGHTS)],
            "token_bytes": self.token_bytes, "device": self.device,
        }


class NearFieldEncoder:
    """Keep spatially supported minima before coarsening, using one dense scan.

    Three collinear, depth-consistent pixels support a surface, preserving a
    one-pixel-wide line. This rejects isolated noise, NOT coherent depth errors.
    No learning, depth-scale fitting, object labels or evaluator inputs are used.
    """

    def __init__(self, camera: Camera, *, device="cpu", tile=16, attention_m=3.):
        if not 1 <= tile <= 64 or not 0 < attention_m < 12:
            raise ValueError("tile must be 1..64 and attention range 0..12m")
        self.camera, self.device = camera, torch.device(device)
        self.tile, self.attention_m = tile, attention_m
        self.fx = camera.width / (2 * math.tan(math.radians(camera.hfov_deg / 2)))
        v, u = torch.meshgrid(
            torch.arange(camera.height, device=self.device, dtype=torch.float32),
            torch.arange(camera.width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        self.ray_y = (u - (camera.width - 1) / 2) / self.fx
        ray_up = -(v - (camera.height - 1) / 2) / self.fx
        pitch = math.radians(camera.pitch_deg)
        self.ray_x = math.cos(pitch) - math.sin(pitch) * ray_up
        self.ray_z = math.sin(pitch) + math.cos(pitch) * ray_up
        angle = torch.atan2(self.ray_y, self.ray_x)
        self.direction = (angle >= math.radians(-10)).long() + (
            angle >= math.radians(10)
        ).long()

    @torch.inference_mode()
    def encode(self, depth: np.ndarray, *, ground_plane=None) -> Evidence:
        c = self.camera
        if depth.shape != (c.height, c.width):
            raise ValueError("depth/calibration dimensions disagree")
        z = torch.as_tensor(np.ascontiguousarray(depth), dtype=torch.float32,
                            device=self.device)
        valid = torch.isfinite(z) & (z > .08) & (z < 12) & (self.ray_x > 0)
        x = z * self.ray_x
        if ground_plane is None:
            height = c.camera_height_m + z * self.ray_z
        else:
            # Camera-relative gravity frame: Z=aX+bY+c. Use vertical height,
            # not normal distance; callers scale the intercept with the depth.
            if len(ground_plane) != 3 or not all(math.isfinite(v) for v in ground_plane):
                raise ValueError("ground plane requires three finite coefficients")
            a, b, intercept = ground_plane
            height = z*self.ray_z - a*x - b*z*self.ray_y - intercept
        band = torch.bucketize(height, torch.tensor(
            HEIGHT_EDGES, dtype=z.dtype, device=self.device
        ), right=True) - 1
        obstacle = valid & (band >= 0) & (band < 3)

        support = torch.zeros_like(valid)
        # A triple must consist of eligible surfaces with compatible depth. Mark
        # endpoints too; requiring two neighbors at every pixel would erode lines.
        for axis in (0, 1):
            slices = []
            for start, stop in ((0, -2), (1, -1), (2, None)):
                sl = [slice(None), slice(None)]
                sl[axis] = slice(start, stop)
                slices.append(tuple(sl))
            a, b, d = (z[sl] for sl in slices)
            compatible = (torch.maximum(torch.maximum(a, b), d) -
                          torch.minimum(torch.minimum(a, b), d)) <= (.08 + .02*b)
            run = compatible & obstacle[slices[0]] & obstacle[slices[1]] & obstacle[slices[2]]
            for sl in slices:
                support[sl] |= run

        masks = torch.stack([support & (band == k) for k in range(3)])
        distances, indices = F.max_pool2d(
            -torch.where(masks, x, torch.inf), self.tile, self.tile,
            ceil_mode=True, return_indices=True,
        )
        distances = -distances
        count = F.avg_pool2d(masks.float(), self.tile, self.tile,
                             ceil_mode=True, divisor_override=1)
        weak = torch.stack([
            obstacle & ~support & (band == k) & (x <= self.attention_m)
            for k in range(3)
        ])
        weak_count = F.avg_pool2d(weak.float(), self.tile, self.tile,
                                  ceil_mode=True, divisor_override=1)
        fraction = F.avg_pool2d(valid[None].float(), self.tile, self.tile,
                                ceil_mode=True)

        # Keep nine additional exact region minima: one image tile can straddle
        # an angular boundary, and retaining only its closest pixel could erase
        # evidence on the other side. These 36 bytes are part of token_bytes.
        region_distance = torch.full((3, 3), torch.inf, device=self.device)
        region_weak = torch.zeros((3, 3), dtype=torch.bool, device=self.device)
        coverage = torch.zeros(3, device=self.device)
        for direction in range(3):
            direction_mask = self.direction == direction
            denominator = direction_mask.sum().clamp_min(1)
            coverage[direction] = (valid & direction_mask).sum() / denominator
            for k in range(3):
                region_distance[direction, k] = torch.where(
                    masks[k] & direction_mask, x, torch.inf
                ).amin()
                region_weak[direction, k] = (weak[k] & direction_mask).any()

        # Host transfer and region summaries belong to encode timing.
        rd = region_distance.cpu().numpy()
        rw, cv = region_weak.cpu().numpy(), coverage.cpu().numpy()
        states = [[
            "OBSTACLE" if rd[d, k] <= self.attention_m else
            "UNKNOWN" if rw[d, k] or cv[d] < .5 else "NO_NEAR_OBSERVED"
            for k in range(3)] for d in range(3)]
        return Evidence(
            distances.cpu().numpy(), indices.cpu().numpy().astype(np.uint32),
            count.cpu().numpy().astype(np.uint16),
            weak_count.cpu().numpy().astype(np.uint16),
            (fraction.cpu().numpy()*255).round().astype(np.uint8),
            rd, states, self.attention_m, str(self.device),
        )


def quantile_baseline(depth: np.ndarray, camera: Camera, *, stride=4):
    """Task-region adapter of legacy 4% quantile / >=8 sample aggregation.

    Both baselines use the same angle/height/attention definitions as the new
    encoder. This is not the unchanged navigation controller or its score.
    """
    offset = 2 if stride == 4 else 0
    v, u = np.mgrid[offset:camera.height:stride, offset:camera.width:stride]
    z = depth[v, u]
    fx = camera.width / (2*math.tan(math.radians(camera.hfov_deg/2)))
    y = z*(u-(camera.width-1)/2)/fx
    up = -z*(v-(camera.height-1)/2)/fx
    pitch = math.radians(camera.pitch_deg)
    x = math.cos(pitch)*z-math.sin(pitch)*up
    height = camera.camera_height_m+math.sin(pitch)*z+math.cos(pitch)*up
    valid = np.isfinite(z) & (z>.08) & (z<12) & (x>0)
    direction = np.searchsorted(np.radians([-10, 10]), np.arctan2(y, x), side="right")
    band = np.searchsorted(HEIGHT_EDGES, height, side="right")-1
    result = np.full((3, 3), np.inf, dtype=np.float32)
    for d in range(3):
        for k in range(3):
            values = x[valid & (direction==d) & (band==k)]
            if len(values) >= 8:
                result[d,k] = np.quantile(values, .04)
    return result
