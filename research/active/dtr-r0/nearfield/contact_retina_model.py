"""NF-G7 bounded contact learner; inputs are RGB and privileged metric ego poses.

BODY then HEAD, each at horizons 1/2/3 seconds. No object identity, boxes,
native depth, future frame, label, or absolute world position enters features.
The structure transform is fixed, computed without gradients, and costs extra.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from sparse_structure import pose_matrix

HISTORY = 8
IMAGE_SIZE = (90, 160)
STRUCTURE_SIZE = (180, 320)
DEPTHS_M = (.5, .75, 1., 1.5, 2., 2.75, 3.75, 5., 7., 10.)
ARMS = ("single_frame", "ordinary_video", "temporal_structure")
META_SIZE = HISTORY * 26
CHANNELS = 17  # RGB3, past-gray7, common spatial2, temporal geometry5.


def cumulative_contact(logits: torch.Tensor) -> torch.Tensor:
    """Discrete conditional hazards produce monotone cumulative probabilities."""
    return 1. - torch.cumprod(torch.sigmoid(-logits), dim=-1)


def frame_poses(frames: list[dict], device: str | torch.device):
    rotations, translations = zip(*(pose_matrix(f["camera_transform"]) for f in frames))
    return (torch.as_tensor(np.stack(rotations), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(translations), dtype=torch.float32, device=device))


def motion_features(frames: list[dict], repeat_history: bool = False) -> np.ndarray:
    """Relative poses only; pose translations in metres and angles in degrees.

    Historical speed and timestamps remain their observed values under the
    history ablation. Only historical images and camera/wearer poses are copied.
    """
    if len(frames) != HISTORY:
        raise ValueError("Require eight chronological observations")
    current = frames[-1]
    rc, tc = pose_matrix(current["camera_transform"])
    values = []
    for observed in frames:
        pose_frame = current if repeat_history else observed
        for key in ("camera_transform", "wearer_transform"):
            rotation, translation = pose_matrix(pose_frame[key])
            values.extend((rc.T @ rotation).reshape(-1).tolist())
            values.extend((rc.T @ (translation - tc)).tolist())
        values.extend((float(observed["speed_m_s"]),
                       float(observed["time_s"] - current["time_s"])))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (META_SIZE,) or not np.isfinite(result).all():
        raise ValueError("Invalid finite relative motion metadata")
    return result


def project_history(rotations: torch.Tensor, translations: torch.Tensor,
                    height: int, width: int, hfov: float,
                    depths: torch.Tensor, rotation_only: bool = False):
    """Reproject current rays into all past cameras, T x D x H x W x 2.

    Camera matrices have columns right/down/forward in UE XYZ, metres.
    Depth is current optical forward depth. A rotation-only projection ignores
    translation and is exactly depth-invariant (testable independent of RGB).
    """
    device, dtype = rotations.device, rotations.dtype
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                            torch.arange(width, device=device, dtype=dtype), indexing="ij")
    focal = width / (2. * math.tan(math.radians(hfov / 2.)))
    rays = torch.stack(((xx - (width-1)/2)/focal,
                        (yy - (height-1)/2)/focal, torch.ones_like(xx)), -1)
    relative_r = rotations[:-1].transpose(-1, -2) @ rotations[-1]
    relative_t = torch.einsum("tij,tj->ti", rotations[:-1].transpose(-1, -2),
                               translations[-1] - translations[:-1])
    if rotation_only:
        relative_t = torch.zeros_like(relative_t)
    rotated = torch.einsum("tij,hwj->thwi", relative_r, rays)
    xyz = rotated[:, None] * depths[None, :, None, None, None]
    xyz = xyz + relative_t[:, None, None, None, :]
    z = xyz[..., 2]
    uv = xyz[..., :2] / z.clamp_min(1e-6)[..., None] * focal
    uv = uv + uv.new_tensor([(width-1)/2, (height-1)/2])
    valid = ((z > 1e-4) & (uv[..., 0] >= 2) & (uv[..., 0] <= width-3)
             & (uv[..., 1] >= 2) & (uv[..., 1] <= height-3))
    return uv, valid


def spatial_features(gray: torch.Tensor):
    gx = F.pad((gray[..., 2:] - gray[..., :-2]) * .5, (1, 1))
    gy = F.pad((gray[..., 2:, :] - gray[..., :-2, :]) * .5, (0, 0, 1, 1))
    magnitude = (gx.square() + gy.square()).sqrt()
    # The minor structure-tensor eigenvalue retains corners/finite endpoints;
    # long straight edges retain their normal through gx/gy and magnitude.
    smooth = lambda x: F.avg_pool2d(x, 5, stride=1, padding=2)
    a, b, c = smooth(gx.square()), smooth(gy.square()), smooth(gx * gy)
    minor = .5 * (a + b - ((a-b).square() + 4*c.square()).clamp_min(0).sqrt())
    spatial = torch.cat(((magnitude / .10).clamp(0, 1),
                         (minor.clamp_min(0).sqrt() / .04).clamp(0, 1)), 1)
    normal = torch.cat((gx, gy), 1) / magnitude.clamp_min(1e-5)
    return spatial, normal


def local_ncc(reference: torch.Tensor, candidate: torch.Tensor):
    pool = lambda x: F.avg_pool2d(x, 5, stride=1, padding=2)
    ra, ca = pool(reference), pool(candidate)
    rv = (pool(reference.square()) - ra.square()).clamp_min(0)
    cv = (pool(candidate.square()) - ca.square()).clamp_min(0)
    return ((pool(reference * candidate) - ra * ca)
            / (rv * cv).sqrt().clamp_min(1e-5)).clamp(-1, 1)


@torch.no_grad()
def structure_features(rgb: torch.Tensor, rotations: torch.Tensor,
                       translations: torch.Tensor, hfov: float = 100.):
    """Fixed dense plane sweep: continuous edge normals, no target crop or truth.

    rgb is T x 3 x 180 x 320, [0,1]. Five geometry maps are selected inverse
    depth, NCC consistency, improvement over rotation-only, separated-depth
    score margin, and edge-normal parallax. Without translation they are zero.
    Returns common spatial2 plus temporal geometry5 at 90 x 160.
    """
    gray = (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    spatial, normals = spatial_features(gray[-1:])
    h, w = gray.shape[-2:]
    geometry = torch.zeros((1, 5, h, w), device=rgb.device)
    if (translations[:-1] - translations[-1]).abs().amax() <= 1e-6:
        return F.max_pool2d(torch.cat((spatial, geometry), 1), 2).squeeze(0)
    depths = rgb.new_tensor(DEPTHS_M)
    uv, valid = project_history(rotations, translations, h, w, hfov, depths)
    rot_uv, rot_valid = project_history(rotations, translations, h, w, hfov,
                                       depths[:1], rotation_only=True)
    all_uv = torch.cat((uv, rot_uv), 1)
    t, d = all_uv.shape[:2]
    grid = all_uv / all_uv.new_tensor([w-1, h-1]) * 2 - 1
    source = gray[:-1, None].expand(t, d, 1, h, w).reshape(t*d, 1, h, w)
    warped = F.grid_sample(source, grid.reshape(t*d, h, w, 2),
                           mode="bilinear", padding_mode="zeros", align_corners=True)
    scores = local_ncc(gray[-1:], warped).reshape(t, d, h, w)
    valid_all = torch.cat((valid, rot_valid), 1)
    # Require multiple actual views; invalid reprojections are never evidence.
    counts = valid_all.sum(0)
    averaged = (scores * valid_all).sum(0) / counts.clamp_min(1)
    averaged = averaged.masked_fill(counts < 2, -1.)
    depth_scores, rotation_score = averaged[:-1], averaged[-1]
    best, index = depth_scores.max(0)
    selected_depth = depths[index]
    separated = (depths[:, None, None] - selected_depth).abs() >= selected_depth * .25
    alternative = depth_scores.masked_fill(~separated, -1.).amax(0)
    delta = uv - rot_uv
    normal = normals[0].permute(1, 2, 0)
    normal_parallax = (delta * normal[None, None]).sum(-1).abs()
    mean_parallax = (normal_parallax * valid).sum(0) / valid.sum(0).clamp_min(1)
    selected_parallax = mean_parallax.gather(0, index[None])[0]
    support = spatial[0, 0] * (selected_parallax >= .05) * (best > -.5)
    geometry = torch.stack((.5 / selected_depth, (best+1.)*.5,
                            (best - rotation_score).clamp(-1, 1),
                            (best - alternative).clamp(0, 1),
                            selected_parallax / (selected_parallax + 1.))) * support
    return F.max_pool2d(torch.cat((spatial, geometry[None]), 1), 2).squeeze(0)


@torch.no_grad()
def prepare_features(rgb: torch.Tensor, frames: list[dict], hfov: float = 100.):
    if tuple(rgb.shape) != (HISTORY, 3, *STRUCTURE_SIZE):
        raise ValueError(f"Expected eight RGB frames at {STRUCTURE_SIZE}")
    rotations, translations = frame_poses(frames, rgb.device)
    high_structure = structure_features(rgb, rotations, translations, hfov)
    low = F.interpolate(rgb, IMAGE_SIZE, mode="area")
    gray = (low * low.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
    # Temporal channels are deviations from current; repeated history is zero.
    visual = torch.cat((low[-1], gray[:-1] - gray[-1:], high_structure), 0)
    metadata = torch.as_tensor(motion_features(frames), device=rgb.device)
    repeated_metadata = torch.as_tensor(motion_features(frames, True), device=rgb.device)
    return visual, metadata, repeated_metadata


def arm_features(visual: torch.Tensor, arm: str, repeated_history: bool = False):
    if arm not in ARMS:
        raise ValueError(arm)
    value = visual.clone()
    if arm == "single_frame" or repeated_history:
        value[:, 3:10] = 0.
    if arm != "temporal_structure" or repeated_history:
        value[:, 12:] = 0.
    return value


class ContactRetina(nn.Module):
    """Identical trainable architecture in all arms, preserving coarse location."""
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(CHANNELS, 24, 5, stride=2, padding=2), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GroupNorm(4, 48), nn.SiLU(),
            nn.AvgPool2d((4, 4)), nn.Flatten())
        self.motion = nn.Sequential(nn.Linear(META_SIZE, 48), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(48*3*5 + 48, 96), nn.SiLU(), nn.Linear(96, 6))

    def forward(self, visual: torch.Tensor, metadata: torch.Tensor):
        logits = self.head(torch.cat((self.visual(visual), self.motion(metadata)), -1))
        return cumulative_contact(logits.reshape(-1, 2, 3))
