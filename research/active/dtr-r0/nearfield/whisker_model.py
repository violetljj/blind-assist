"""NF-G8 RGB-only near-field learner. No metric pose, speed, flow or depth inputs."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

ARMS = ("single_frame", "ordinary_video", "bio_video")
HISTORY = 3
IMAGE_SIZE = (144, 256)
SUPPORT_SIZE = (18, 32)
TARGET_ORDER = ("BODYnear", "HEADnear", "BODYapproaching", "HEADapproaching")


def repeat_history(rgb):
    """Replace only image history, without accepting or altering metadata."""
    return rgb[:, -1:].expand_as(rgb)


def reichardt(rgb):
    """Two adjacent-time pairs x four signed local directional correlations.

    A one-frame delayed luminance times a neighboring current luminance is
    opposed to the reverse product. Replicate padding avoids wraparound edges.
    This is an image correlation detector, not a metric optical-flow estimator.
    """
    gray = (rgb * rgb.new_tensor([.299, .587, .114])[None, None, :, None, None]).sum(2)
    before, now = gray[:, :-1], gray[:, 1:]
    def shift(value, dy, dx):
        padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
        h, w = value.shape[-2:]
        return padded[..., 1+dy:1+dy+h, 1+dx:1+dx+w]
    return torch.cat([before * shift(now, dy, dx) - now * shift(before, dy, dx)
                      for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))], 1)


class WhiskerModel(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 24, 3, 2, 1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU())
        self.support = nn.Conv2d(32, 2, 1)
        self.near = nn.Sequential(nn.AvgPool2d((6, 8)), nn.Flatten(), nn.Linear(384, 2))
        if arm == "bio_video":
            self.excitation = nn.Conv2d(8, 16, 5, 2, 2)
            self.inhibition = nn.Conv2d(8, 16, 5, 2, 2)
        else:
            self.video = nn.Conv3d(3, 16, (3, 5, 5), (1, 2, 2), (0, 2, 2))
        self.temporal = nn.Sequential(nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU())
        self.approach = nn.Sequential(nn.AvgPool2d((6, 8)), nn.Flatten(),
                                      nn.Linear(768, 32), nn.SiLU(), nn.Linear(32, 2))

    def forward(self, rgb):
        if rgb.ndim != 5 or rgb.shape[1:3] != (HISTORY, 3) or tuple(rgb.shape[-2:]) != IMAGE_SIZE:
            raise ValueError("Expected B x 3 history x 3 RGB x 144 x 256")
        spatial = self.spatial(rgb[:, -1])
        near, support = self.near(spatial), self.support(spatial)
        if self.arm == "single_frame":
            rgb = repeat_history(rgb)
        b, t, c, h, w = rgb.shape
        small = F.avg_pool2d(rgb.reshape(b*t, c, h, w), 2).reshape(b, t, c, h//2, w//2)
        if self.arm == "bio_video":
            correlation = reichardt(small)
            motion = F.softplus(self.excitation(correlation)) - F.softplus(self.inhibition(correlation))
        else:
            motion = self.video(small.transpose(1, 2)).squeeze(2)
        motion = self.temporal(motion)
        approach = self.approach(torch.cat((spatial, motion), 1))
        return torch.cat((near, approach), 1), support


def masked_bce(logits, targets):
    known = targets >= 0
    loss = F.binary_cross_entropy_with_logits(logits, targets.clamp_min(0), reduction="none")
    return (loss * known).sum() / known.sum().clamp_min(1)


def support_bce(logits, targets, near_targets):
    """Mean each present class, then average classes over known pixels only.

    A thin positive trace receives the same total weight as known background.
    Empty classes are omitted; an all-UNKNOWN batch has zero loss/gradient.
    """
    known = (near_targets >= 0)[..., None, None].expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive, negative = known & (targets > .5), known & (targets <= .5)
    positive_count, negative_count = positive.sum(), negative.sum()
    positive_mean = (loss * positive).sum() / positive_count.clamp_min(1)
    negative_mean = (loss * negative).sum() / negative_count.clamp_min(1)
    classes = (positive_count > 0).to(loss.dtype) + (negative_count > 0).to(loss.dtype)
    return (positive_mean + negative_mean) / classes.clamp_min(1)
