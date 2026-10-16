#!/usr/bin/env python3
"""Controlled DINO/RGB models for full-image functional-region experiments.

This is a new model family, separate from ``full_train_baselines.DinoDense``.
Every arm receives the SAME real RGB stride-4/8 features and DINO-S/16 features.
The ordinary high-resolution control therefore has actual image detail; merely
upsampling DINO tokens is not its source of high-resolution evidence.

The local-fusion ablation has the same learned tensors and convolution MACs:
ordinary fusion adds a learned local residual; gated fusion multiplies the local
features by ``2 * sigmoid(residual)``.  The residual's final layer starts at zero,
so both paths start from the same feature function.  The gate is conditioned on
predicted class distribution/entropy and image features, NOT a GT/query mask.
It is an experimental mechanism, not an established novelty claim.

Six arms (a missing head returns None and is not executed):
  plain              ordinary RGB-FPN, segmentation supervision
  plain_quality      ordinary RGB-FPN + point-quality supervision/readout
  detail             conditional local gate, segmentation supervision
  detail_quality     conditional local gate + point-quality supervision/readout
  candidate_quality  ordinary RGB-FPN + dense new offsets + point quality
  joint              conditional local gate + dense new offsets + point quality

Common tensors are initialized under independent named CPU RNG seeds, regardless
of arm construction order or optional heads.  The pretrained encoder is loaded
only from a local SelectiveCL checkpoint; no pretrained network is downloaded.
Only its final four blocks and final norm are trainable by default, matching the
previous DINO experiment.  Heads are initialized anew, not from its decoder.

Losses, GT-derived training targets, point selection and checkpoint selection are
owned by the training/evaluation harness.  In particular the ordinary quality
and candidate-quality controls must receive the same targets and loss weights
as their gated counterparts.  Match actual training/update cost and report the
extra quality/offset cost; do not spend dummy operations to make costs equal.

Forward tensors for input Bx3xHxW (H/W divisible by 16):
  seg_logits:         B x C x H/4 x W/4
  quality_logits:     B x (C-1) x H/4 x W/4, or None
  offsets:            B x (C-1) x 2 x H/4 x W/4, or None
  coarse_seg_logits:  B x C x H/16 x W/16

Offset axis 0 is x, axis 1 is y.  Values have ALREADY passed tanh and are in
input-image width/height units; do not tanh or multiply by stride a second time.
``offset_candidate_coordinates`` creates one new coordinate per dense cell and
class.  Its cell centers are not the old fixed three-point candidate pool.
The selector can sample segmentation/quality at these destinations using the
same align_corners=False geometry.  All selection inputs must be model outputs.

Run ``python full_joint_models.py selftest`` on the registered research worker
for a synthetic CPU-only structural check.  That check uses an injected tiny
encoder, never timm/pretrained weights or CUDA.  No training is launched here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ARM_FLAGS = {
    'plain': (False, False, False),
    'plain_quality': (False, True, False),
    'detail': (True, False, False),
    'detail_quality': (True, True, False),
    'candidate_quality': (False, True, True),
    'joint': (True, True, True),
}
MODEL_FAMILY = 'dino_rgb_point_joint_v1'


@dataclass(frozen=True)
class JointConfig:
    arm: str = 'plain'
    num_classes: int = 10
    width: int = 64
    rgb_width: int = 32
    point_width: int = 32
    offset_radius: float = .125
    patch_size: int = 16
    train_last_blocks: int = 4
    init_seed: int = 20261016

    def __post_init__(self):
        if self.arm not in ARM_FLAGS:
            raise ValueError(f'Unknown model arm: {self.arm}')
        if self.num_classes < 2:
            raise ValueError('At least background and one foreground class required')
        if min(self.width, self.rgb_width, self.point_width) < 8:
            raise ValueError('Feature widths must be at least eight')
        if self.patch_size != 16:
            raise ValueError('This family fixes the inherited DINO patch size to 16')
        if self.train_last_blocks < 0:
            raise ValueError('train_last_blocks must be nonnegative')
        if not math.isfinite(self.offset_radius) or not 0 < self.offset_radius <= 1:
            raise ValueError('offset_radius must be in (0, 1] input-image units')

    @property
    def use_local_gate(self):
        return ARM_FLAGS[self.arm][0]

    @property
    def use_quality(self):
        return ARM_FLAGS[self.arm][1]

    @property
    def use_offsets(self):
        return ARM_FLAGS[self.arm][2]


def _groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


def _named_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f'{seed}:{MODEL_FAMILY}:{name}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2 ** 31)


def _initialize(builder: Callable[[], nn.Module], seed: int, name: str) -> nn.Module:
    # Construction is CPU-only.  Preserve caller CPU RNG and do not initialize
    # or touch a CUDA RNG/device in this model-building helper.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(_named_seed(seed, name))
        return builder()


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=groups, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


class SeparableBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            ConvNormAct(in_channels, in_channels, 3, stride, groups=in_channels),
            ConvNormAct(in_channels, out_channels, 1),
        )


class RGBPyramid(nn.Module):
    """Shallow learned RGB evidence, shared unchanged by all arms."""

    def __init__(self, config: JointConfig):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, config.rgb_width, 3, 2),
            ConvNormAct(config.rgb_width, config.width, 3, 2),
        )
        self.level4 = SeparableBlock(config.width, config.width)
        self.level8 = SeparableBlock(config.width, config.width, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        level4 = self.level4(self.stem(x))
        return level4, self.level8(level4)


class ContextPyramid(nn.Module):
    def __init__(self, config: JointConfig):
        super().__init__()
        self.project = ConvNormAct(384, config.width, 1)
        self.level8 = SeparableBlock(config.width, config.width)
        self.level4 = SeparableBlock(config.width, config.width)
        self.classifier = nn.Conv2d(config.width, config.num_classes, 1)

    def forward(self, tokens: Tensor, size4: tuple[int, int]) -> tuple[Tensor, Tensor]:
        coarse = self.project(tokens)
        logits = self.classifier(coarse)
        size8 = (size4[0] // 2, size4[1] // 2)
        context8 = self.level8(F.interpolate(coarse, size=size8, mode='bilinear', align_corners=False))
        context4 = self.level4(F.interpolate(context8, size=size4, mode='bilinear', align_corners=False))
        return context4, logits


class EvidenceFusion(nn.Module):
    """Same tensors/convolutions for additive and conditional gated fusion."""

    def __init__(self, config: JointConfig):
        super().__init__()
        width = config.width
        self.gated = config.use_local_gate
        self.guide = nn.Conv2d(width + config.num_classes + 1, width * 2, 1)
        self.residual = nn.Sequential(
            SeparableBlock(width * 2, width * 2),
            nn.Conv2d(width * 2, width * 2, 1),
        )
        # Identity start in both arms, with identical ordinary/gated tensors.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.fuse = nn.Sequential(
            ConvNormAct(width * 3, width, 1),
            SeparableBlock(width, width),
        )

    def forward(self, context: Tensor, rgb4: Tensor, rgb8: Tensor,
                coarse_logits: Tensor) -> Tensor:
        size = context.shape[-2:]
        up_rgb8 = F.interpolate(rgb8, size=size, mode='bilinear', align_corners=False)
        local = torch.cat((rgb4, up_rgb8), dim=1)
        coarse_up = F.interpolate(coarse_logits.float(), size=size,
                                  mode='bilinear', align_corners=False)
        log_probability = coarse_up.log_softmax(dim=1)
        probability = log_probability.exp()
        entropy = -(probability * log_probability).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(coarse_logits.shape[1])
        guide_input = torch.cat((context, probability.to(context.dtype),
                                 entropy.to(context.dtype)), dim=1)
        residual = self.residual(local + self.guide(guide_input))
        if self.gated:
            local = local * (2 * residual.sigmoid())
        else:
            local = local + residual
        return self.fuse(torch.cat((context, local), dim=1))


class PointHead(nn.Sequential):
    def __init__(self, config: JointConfig, outputs: int):
        super().__init__(
            SeparableBlock(config.width, config.point_width),
            nn.Conv2d(config.point_width, outputs, 1),
        )


def _local_dino_encoder(weights: str | Path | None, config: JointConfig) -> nn.Module:
    import timm

    encoder = _initialize(lambda: timm.create_model(
        'vit_small_patch16_224', pretrained=False, num_classes=0,
        dynamic_img_size=True), config.init_seed, 'encoder')
    if weights is not None:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f'Local SelectiveCL checkpoint required: {path}')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.removeprefix('module.'): value for k, value in state.items()}
        state = {k.removeprefix('vit_model.'): value for k, value in state.items()
                 if k.startswith('vit_model.')}
        if not state:
            raise ValueError('No SelectiveCL vit_model state in local checkpoint')
        encoder.load_state_dict(state, strict=True)
    return encoder


class DinoJoint(nn.Module):
    model_family = MODEL_FAMILY

    def __init__(self, config: JointConfig | Mapping | None = None,
                 weights: str | Path | None = None, encoder: nn.Module | None = None):
        super().__init__()
        if config is None:
            config = JointConfig()
        elif not isinstance(config, JointConfig):
            config = JointConfig(**dict(config))
        self.config = config
        if encoder is not None and weights is not None:
            raise ValueError('Pass either a local checkpoint or an injected encoder')
        self.encoder = encoder if encoder is not None else _local_dino_encoder(weights, config)
        if not hasattr(self.encoder, 'blocks') or not hasattr(self.encoder, 'norm'):
            raise TypeError('Expected timm-compatible encoder.blocks and encoder.norm')
        if config.train_last_blocks > len(self.encoder.blocks):
            raise ValueError('train_last_blocks exceeds encoder depth')
        self.encoder.requires_grad_(False)
        if config.train_last_blocks:
            for block in self.encoder.blocks[-config.train_last_blocks:]:
                block.requires_grad_(True)
        self.encoder.norm.requires_grad_(True)

        self.rgb = _initialize(lambda: RGBPyramid(config), config.init_seed, 'rgb')
        self.context = _initialize(lambda: ContextPyramid(config), config.init_seed, 'context')
        self.fusion = _initialize(lambda: EvidenceFusion(config), config.init_seed, 'fusion')
        self.seg_head = _initialize(lambda: nn.Conv2d(config.width, config.num_classes, 1),
                                    config.init_seed, 'seg_head')
        self.quality_head = (_initialize(lambda: PointHead(config, config.num_classes - 1),
                                         config.init_seed, 'quality_head')
                             if config.use_quality else None)
        self.offset_head = (_initialize(lambda: PointHead(config, (config.num_classes - 1) * 2),
                                        config.init_seed, 'offset_head')
                            if config.use_offsets else None)
        if self.offset_head is not None:
            # Every cell starts at its own coordinate, without reading a GT point.
            nn.init.zeros_(self.offset_head[-1].weight)
            nn.init.zeros_(self.offset_head[-1].bias)

    def forward(self, x: Tensor) -> dict[str, Tensor | None]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError('Expected input Bx3xHxW')
        height, width = x.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError('Input height/width must be divisible by inherited patch size 16')
        features = self.encoder.forward_features(x)
        prefix = getattr(self.encoder, 'num_prefix_tokens', 1)
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise TypeError('Expected timm forward_features tensor Bx(tokens+prefix)x384')
        h16, w16 = height // 16, width // 16
        if features.shape[1:] != (h16 * w16 + prefix, 384):
            raise ValueError(f'Unexpected DINO feature shape {tuple(features.shape)}')
        tokens = features[:, prefix:].transpose(1, 2).reshape(x.shape[0], 384, h16, w16)
        rgb4, rgb8 = self.rgb(x)
        context, coarse_logits = self.context(tokens, rgb4.shape[-2:])
        fused = self.fusion(context, rgb4, rgb8, coarse_logits)
        quality = self.quality_head(fused) if self.quality_head is not None else None
        offsets = None
        if self.offset_head is not None:
            raw = self.offset_head(fused)
            offsets = raw.reshape(x.shape[0], self.config.num_classes - 1, 2,
                                  raw.shape[-2], raw.shape[-1]).tanh() * self.config.offset_radius
        return {'seg_logits': self.seg_head(fused), 'quality_logits': quality,
                'offsets': offsets, 'coarse_seg_logits': coarse_logits}

    def config_dict(self) -> dict:
        return dict(asdict(self.config), model_family=self.model_family,
                    encoder_identity='DINO ViT-S/16 from SelectiveCL, NOT DINOv2',
                    real_rgb_detail=True, output_stride=4,
                    offsets_axis_order='xy', offsets_units='normalized_input_width_height')

    def parameter_report(self) -> dict:
        components = {}
        for name, module in self.named_children():
            components[name] = {
                'parameters': sum(p.numel() for p in module.parameters()),
                'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad),
            }
        return {
            'model_family': self.model_family, 'arm': self.config.arm,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'components': components,
            'frozen_encoder_blocks': len(self.encoder.blocks) - self.config.train_last_blocks,
            'trainable_last_encoder_blocks': self.config.train_last_blocks,
            'quality_head_executed': self.config.use_quality,
            'offset_head_executed': self.config.use_offsets,
            'forward_compute_note': 'Measure actual device forward/backward; no dummy heads execute.',
        }


def offset_candidate_coordinates(offsets: Tensor) -> Tensor:
    """GT-free dense destinations, B x foreground_classes x h x w x (x,y).

    Centers ``(column+.5)/w, (row+.5)/h`` match align_corners=False sampling.
    Returned normalized coordinates are clamped to [0,1]; floor-to-native-pixel
    evaluation must clamp the integer result to width-1/height-1 at the border.
    No candidate selection, confidence or GT access is performed by this helper.
    """
    if offsets.ndim != 5 or offsets.shape[2] != 2:
        raise ValueError('Expected Bxforeground_classesx2xhxw normalized xy offsets')
    height, width = offsets.shape[-2:]
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=offsets.device, dtype=offsets.dtype) + .5) / height,
        (torch.arange(width, device=offsets.device, dtype=offsets.dtype) + .5) / width,
        indexing='ij')
    base = torch.stack((xx, yy), dim=-1)[None, None]
    return (base + offsets.permute(0, 1, 3, 4, 2)).clamp(0, 1)


def selftest() -> dict:
    """Synthetic CPU check only; this does not validate research performance."""
    torch.set_num_threads(1)
    caller_rng = torch.get_rng_state().clone()
    cuda_was_initialized = torch.cuda.is_initialized()

    class TinyEncoder(nn.Module):
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            self.patch = nn.Conv2d(3, 384, 16, stride=16)
            self.blocks = nn.ModuleList(nn.Linear(384, 384) for _ in range(6))
            self.norm = nn.LayerNorm(384)

        def forward_features(self, x):
            z = self.patch(x).flatten(2).transpose(1, 2)
            for block in self.blocks:
                z = z + F.gelu(block(z))
            z = self.norm(z)
            return torch.cat((z.new_zeros(z.shape[0], 1, 384), z), dim=1)

    def new_model(arm):
        encoder = _initialize(TinyEncoder, 17, 'synthetic_encoder')
        return DinoJoint(JointConfig(arm=arm, width=16, rgb_width=8, point_width=8), encoder=encoder)

    x = torch.linspace(-1, 1, 3 * 64 * 80).reshape(1, 3, 64, 80)
    baseline = new_model('candidate_quality')
    gated = new_model('joint')
    for key, value in baseline.state_dict().items():
        if not torch.equal(value, gated.state_dict()[key]):
            raise AssertionError(f'Mismatched common initialization: {key}')
    baseline.eval(); gated.eval()
    left, right = baseline(x), gated(x)
    for key in ('seg_logits', 'quality_logits', 'offsets', 'coarse_seg_logits'):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert left['seg_logits'].shape == (1, 10, 16, 20)
    assert left['quality_logits'].shape == (1, 9, 16, 20)
    assert left['offsets'].shape == (1, 9, 2, 16, 20)
    assert torch.count_nonzero(left['offsets']).item() == 0
    candidates = offset_candidate_coordinates(left['offsets'])
    torch.testing.assert_close(candidates[0, 0, 0, 0], x.new_tensor([.5 / 20, .5 / 16]))
    assert candidates.shape == (1, 9, 16, 20, 2)
    assert all(not p.requires_grad for block in gated.encoder.blocks[:-4] for p in block.parameters())
    assert all(p.requires_grad for block in gated.encoder.blocks[-4:] for p in block.parameters())
    loss = (right['seg_logits'].square().mean() + right['quality_logits'].square().mean()
            + right['offsets'].mean())
    loss.backward()
    assert gated.rgb.stem[0][0].weight.grad is not None
    assert torch.isfinite(gated.rgb.stem[0][0].weight.grad).all()
    assert gated.offset_head[-1].weight.grad is not None
    assert torch.count_nonzero(gated.offset_head[-1].bias.grad).item() == 18
    baseline_keys = baseline.state_dict()
    reports = {}
    for arm in ARM_FLAGS:
        model = new_model(arm)
        for key, value in model.state_dict().items():
            if not torch.equal(value, baseline_keys[key]):
                raise AssertionError(f'Optional head shifted common initialization: {arm}/{key}')
        result = model(x)
        assert (result['quality_logits'] is not None) == model.config.use_quality
        assert (result['offsets'] is not None) == model.config.use_offsets
        reports[arm] = model.parameter_report()
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert torch.cuda.is_initialized() == cuda_was_initialized
    return {'status': 'PASS', 'test': 'synthetic_cpu_no_weights_no_cuda',
            'common_initialization_exact': True, 'additive_gated_initial_function_exact': True,
            'caller_cpu_rng_preserved': True, 'cuda_initialization_unchanged': True,
            'real_rgb_gradient_checked': True, 'offset_coordinate_geometry_checked': True,
            'note': 'Tiny injected encoder; actual timm/GPU profile remains required.',
            'arms': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['selftest'])
    args = parser.parse_args()
    if args.command == 'selftest':
        print(json.dumps(selftest(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
