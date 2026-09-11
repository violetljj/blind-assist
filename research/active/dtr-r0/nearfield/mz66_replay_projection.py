"""One closed-form projection of an actual Adam parameter displacement.

For d = theta_after - theta_before, project onto {v: replay_gradient @ v <= 0}.
The Euclidean solution in real arithmetic is d - max(g @ d, 0) * g / (g @ g).
This module computes the proposal in FP64 on the input device, then assigns FP32
once. It neither changes Adam moments nor recomputes/backtracks the update.

A-GEM section 4 motivates the single-halfspace construction:
https://arxiv.org/pdf/1812.00420 . This is an actual-Adam-displacement projection,
not a reproduction of A-GEM's gradient update or a nonlinear loss guarantee.
"""

from __future__ import annotations

import math
from typing import Any

import torch


@torch.no_grad()
def project_adam_delta(
    theta_before: torch.Tensor,
    theta_after: torch.Tensor,
    replay_gradient: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a new FP32 parameter vector and measured numerical diagnostics.

    Inputs must be nonempty, finite, flattened FP32 tensors on the same device.
    ``replay_gradient`` is evaluated at ``theta_before`` by the caller. Inputs
    are not mutated. There is exactly one constraint and no numerical tolerance
    in the branch decision. Zero gradient and nonconflict preserve after's bits.

    ``theta_ideal64`` is the actual uncast FP64 proposal, not an exact-real
    arithmetic promise. It stays on the input device; remove this tensor before
    JSON serialization. All other diagnostics are JSON scalar values.

    Let p be that FP64 proposal and x the assigned FP32 vector, both interpreted
    as real vectors. The identity g@(x-b) = g@(p-b) + g@(x-p) gives the diagnostic
    bound max(ideal_dot, 0) + rounding_bound + arithmetic_roundoff_bound. The
    rounding term is sum(abs(g) * abs(x64-p)), not a tunable tolerance.

    The arithmetic term conservatively envelopes the subtractions, FP64 dot and
    positive-sum reductions in that identity: gamma=(4*n+32)*u/(1-(4*n+32)*u),
    u=2**-53; gamma/(1-gamma) times the measured absolute-product scales. This
    assumes ordinary IEEE round-to-nearest FP64 without overflow/underflow. The
    measured ideal_dot separately exposes roundoff in computing the projection.
    No diagnostic changes the returned parameters or certifies nonlinear loss,
    other replay batches, held outcomes, or exact FP32 halfspace feasibility.
    """
    named = (
        ("theta_before", theta_before),
        ("theta_after", theta_after),
        ("replay_gradient", replay_gradient),
    )
    for name, value in named:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype != torch.float32 or value.layout != torch.strided:
            raise TypeError(f"{name} must be a strided FP32 tensor")
        if value.ndim != 1 or value.numel() == 0:
            raise ValueError(f"{name} must be a nonempty flattened vector")
        if value.shape != theta_before.shape:
            raise ValueError("all three input shapes must match")
        if value.device != theta_before.device:
            raise ValueError("all three input devices must match")
    if not torch.stack([torch.isfinite(t).all() for _, t in named]).all().item():
        raise ValueError("all three inputs must be finite")

    n = theta_before.numel()
    ku = (4 * n + 32) * (2.0**-53)
    if ku >= 0.5:
        raise ValueError("vector too large for the stated FP64 roundoff bound")
    gamma = ku / (1.0 - ku)
    before64 = theta_before.detach().to(dtype=torch.float64)
    after64 = theta_after.detach().to(dtype=torch.float64)
    g64 = replay_gradient.detach().to(dtype=torch.float64)
    delta64 = after64 - before64
    pre = torch.dot(g64, delta64)
    norm2 = torch.dot(g64, g64)
    pre_value, norm2_value = torch.stack((pre, norm2)).tolist()
    zero_gradient = norm2_value == 0.0
    projected = pre_value > 0.0 and not zero_gradient
    coefficient = pre / norm2 if projected else pre.new_zeros(())
    if projected:
        theta_ideal64 = after64 - coefficient * g64
        assigned = theta_ideal64.to(dtype=torch.float32)
        if not torch.isfinite(assigned).all().item():
            raise FloatingPointError("projected parameters are not finite FP32")
    else:
        theta_ideal64 = after64
        # Reconstructing b + d can change signed-zero bits on an identity path.
        assigned = theta_after.detach().clone()

    assigned64 = assigned.to(dtype=torch.float64)
    ideal_delta = theta_ideal64 - before64
    assigned_delta = assigned64 - before64
    rounding_delta = assigned64 - theta_ideal64
    ideal_dot = torch.dot(g64, ideal_delta)
    post_dot = torch.dot(g64, assigned_delta)
    rounding_dot = torch.dot(g64, rounding_delta)
    abs_g = g64.abs()
    rounding_bound = torch.sum(abs_g * rounding_delta.abs())
    ideal_scale = torch.sum(abs_g * ideal_delta.abs())
    post_scale = torch.sum(abs_g * assigned_delta.abs())
    arithmetic_bound = (gamma / (1.0 - gamma)) * (
        ideal_scale + post_scale + rounding_bound
    )
    # Inflation by one FP64 ULP covers the final scalar-bound rounding.
    arithmetic_bound = torch.nextafter(
        arithmetic_bound, torch.full_like(arithmetic_bound, math.inf)
    )
    residual_bound = ideal_dot.clamp_min(0) + rounding_bound + arithmetic_bound
    residual_bound = torch.nextafter(
        residual_bound, torch.full_like(residual_bound, math.inf)
    )
    original_delta_norm = torch.linalg.vector_norm(delta64)
    gradient_norm = torch.sqrt(norm2)
    denominator = gradient_norm * original_delta_norm
    positive_residual = post_dot.clamp_min(0)
    relative_residual = positive_residual / torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    scalar_tensors = {
        "pre_dot": pre,
        "gradient_norm_squared": norm2,
        "projection_coefficient": coefficient,
        "ideal_dot": ideal_dot,
        "post_dot": post_dot,
        "rounding_dot": rounding_dot,
        "rounding_bound": rounding_bound,
        "arithmetic_roundoff_bound": arithmetic_bound,
        "residual_bound": residual_bound,
        "post_positive_residual": positive_residual,
        "residual_excess_over_bound": positive_residual - residual_bound,
        "relative_residual": relative_residual,
        "relative_residual_denominator": denominator,
        "original_delta_norm": original_delta_norm,
        "assigned_delta_norm": torch.linalg.vector_norm(assigned_delta),
        "correction_norm": torch.linalg.vector_norm(theta_ideal64 - after64),
        "rounding_norm": torch.linalg.vector_norm(rounding_delta),
        "identity_roundoff": post_dot - ideal_dot - rounding_dot,
    }
    values = torch.stack(tuple(scalar_tensors.values())).tolist()
    diagnostics: dict[str, Any] = dict(zip(scalar_tensors, values))
    diagnostics.update(
        theta_ideal64=theta_ideal64,
        projected=projected,
        zero_gradient=zero_gradient,
        dimensions=n,
        device=str(theta_before.device),
        parameter_dtype="float32",
        arithmetic_dtype="float64",
        roundoff_gamma=gamma,
        roundoff_unit=2.0**-53,
        post_residual_within_bound=(
            diagnostics["post_positive_residual"] <= diagnostics["residual_bound"]
        ),
    )
    return assigned, diagnostics
