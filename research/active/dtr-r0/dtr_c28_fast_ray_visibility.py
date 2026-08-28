"""Measured CPU/GPU fast paths for C28 query-voxel ray visibility.

The exact DDA implementation in :mod:`dtr_c28_ray_visibility` is the semantic
reference.  This module keeps the same continuous geometry but avoids walking
every voxel of every raw ray.  CPU first narrows candidate rays with the exact
circumscribed cone of each query voxel and then performs a slab ray-box test.
CUDA evaluates the same slab test in query batches.  Backend selection and its
receipt remain the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from dtr_c28_ray_visibility import VOXEL_SIZE_M, Visibility


STATE_CODE = {
    Visibility.UNSENSED: 0,
    Visibility.OCCLUDED: 1,
    Visibility.KNOWN_FREE: 2,
    Visibility.HIT: 3,
}
CODE_STATE = tuple(
    state for state, _code in sorted(STATE_CODE.items(), key=lambda item: item[1])
)
HALF_DIAGONAL_M = math.sqrt(3.0) * VOXEL_SIZE_M / 2.0


def _points(values: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    if output.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if output.ndim != 2 or output.shape[1] != 3 or not np.isfinite(output).all():
        raise ValueError(f"{name} must have shape (N,3) with finite values")
    return output


def _keys(values: Sequence[Sequence[int]]) -> np.ndarray:
    output = np.asarray(values)
    if output.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if output.ndim != 2 or output.shape[1] != 3 or not np.issubdtype(output.dtype, np.integer):
        raise ValueError("query voxel keys must have integer shape (Q,3)")
    return output.astype(np.int64, copy=False)


def _slab_intervals(
    origin: np.ndarray,
    directions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive-length ray-box intervals for normalized directions."""

    count = len(directions)
    enter = np.full(count, -np.inf, dtype=np.float64)
    leave = np.full(count, np.inf, dtype=np.float64)
    valid = np.ones(count, dtype=bool)
    for axis in range(3):
        values = directions[:, axis]
        parallel = values == 0.0
        valid &= ~(parallel & ((origin[axis] < lower[axis]) | (origin[axis] > upper[axis])))
        moving = ~parallel
        near = np.full(count, -np.inf, dtype=np.float64)
        far = np.full(count, np.inf, dtype=np.float64)
        near[moving] = (lower[axis] - origin[axis]) / values[moving]
        far[moving] = (upper[axis] - origin[axis]) / values[moving]
        axis_enter = np.minimum(near, far)
        axis_leave = np.maximum(near, far)
        enter = np.maximum(enter, axis_enter)
        leave = np.minimum(leave, axis_leave)
    valid &= leave > np.maximum(enter, 0.0)
    return enter, leave, valid


@dataclass(frozen=True)
class CpuRayIndex:
    origin: np.ndarray
    endpoints: np.ndarray
    endpoint_keys: frozenset[tuple[int, int, int]]
    directions: np.ndarray
    ranges: np.ndarray
    tree: cKDTree | None

    @classmethod
    def build(
        cls,
        sensor_origin_world: Sequence[float],
        raw_endpoints_world: Sequence[Sequence[float]],
    ) -> "CpuRayIndex":
        origin = np.asarray(sensor_origin_world, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("sensor origin must have three finite coordinates")
        endpoints = _points(raw_endpoints_world, name="raw endpoints")
        endpoint_key_array = np.floor(endpoints / VOXEL_SIZE_M).astype(np.int64)
        endpoint_keys = frozenset(map(tuple, endpoint_key_array.tolist()))
        delta = endpoints - origin
        ranges = np.linalg.norm(delta, axis=1)
        nonzero = ranges > 0.0
        directions = delta[nonzero] / ranges[nonzero, None]
        return cls(
            origin=origin,
            endpoints=endpoints,
            endpoint_keys=endpoint_keys,
            directions=directions,
            ranges=ranges[nonzero],
            tree=cKDTree(directions) if len(directions) else None,
        )

    def classify_codes(self, query_voxel_keys: Sequence[Sequence[int]]) -> np.ndarray:
        queries = _keys(query_voxel_keys)
        output = np.zeros(len(queries), dtype=np.int8)
        for index, key_array in enumerate(queries):
            key = tuple(int(value) for value in key_array)
            if key in self.endpoint_keys:
                output[index] = STATE_CODE[Visibility.HIT]
                continue
            if self.tree is None:
                continue
            lower = key_array.astype(np.float64) * VOXEL_SIZE_M
            upper = lower + VOXEL_SIZE_M
            center = (lower + upper) * 0.5
            center_delta = center - self.origin
            center_range = float(np.linalg.norm(center_delta))
            if center_range <= HALF_DIAGONAL_M:
                candidates = list(range(len(self.directions)))
            else:
                angular_radius = math.asin(min(1.0, HALF_DIAGONAL_M / center_range))
                chord_radius = 2.0 * math.sin(angular_radius / 2.0)
                candidates = self.tree.query_ball_point(
                    center_delta / center_range,
                    math.nextafter(chord_radius, math.inf),
                )
            if not candidates:
                continue
            candidate = np.asarray(candidates, dtype=np.int64)
            enter, leave, intersects = _slab_intervals(
                self.origin, self.directions[candidate], lower, upper
            )
            if not np.any(intersects):
                continue
            ray_range = self.ranges[candidate]
            if np.any(intersects & (ray_range >= leave)):
                output[index] = STATE_CODE[Visibility.KNOWN_FREE]
            elif np.any(intersects & (ray_range <= np.maximum(enter, 0.0))):
                output[index] = STATE_CODE[Visibility.OCCLUDED]
        return output

    def classify(self, query_voxel_keys: Sequence[Sequence[int]]) -> tuple[Visibility, ...]:
        return tuple(CODE_STATE[int(value)] for value in self.classify_codes(query_voxel_keys))


def classify_gpu_codes(
    sensor_origin_world: Sequence[float],
    raw_endpoints_world: Sequence[Sequence[float]],
    query_voxel_keys: Sequence[Sequence[int]],
    *,
    query_batch: int = 128,
):
    """Return CUDA int8 state codes using exact batched slab intersections."""

    import torch

    origin = torch.as_tensor(sensor_origin_world, dtype=torch.float64, device="cuda")
    endpoints = torch.as_tensor(raw_endpoints_world, dtype=torch.float64, device="cuda")
    queries = torch.as_tensor(query_voxel_keys, dtype=torch.int64, device="cuda")
    if origin.shape != (3,) or endpoints.ndim != 2 or endpoints.shape[1:] != (3,):
        raise ValueError("invalid CUDA ray geometry")
    if queries.ndim != 2 or queries.shape[1:] != (3,):
        raise ValueError("invalid CUDA query keys")
    output = torch.zeros(len(queries), dtype=torch.int8, device="cuda")
    if not len(queries) or not len(endpoints):
        return output
    delta = endpoints - origin
    ranges = torch.linalg.vector_norm(delta, dim=1)
    nonzero = ranges > 0.0
    directions = delta[nonzero] / ranges[nonzero, None]
    ranges = ranges[nonzero]
    endpoint_keys = torch.floor(endpoints / VOXEL_SIZE_M).to(torch.int64)
    for start in range(0, len(queries), query_batch):
        stop = min(len(queries), start + query_batch)
        batch = queries[start:stop]
        hit = (batch[:, None, :] == endpoint_keys[None, :, :]).all(dim=2).any(dim=1)
        lower = batch.to(torch.float64) * VOXEL_SIZE_M
        upper = lower + VOXEL_SIZE_M
        count = len(batch)
        enter = torch.full((count, len(directions)), -torch.inf, dtype=torch.float64, device="cuda")
        leave = torch.full_like(enter, torch.inf)
        valid = torch.ones_like(enter, dtype=torch.bool)
        for axis in range(3):
            values = directions[:, axis][None, :]
            parallel = values == 0.0
            outside = (origin[axis] < lower[:, axis, None]) | (origin[axis] > upper[:, axis, None])
            valid &= ~(parallel & outside)
            safe = torch.where(parallel, torch.ones_like(values), values)
            near = (lower[:, axis, None] - origin[axis]) / safe
            far = (upper[:, axis, None] - origin[axis]) / safe
            near = torch.where(parallel, torch.full_like(near, -torch.inf), near)
            far = torch.where(parallel, torch.full_like(far, torch.inf), far)
            enter = torch.maximum(enter, torch.minimum(near, far))
            leave = torch.minimum(leave, torch.maximum(near, far))
        valid &= leave > torch.maximum(enter, torch.zeros_like(enter))
        free = (valid & (ranges[None, :] >= leave)).any(dim=1)
        occluded = (
            valid & (ranges[None, :] <= torch.maximum(enter, torch.zeros_like(enter)))
        ).any(dim=1)
        codes = torch.where(
            hit,
            torch.full_like(output[start:stop], STATE_CODE[Visibility.HIT]),
            torch.where(
                free,
                torch.full_like(output[start:stop], STATE_CODE[Visibility.KNOWN_FREE]),
                torch.where(
                    occluded,
                    torch.full_like(output[start:stop], STATE_CODE[Visibility.OCCLUDED]),
                    torch.zeros_like(output[start:stop]),
                ),
            ),
        )
        output[start:stop] = codes
    return output


def self_check() -> dict[str, list[str]]:
    from dtr_c28_ray_visibility import classify_query_voxels

    origin = (0.06, 0.06, 0.06)
    endpoints = ((0.30, 0.06, 0.06), (0.06, 0.30, 0.06))
    queries = ((2, 0, 0), (1, 0, 0), (3, 0, 0), (0, 2, 0), (0, 1, 0), (4, 4, 0))
    reference = classify_query_voxels(origin, endpoints, queries)
    cpu = CpuRayIndex.build(origin, endpoints).classify(queries)
    if cpu != reference:
        raise AssertionError(f"CPU fast/reference mismatch: {cpu!r} != {reference!r}")
    return {"reference": [value.value for value in reference], "cpu": [value.value for value in cpu]}


if __name__ == "__main__":
    print(self_check())
