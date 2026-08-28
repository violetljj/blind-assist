"""Exact CPU reference for frozen-grid LiDAR ray visibility.

The module classifies integer 3-D voxel keys using one sensor origin and the
current raw LiDAR endpoints from that sensor.  It deliberately has no angular,
range, or support threshold: visibility follows only exact Amanatides-Woo grid
traversal on the existing 0.12 metre DTR voxel grid.

Classification precedence is ``HIT > KNOWN_FREE > OCCLUDED > UNSENSED``:

* an endpoint voxel is ``HIT``;
* a voxel traversed before at least one endpoint is ``KNOWN_FREE``;
* a voxel traversed by the same ray after a nearer endpoint is ``OCCLUDED``;
* every other voxel is ``UNSENSED``.

This is a correctness reference.  Backend selection and acceleration belong
at the caller after a representative workload has been measured.
"""

from __future__ import annotations

from enum import Enum
import math
from numbers import Integral
from typing import Iterable, Sequence


VOXEL_SIZE_M = 0.12

VoxelKey = tuple[int, int, int]


class Visibility(str, Enum):
    """Truth-blind geometric state of a query voxel for one LiDAR sensor."""

    HIT = "HIT"
    KNOWN_FREE = "KNOWN_FREE"
    OCCLUDED = "OCCLUDED"
    UNSENSED = "UNSENSED"


def _point3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        row = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain exactly three finite coordinates") from error
    if len(row) != 3 or not all(math.isfinite(item) for item in row):
        raise ValueError(f"{name} must contain exactly three finite coordinates")
    return row[0], row[1], row[2]


def voxel_key(point_world: Sequence[float]) -> VoxelKey:
    """Return the frozen-grid key containing one finite world-space point."""

    point = _point3(point_world, "point_world")
    return tuple(math.floor(value / VOXEL_SIZE_M) for value in point)  # type: ignore[return-value]


def traverse_voxels(
    start_world: Sequence[float],
    end_world: Sequence[float],
) -> tuple[VoxelKey, ...]:
    """Return voxels intersected by the closed segment, in traversal order.

    Simultaneous boundary crossings advance every tied axis.  Consequently a
    ray that touches only a voxel edge or corner does not manufacture positive
    path length in the adjacent cells.  Both the start and end voxel are
    included.
    """

    start = _point3(start_world, "start_world")
    end = _point3(end_world, "end_world")
    start_key = voxel_key(start)
    end_key = voxel_key(end)
    current = list(start_key)
    direction = tuple(end[axis] - start[axis] for axis in range(3))

    step = [0, 0, 0]
    t_max = [math.inf, math.inf, math.inf]
    t_delta = [math.inf, math.inf, math.inf]
    for axis in range(3):
        if direction[axis] > 0.0:
            step[axis] = 1
            boundary = (float(current[axis]) + 1.0) * VOXEL_SIZE_M
            t_max[axis] = (boundary - start[axis]) / direction[axis]
            t_delta[axis] = VOXEL_SIZE_M / direction[axis]
        elif direction[axis] < 0.0:
            step[axis] = -1
            boundary = float(current[axis]) * VOXEL_SIZE_M
            t_max[axis] = (boundary - start[axis]) / direction[axis]
            t_delta[axis] = -VOXEL_SIZE_M / direction[axis]

    # Each transition changes at least one key coordinate toward end_key.
    transition_budget = sum(abs(end_key[axis] - start_key[axis]) for axis in range(3)) + 1
    traversed: list[VoxelKey] = []
    for _ in range(transition_budget):
        traversed.append((int(current[0]), int(current[1]), int(current[2])))
        if tuple(current) == end_key:
            return tuple(traversed)
        next_t = min(t_max)
        for axis in range(3):
            if t_max[axis] == next_t:
                current[axis] += step[axis]
                t_max[axis] += t_delta[axis]

    raise RuntimeError("DDA transition budget exhausted before reaching endpoint voxel")


def _farthest_query_corner_distance(
    origin: tuple[float, float, float],
    query_keys: Sequence[VoxelKey],
) -> float:
    """Return a radius enclosing every corner of every query voxel."""

    farthest = 0.0
    for key in query_keys:
        squared_distance = 0.0
        for axis in range(3):
            lower = key[axis] * VOXEL_SIZE_M
            upper = lower + VOXEL_SIZE_M
            far_corner = lower if abs(lower - origin[axis]) >= abs(upper - origin[axis]) else upper
            squared_distance += (far_corner - origin[axis]) ** 2
        farthest = max(farthest, math.sqrt(squared_distance))
    return farthest


def _normalized_query_keys(query_voxel_keys: Iterable[Sequence[int]]) -> tuple[VoxelKey, ...]:
    rows = []
    for value in query_voxel_keys:
        try:
            array = tuple(value)
        except TypeError as error:
            raise ValueError("each query voxel key must have shape (3,)") from error
        if len(array) != 3:
            raise ValueError("each query voxel key must have shape (3,)")
        if not all(isinstance(item, Integral) for item in array):
            raise ValueError("query voxel keys must be integers")
        rows.append((int(array[0]), int(array[1]), int(array[2])))
    return tuple(rows)


def classify_query_voxels(
    sensor_origin_world: Sequence[float],
    raw_endpoints_world: Iterable[Sequence[float]],
    query_voxel_keys: Iterable[Sequence[int]],
) -> tuple[Visibility, ...]:
    """Classify query voxels for one sensor sweep in input order.

    Duplicate queries are preserved.  Duplicate endpoints do not change the
    result.  An endpoint equal to the sensor origin can mark only its own voxel
    as ``HIT`` because it has neither a free segment nor a direction to extend.
    """

    origin = _point3(sensor_origin_world, "sensor_origin_world")
    endpoints = tuple(_point3(value, "raw endpoint") for value in raw_endpoints_world)
    queries = _normalized_query_keys(query_voxel_keys)
    if not queries:
        return ()

    query_set = set(queries)
    hit: set[VoxelKey] = set()
    known_free: set[VoxelKey] = set()
    occluded: set[VoxelKey] = set()
    farthest_query_m = _farthest_query_corner_distance(origin, queries)

    for endpoint in endpoints:
        endpoint_key = voxel_key(endpoint)
        if endpoint_key in query_set:
            hit.add(endpoint_key)

        finite_ray = traverse_voxels(origin, endpoint)
        known_free.update(query_set.intersection(finite_ray[:-1]))

        direction = tuple(endpoint[axis] - origin[axis] for axis in range(3))
        endpoint_distance_m = math.sqrt(sum(value * value for value in direction))
        if endpoint_distance_m == 0.0 or farthest_query_m <= endpoint_distance_m:
            continue
        scale = math.nextafter(
            farthest_query_m / endpoint_distance_m,
            math.inf,
        )
        extension = tuple(origin[axis] + direction[axis] * scale for axis in range(3))
        behind_endpoint = traverse_voxels(endpoint, extension)[1:]
        occluded.update(query_set.intersection(behind_endpoint))

    output = []
    for query in queries:
        if query in hit:
            output.append(Visibility.HIT)
        elif query in known_free:
            output.append(Visibility.KNOWN_FREE)
        elif query in occluded:
            output.append(Visibility.OCCLUDED)
        else:
            output.append(Visibility.UNSENSED)
    return tuple(output)


def self_check() -> dict[VoxelKey, str]:
    """Run a minimal deterministic axial-ray contract check."""

    origin = (0.06, 0.06, 0.06)
    endpoints = ((0.30, 0.06, 0.06),)
    queries: tuple[VoxelKey, ...] = (
        (2, 0, 0),  # endpoint
        (1, 0, 0),  # before endpoint
        (3, 0, 0),  # same ray, behind endpoint
        (0, 2, 0),  # outside the sensed ray
    )
    observed = classify_query_voxels(origin, endpoints, queries)
    expected = (
        Visibility.HIT,
        Visibility.KNOWN_FREE,
        Visibility.OCCLUDED,
        Visibility.UNSENSED,
    )
    if observed != expected:
        raise AssertionError(f"ray visibility self-check failed: {observed!r}")
    return {query: state.value for query, state in zip(queries, observed)}


if __name__ == "__main__":
    print(self_check())
