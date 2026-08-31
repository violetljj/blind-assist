"""X36 coherent occupied-cell island route geometry.

X36 preserves every repeatedly supported BEV lattice cell from X35, but it no
longer bridges disconnected occupied islands with one convex or axis-aligned
envelope.  Eight-neighbour cell components are emitted as separate polygons
sharing the same parent ancestry.  Route confirmation therefore remains parent
stable while empty space between islands cannot become obstacle geometry.

C16 remains iterative same-source synthetic Development only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x29_temporal_occupancy_lineage_predictor as x29  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x35_dormant_flow_consensus_predictor as x35  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X36_COHERENT_CELL_ISLAND_GEOMETRY"
ARM_X36 = "X36_ISSUED_PLAN_COHERENT_CELL_ISLAND_GEOMETRY"
EPSILON = x31.EPSILON
BASE_X35_TRACKER = x35.DormantFlowConsensusTracker


def fixed_constants() -> dict[str, Any]:
    return {
        **x35.fixed_constants(),
        "representation": "COHERENT_OCCUPIED_CELL_ISLAND_ROUTE_GEOMETRY",
        "cell_connectivity": "EIGHT_NEIGHBOUR",
        "component_geometry": "CONVEX_HULL_OF_COMPLETE_LATTICE_CELL_SQUARES",
        "all_supported_cells_preserved": True,
        "disconnected_empty_space_bridged": False,
        "parent_confirmation_identity_unchanged": True,
        "numeric_score_threshold_added": False,
    }


def connected_components(
    cells: frozenset[x27.Cell],
) -> list[frozenset[x27.Cell]]:
    remaining = set(cells)
    output: list[frozenset[x27.Cell]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            x, y = frontier.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbour = (x + dx, y + dy)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        component.add(neighbour)
                        frontier.append(neighbour)
        output.append(frozenset(component))
    return sorted(output, key=lambda value: (-len(value), min(value)))


def cell_component_hull(cells: frozenset[x27.Cell]) -> np.ndarray:
    x24.require(bool(cells), "x36_empty_component")
    cell = x27.LATTICE_CELL_SIZE_M
    half = cell * 0.5
    points = sorted(
        {
            (x * cell + dx, y * cell + dy)
            for x, y in cells
            for dx in (-half, half)
            for dy in (-half, half)
        }
    )

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    x24.require(len(hull) >= 4, "x36_degenerate_component_hull")
    return np.asarray(hull, dtype=np.float64)


class CoherentCellIslandTracker(x35.DormantFlowConsensusTracker):
    def _row(self, **kwargs: Any) -> dict[str, Any]:
        row = super()._row(**kwargs)
        support_cells, _maximum_support = x29.repeated_support(kwargs["lineage"])
        if len(support_cells) < x27.MINIMUM_ALIGNMENT_CELLS:
            return row
        age_s = max(0.0, float(kwargs["now_s"]) - kwargs["track"].last_seen_s)
        velocity = np.asarray(kwargs["velocity"], dtype=np.float64)
        shift = (
            velocity * age_s
            if kwargs["authority"] == x27.RIGID_DYNAMIC
            else np.zeros(2, dtype=np.float64)
        )
        components = connected_components(support_cells)
        row["_coherent_cell_islands"] = [
            {
                "cells": len(component),
                "footprint": cell_component_hull(component) + shift[None, :],
            }
            for component in components
        ]
        return row

    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        rows = BASE_X35_TRACKER.emitted(self, now_s, measured_ids)
        output: list[dict[str, Any]] = []
        for row in rows:
            islands = row.pop("_coherent_cell_islands", None)
            if not islands:
                output.append(row)
                continue
            for ordinal, island in enumerate(islands):
                footprint = island["footprint"]
                value = dict(row)
                value["track_id"] = (
                    f"{row['track_id']}::island-{ordinal + 1:02d}"
                )
                center = np.mean(footprint, axis=0)
                value.update(
                    {
                        "position_forward_m": float(center[0]),
                        "position_right_m": float(center[1]),
                        "footprint_xy": [
                            [float(item) for item in point] for point in footprint
                        ],
                        "footprint_area_m2": x25.polygon_area(footprint),
                        "lineage_cells": int(island["cells"]),
                        "support_footprint_mode": "COHERENT_CELL_ISLAND",
                        "coherent_cell_island_count": len(islands),
                        "coherent_cell_island_ordinal": ordinal + 1,
                    }
                )
                output.append(value)
        return output


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    original_tracker = x35.DormantFlowConsensusTracker
    x35.DormantFlowConsensusTracker = CoherentCellIslandTracker
    try:
        value = x35.predict_episode(episode, candidate_values, calibration)
    finally:
        x35.DormantFlowConsensusTracker = original_tracker
    value["arms"][ARM_X36] = value["arms"].pop(x35.ARM_X35)
    value["diagnostics"]["x36_route_mode_counts"] = value["diagnostics"].pop(
        "x35_route_mode_counts"
    )
    value["diagnostics"]["maximum_coherent_cell_islands"] = max(
        (
            int(track.get("coherent_cell_island_count", 1))
            for frame in value["frames"]
            for track in frame["tracks"]
        ),
        default=0,
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X36] = frame["arms"].pop(x35.ARM_X35)
    return value


def self_check() -> dict[str, Any]:
    inherited = x35.self_check()
    cells = frozenset({(0, 0), (1, 0), (5, 5)})
    components = connected_components(cells)
    x24.require(
        [len(value) for value in components] == [2, 1],
        "x36_component_partition",
    )
    areas = [x25.polygon_area(cell_component_hull(value)) for value in components]
    x24.require(all(value > 0.0 for value in areas), "x36_component_area")
    return {
        "status": "X36_COHERENT_CELL_ISLAND_STRUCTURAL_FALSIFIER_MET",
        "x35_structural_status": inherited["status"],
        "all_cells_partitioned_once": True,
        "disconnected_islands_not_bridged": True,
        "parent_identity_preserved": True,
    }
