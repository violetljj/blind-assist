from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_minimal.local_expansion import (
    fit_fixed_grid_local_affine,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.sparse_flow import (
    SparseTrackResult,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    stage_b_translation_depth_oracle_object_approach_r0 as runner,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_stage_b_translation_depth_oracle_execution_independent_r0 as validator,
)


ROOT = Path(__file__).resolve().parents[4]


def _protocol() -> dict:
    return json.loads(
        (ROOT / runner.PROTOCOL_RELATIVE).read_text(encoding="utf-8")
    )


def _grid_tracks(expansion: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    points = []
    current = []
    dt = 1.0 / 60.0
    for index in range(9):
        row, column = divmod(index, 3)
        x0 = column * runner.geometry.WIDTH / 3.0
        x1 = (column + 1) * runner.geometry.WIDTH / 3.0
        y0 = row * runner.geometry.HEIGHT / 3.0
        y1 = (row + 1) * runner.geometry.HEIGHT / 3.0
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        for y in np.linspace(y0 + 12, y1 - 12, 4):
            for x in np.linspace(x0 + 12, x1 - 12, 4):
                point = np.asarray((x, y), dtype=np.float64)
                velocity = expansion * (point - np.asarray((cx, cy)))
                points.append(point)
                current.append(point + velocity * dt)
    return np.asarray(points), np.asarray(current)


def test_memory_successor_is_6_gib_with_4_gib_inflight_floor() -> None:
    spec = json.loads((ROOT / runner.SPEC_RELATIVE).read_text(encoding="utf-8"))
    assert spec["memory_gate"]["launch_and_refill_available_ram_bytes"] == 6 * 1024**3
    assert spec["memory_gate"]["scheduler_heartbeat_available_ram_bytes"] == 6 * 1024**3
    assert spec["memory_gate"]["exactly_6_gib_passes"] is True
    assert spec["memory_gate"]["in_flight_emergency_runtime_floor_bytes"] == 4 * 1024**3


def test_final_track_splice_uses_managed_only_for_activated_cells() -> None:
    previous, current = _grid_tracks()
    initial = SparseTrackResult(
        previous.astype(np.float32),
        current.astype(np.float32),
        np.zeros(len(previous), dtype=np.float32),
        len(previous),
    )
    managed_current = current.copy()
    managed_current[runner._cell_mask(previous, 4), 0] += 2.0
    managed = SparseTrackResult(
        previous.astype(np.float32),
        managed_current.astype(np.float32),
        np.zeros(len(previous), dtype=np.float32),
        len(previous),
    )
    final = runner._final_compensated_tracks(
        [initial, initial, managed, managed], [4]
    )
    selected = runner._cell_mask(final.previous_points, 4)
    np.testing.assert_allclose(
        final.current_points[selected, 0],
        managed_current[runner._cell_mask(previous, 4), 0],
    )
    outside = ~selected
    np.testing.assert_allclose(
        final.current_points[outside], current[~runner._cell_mask(previous, 4)]
    )


def test_independent_affine_reconstruction_matches_frozen_fit() -> None:
    previous, current = _grid_tracks()
    parameters = _protocol()["local_affine"]
    frozen = fit_fixed_grid_local_affine(
        runner._tracks(previous, current),
        1.0 / 60.0,
        (runner.geometry.HEIGHT, runner.geometry.WIDTH),
        parameters,
    )
    independent = validator.fit_cells(
        previous.astype(np.float32),
        current.astype(np.float32),
        1.0 / 60.0,
        parameters,
    )
    assert all(cell.evaluable for cell in frozen)
    assert all(cell["evaluable"] for cell in independent)
    np.testing.assert_allclose(
        [cell.expansion for cell in frozen],
        [cell["expansion"] for cell in independent],
        atol=2e-7,
        rtol=0.0,
    )


def test_absolute_reduction_is_median_abs_not_abs_median() -> None:
    pairs = [
        {
            "full_scene": {
                "evaluable": True,
                "baseline_signed_per_s": 0.0,
                "oracle_signed_per_s": 0.0,
                "baseline_absolute_per_s": value,
                "oracle_absolute_per_s": value / 2.0,
            }
        }
        for value in (0.01, 0.02, 0.50)
    ]
    reduced = validator.reduce_pairs(pairs, "full_scene")
    assert reduced["baseline_absolute_p50_per_s"] == 0.02
    assert reduced["oracle_absolute_p50_per_s"] == 0.01


def test_trigger_is_strict_and_abstention_resets() -> None:
    values = [0.011, 0.011, None, 0.011, 0.011, 0.011, 0.01]
    pairs = []
    for value in values:
        pairs.append(
            {
                "full_scene": {
                    "evaluable": value is not None,
                    "oracle_signed_per_s": value,
                }
            }
        )
    count, longest = validator.triggers(
        pairs, "full_scene", "oracle_signed_per_s"
    )
    assert count == 1
    assert longest == 3


def test_target_mask_requires_one_cell_but_full_scene_keeps_five() -> None:
    previous, current = _grid_tracks()
    parameters = _protocol()["local_affine"]
    target = np.zeros(len(previous), dtype=np.uint8)
    target[runner._cell_mask(previous, 4)] = 1
    full, _, _ = validator.recompute_pair(
        previous,
        current,
        current,
        target,
        1.0 / 60.0,
        True,
        parameters,
        False,
    )
    target_only, _, _ = validator.recompute_pair(
        previous,
        current,
        current,
        target,
        1.0 / 60.0,
        full["evaluable"],
        parameters,
        True,
    )
    assert len(full["common_cell_indices"]) == 9
    assert full["evaluable"] is True
    assert target_only["common_cell_indices"] == [4]
    assert target_only["evaluable"] is True
