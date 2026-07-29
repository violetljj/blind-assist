from __future__ import annotations

import ast
import copy
from contextlib import ExitStack
import hashlib
import inspect
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    r3_rotation_leakage_source_localization_r0 as localization,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_r3_rotation_leakage_source_localization_execution_independent_r0
    as validator,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(localization.canonical_bytes(value))


def _execute_pilot_claim_fixture(
    root: Path,
    output: Path,
    atomic_claim: object,
) -> None:
    pilot_parent = root / "pilots"
    pilot_parent.mkdir(exist_ok=True)
    runner_path = root / "runner.py"
    runner_path.write_text("# synthetic runner binding\n", encoding="utf-8")
    manifest_path = pilot_parent / "manifest.json"
    _write_json(manifest_path, {"fixture": "disjoint"})
    disjoint_path = pilot_parent / "disjoint.json"
    _write_json(disjoint_path, {"status": "PASS"})
    authority = {
        "activation_sha256": "a" * 64,
        "contract_sha256": "c" * 64,
        "identity_sha256": "i" * 64,
        "preflight_receipt_sha256": "p" * 64,
        "protocol_sha256": "q" * 64,
        "source_bindings": [],
    }
    with (
        mock.patch.object(localization, "repo_root", return_value=root),
        mock.patch.object(localization, "__file__", runner_path),
        mock.patch.object(localization, "PILOT_PARENT_RELATIVE", "pilots"),
        mock.patch.object(localization, "FORMAL_ROOT_RELATIVE", "formal"),
        mock.patch.object(
            localization, "validate_authority", return_value=authority
        ),
        mock.patch.object(
            localization,
            "validate_pilot_disjoint_receipt",
            return_value={
                "sha256": "d" * 64,
                "receipt": {"status": "PASS"},
            },
        ),
        mock.patch.object(
            localization,
            "pilot_tasks",
            return_value=[
                {
                    "cluster_id": "PILOT_ONLY_CLUSTER_000",
                    "pair_count": 1,
                }
            ],
        ),
        mock.patch.object(
            localization.psutil,
            "virtual_memory",
            return_value=SimpleNamespace(available=8 * localization.GIB),
        ),
        mock.patch.object(
            localization, "_atomic_claim_root", side_effect=atomic_claim
        ),
    ):
        localization.execute(
            mode=localization.PILOT_MODE,
            output_root=output,
            pilot_manifest_path=manifest_path,
            pilot_disjoint_receipt_path=disjoint_path,
            pilot_workers=1,
        )


def _primitive_arrays() -> dict[str, np.ndarray]:
    initial_previous = np.asarray(
        [[10.0, 10.0], [20.0, 20.0]], dtype=np.float32
    )
    current = np.asarray(
        [[11.0, 10.0], [21.0, 20.0]], dtype=np.float32
    )
    source_aligned = current.astype(np.float64)
    source_valid = np.asarray([True, True], dtype=np.bool_)
    source_index = np.asarray([0, 1], dtype=np.int32)
    fb_error = np.zeros(2, dtype=np.float32)
    consensus_pair_cell = np.asarray(
        [
            [0, cell_index]
            for _path_index in range(2)
            for cell_index in range(validator.GRID_CELLS)
        ],
        dtype=np.int32,
    )
    consensus_path = np.asarray(
        ["RAW_FINAL"] * validator.GRID_CELLS
        + ["COMPENSATED_FINAL"] * validator.GRID_CELLS,
        dtype="<U20",
    )
    return {
        "initial_offsets": np.asarray([0, 2], dtype=np.int64),
        "initial_previous": initial_previous.copy(),
        "initial_source_offsets": np.asarray([0, 2], dtype=np.int64),
        "initial_source_index": source_index.copy(),
        "initial_source_depth_m": np.asarray(
            [2.0, 3.0], dtype=np.float64
        ),
        "initial_source_object_id": np.asarray(
            [1001, 1001], dtype=np.int32
        ),
        "initial_source_world": np.asarray(
            [[0.0, 0.0, 2.0], [0.1, 0.1, 3.0]], dtype=np.float64
        ),
        "initial_source_current_pixel": source_aligned.copy(),
        "initial_source_aligned": source_aligned.copy(),
        "initial_source_valid": source_valid.copy(),
        "lk_forward_offsets": np.asarray([0, 2], dtype=np.int64),
        "lk_forward_source_index": source_index.copy(),
        "lk_forward_points": current.copy(),
        "lk_forward_available": np.asarray(
            [True, True], dtype=np.bool_
        ),
        "lk_backward_offsets": np.asarray([0, 2], dtype=np.int64),
        "lk_backward_source_index": source_index.copy(),
        "lk_backward_points": initial_previous.copy(),
        "lk_backward_level": np.asarray([0, 0], dtype=np.int16),
        "lk_backward_error": fb_error.copy(),
        "lk_backward_available": np.asarray(
            [True, True], dtype=np.bool_
        ),
        "lk_fb_pass": np.asarray([True, True], dtype=np.bool_),
        "lk_mask_pass": np.asarray([True, True], dtype=np.bool_),
        "lk_accepted": np.asarray([True, True], dtype=np.bool_),
        "lk_rejection_reason": np.asarray(
            ["ACCEPTED", "ACCEPTED"], dtype="<U24"
        ),
        "accepted_offsets": np.asarray([0, 2], dtype=np.int64),
        "accepted_previous": initial_previous.copy(),
        "accepted_current": current.copy(),
        "accepted_fb_error": fb_error.copy(),
        "accepted_source_offsets": np.asarray([0, 2], dtype=np.int64),
        "accepted_source_index": source_index.copy(),
        "accepted_source_aligned": source_aligned.copy(),
        "accepted_source_valid": source_valid.copy(),
        "managed_offsets": np.asarray([0, 2], dtype=np.int64),
        "managed_previous": initial_previous.copy(),
        "managed_current": current.copy(),
        "managed_fb_error": fb_error.copy(),
        "managed_source_offsets": np.asarray([0, 2], dtype=np.int64),
        "managed_source_index": source_index.copy(),
        "managed_source_aligned": source_aligned.copy(),
        "managed_source_valid": source_valid.copy(),
        "managed_path_class": np.asarray(
            ["BASELINE", "BASELINE"], dtype="<U16"
        ),
        "fit_offsets": np.asarray([0, 2, 4, 4, 4], dtype=np.int64),
        "fit_previous": np.vstack(
            (initial_previous, initial_previous)
        ).astype(np.float32),
        "fit_current": np.vstack((current, current)).astype(np.float32),
        "fit_fb_error": np.zeros(4, dtype=np.float32),
        "fit_group_path": np.asarray(
            validator.FIT_GROUP_PATHS, dtype="<U24"
        ),
        "activated_offsets": np.asarray([0, 0], dtype=np.int64),
        "activated_cell_index": np.empty((0,), dtype=np.int8),
        "merge_candidate_offsets": np.asarray(
            [0, 0, 0], dtype=np.int64
        ),
        "merge_candidate_previous": np.empty((0, 2), dtype=np.float32),
        "merge_candidate_current": np.empty((0, 2), dtype=np.float32),
        "merge_candidate_fb_error": np.empty((0,), dtype=np.float32),
        "merge_candidate_path_class": np.asarray([], dtype="<U16"),
        "merge_candidate_selected": np.empty((0,), dtype=np.bool_),
        "merge_group_path": np.asarray(
            validator.MERGE_GROUP_PATHS, dtype="<U24"
        ),
        "consensus_offsets": np.zeros(
            validator.GRID_CELLS * 2 + 1, dtype=np.int64
        ),
        "consensus_previous": np.empty((0, 2), dtype=np.float32),
        "consensus_current": np.empty((0, 2), dtype=np.float32),
        "consensus_pair_cell": consensus_pair_cell,
        "consensus_path": consensus_path,
    }


def _copy_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in arrays.items()}


def _early_warp_arrays(pair_count: int = 2) -> dict[str, np.ndarray]:
    fit_groups = pair_count * len(validator.FIT_GROUP_PATHS)
    merge_groups = pair_count * len(validator.MERGE_GROUP_PATHS)
    consensus_groups = pair_count * validator.GRID_CELLS * 2
    return {
        "fit_offsets": np.zeros(fit_groups + 1, dtype=np.int64),
        "fit_previous": np.empty((0, 2), dtype=np.float32),
        "fit_current": np.empty((0, 2), dtype=np.float32),
        "fit_fb_error": np.empty((0,), dtype=np.float32),
        "fit_group_path": np.asarray(
            list(validator.FIT_GROUP_PATHS) * pair_count, dtype="<U24"
        ),
        "activated_offsets": np.zeros(pair_count + 1, dtype=np.int64),
        "activated_cell_index": np.empty((0,), dtype=np.int8),
        "merge_candidate_offsets": np.zeros(
            merge_groups + 1, dtype=np.int64
        ),
        "merge_group_path": np.asarray(
            list(validator.MERGE_GROUP_PATHS) * pair_count, dtype="<U24"
        ),
        "consensus_offsets": np.zeros(
            consensus_groups + 1, dtype=np.int64
        ),
        "consensus_previous": np.empty((0, 2), dtype=np.float32),
        "consensus_current": np.empty((0, 2), dtype=np.float32),
        "consensus_pair_cell": np.asarray(
            [
                [pair_index, cell_index]
                for pair_index in range(pair_count)
                for _path_index in range(2)
                for cell_index in range(validator.GRID_CELLS)
            ],
            dtype=np.int32,
        ),
        "consensus_path": np.asarray(
            [
                path
                for _pair_index in range(pair_count)
                for path in ("RAW_FINAL", "COMPENSATED_FINAL")
                for _cell_index in range(validator.GRID_CELLS)
            ],
            dtype="<U20",
        ),
    }


class ReductionTests(unittest.TestCase):
    def test_hf7_matches_linear_interpolation(self) -> None:
        values = [0.0, 1.0, 2.0, 10.0]
        self.assertEqual(localization.hf7(values, 0.0), 0.0)
        self.assertEqual(localization.hf7(values, 1.0), 10.0)
        self.assertAlmostEqual(localization.hf7(values, 0.5), 1.5)
        self.assertAlmostEqual(localization.hf7(values, 0.9), 7.6)

    def test_hf7_rejects_empty_or_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            localization.hf7([], 0.9)
        with self.assertRaises(ValueError):
            localization.hf7([0.0, float("nan")], 0.9)

    def test_local_cells_early_exit_keeps_fixed_consensus_groups(
        self,
    ) -> None:
        capture = SimpleNamespace(common=None, fit_calls=[])
        summary, cells, consensus = localization._local_cells(capture, {})

        self.assertFalse(summary["evaluable"])
        self.assertFalse(summary["numeric_reproduced"])
        self.assertEqual(cells, {"RAW_FINAL": [], "COMPENSATED_FINAL": []})
        self.assertEqual(len(consensus), 18)
        self.assertEqual(
            [(path, cell_index) for path, cell_index, _ in consensus],
            [
                (path, cell_index)
                for path in ("RAW_FINAL", "COMPENSATED_FINAL")
                for cell_index in range(9)
            ],
        )
        for _, _, values in consensus:
            self.assertEqual(values.shape, (0, 4))
            self.assertEqual(values.dtype, np.float32)

    def test_validator_accepts_independently_replayed_early_warp_abstention(
        self,
    ) -> None:
        arrays = _early_warp_arrays()
        reconstructed = validator._reconstruct_local_fit(
            arrays,
            pair_index=1,
            dt_s=0.1,
            activated=[],
            overlap_fraction=0.5,
        )
        self.assertTrue(reconstructed["early_warp_abstention"])
        self.assertEqual(reconstructed["raw_cells"], [])
        self.assertEqual(reconstructed["cells"], [])

        local_record = {
            "evaluable": False,
            "numeric_reproduced": False,
            "reproduction_error_max_abs": None,
            "common_cell_indices": [],
            "signed_per_s": None,
            "absolute_per_s": None,
            "raw_cells": [],
            "compensated_cells": [],
        }
        r3_record = {
            "evaluable": False,
            "reason": "ROTATION_WARP_VALID_COVERAGE_BELOW_0_75",
            "signed_per_s": None,
            "absolute_per_s": None,
            "common_cell_count": None,
        }
        validator._validate_early_warp_abstention_records(
            local_record, r3_record, "PAIR:1"
        )

        with self.assertRaisesRegex(
            validator.InvalidExecution, "ACTIVATED_CELL_REPLAY"
        ):
            validator._reconstruct_local_fit(
                arrays,
                pair_index=1,
                dt_s=0.1,
                activated=[],
                overlap_fraction=validator.WARP_OVERLAP_FLOOR,
            )

    def test_validator_rejects_early_warp_abstention_mutations(self) -> None:
        mutations = []
        for name, offset_key in (
            ("fit", "fit_offsets"),
            ("activated", "activated_offsets"),
            ("merge", "merge_candidate_offsets"),
            ("consensus", "consensus_offsets"),
        ):
            arrays = _early_warp_arrays()
            arrays[offset_key][-1] = 1
            mutations.append((name, arrays))

        for name, arrays in mutations:
            with self.subTest(name=name):
                with self.assertRaises(validator.InvalidExecution):
                    validator._reconstruct_local_fit(
                        arrays,
                        pair_index=1,
                        dt_s=0.1,
                        activated=[],
                        overlap_fraction=0.5,
                    )

        bad_r3 = {
            "evaluable": False,
            "reason": "COMMON_GRID_SUPPORT_BELOW_5_OF_9",
            "signed_per_s": None,
            "absolute_per_s": None,
            "common_cell_count": None,
        }
        with self.assertRaisesRegex(
            validator.InvalidExecution, "EARLY_WARP_R3"
        ):
            validator._validate_early_warp_abstention_records(
                {
                    "evaluable": False,
                    "numeric_reproduced": False,
                    "reproduction_error_max_abs": None,
                    "common_cell_indices": [],
                    "signed_per_s": None,
                    "absolute_per_s": None,
                    "raw_cells": [],
                    "compensated_cells": [],
                },
                bad_r3,
                "PAIR:1",
            )

    def test_audit_coefficients_follow_numeric_representation_amendment(
        self,
    ) -> None:
        coefficients = np.asarray(
            [[24.0, -2.0], [3.0, 64.0], [7.0, 9.0]],
            dtype=np.float64,
        )
        x0, y0, x1, y1 = validator.cell_bounds(1)
        expansion = float(
            0.5
            * (
                coefficients[0, 0] / max(0.5 * (x1 - x0), 1.0)
                + coefficients[1, 1] / max(0.5 * (y1 - y0), 1.0)
            )
        )
        validator._validate_audit_coefficients(
            coefficients,
            expansion,
            1,
            "COEFFICIENT_TEST",
        )

        with self.assertRaisesRegex(
            validator.InvalidExecution, "EXPANSION_FORMULA"
        ):
            validator._validate_audit_coefficients(
                coefficients,
                expansion + 1e-9,
                1,
                "COEFFICIENT_TEST",
            )
        nonfinite = coefficients.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(
            validator.InvalidExecution, "SHAPE_OR_FINITE"
        ):
            validator._validate_audit_coefficients(
                nonfinite,
                expansion,
                1,
                "COEFFICIENT_TEST",
            )

    def test_reduce_layer_uses_fixed_denominator_and_absolute_channel(self) -> None:
        rows = []
        for index in range(451):
            rows.append(
                {
                    "pair_index": index,
                    "audit_evaluable": True,
                    "audit_signed_per_s": -0.02 if index < 200 else 0.01,
                    "audit_absolute_per_s": 0.02 if index < 200 else 0.01,
                }
            )
        for index in range(451, 601):
            rows.append(
                {
                    "pair_index": index,
                    "audit_evaluable": False,
                    "audit_signed_per_s": None,
                    "audit_absolute_per_s": None,
                }
            )
        result = localization.reduce_layer(rows, "audit", planned_pairs=601)
        self.assertEqual(result["evaluable_pairs"], 451)
        self.assertEqual(result["coverage_status"], "PASS")
        self.assertAlmostEqual(result["coverage_fraction_fixed"], 451 / 601)
        self.assertGreater(result["absolute_p90_per_s"], 0.0)
        self.assertNotEqual(
            result["absolute_p90_per_s"],
            abs(result["signed_p50_per_s"]),
        )

        rows[450]["audit_evaluable"] = False
        rows[450]["audit_signed_per_s"] = None
        rows[450]["audit_absolute_per_s"] = None
        result = localization.reduce_layer(rows, "audit", planned_pairs=601)
        self.assertEqual(result["evaluable_pairs"], 450)
        self.assertEqual(result["coverage_status"], "NOT_EVALUABLE")


class TriggerSemanticsTests(unittest.TestCase):
    @staticmethod
    def _reduction(
        values: list[float | None],
    ) -> dict[str, object]:
        pairs = [
            {
                "evaluable": value is not None,
                "signed": 0.0 if value is None else value,
                "absolute": 0.0 if value is None else abs(value),
            }
            for value in values
        ]
        return validator._signed_absolute_reduction(
            pairs,
            evaluable=lambda pair: pair["evaluable"],
            signed=lambda pair: pair["signed"],
            absolute=lambda pair: pair["absolute"],
            planned_pair_count=len(pairs),
        )

    def test_threshold_equality_is_not_positive_and_resets_streak(self) -> None:
        result = self._reduction([0.02, 0.01, 0.02, 0.02, 0.02])
        self.assertEqual(result["three_pair_trigger_count"], 1)
        self.assertEqual(result["longest_positive_streak"], 3)

    def test_abstention_resets_streak(self) -> None:
        result = self._reduction(
            [0.02, 0.02, None, 0.02, 0.02, 0.02]
        )
        self.assertEqual(result["three_pair_trigger_count"], 1)
        self.assertEqual(result["longest_positive_streak"], 3)

    def test_runner_ast_keeps_strict_greater_than_and_abstention_reset(
        self,
    ) -> None:
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(localization._cluster_worker))
        )
        matching_guards = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            has_strict_if_expression = any(
                isinstance(candidate, ast.IfExp)
                and isinstance(candidate.test, ast.Compare)
                and isinstance(candidate.test.left, ast.Name)
                and candidate.test.left.id == "response"
                and len(candidate.test.ops) == 1
                and isinstance(candidate.test.ops[0], ast.Gt)
                and len(candidate.test.comparators) == 1
                and isinstance(candidate.test.comparators[0], ast.Name)
                and candidate.test.comparators[0].id
                == "THRESHOLD_PER_S"
                for statement in node.body
                for candidate in ast.walk(statement)
            )
            resets_on_else = any(
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "streak"
                    for target in statement.targets
                )
                and isinstance(statement.value, ast.Constant)
                and statement.value.value == 0
                for statement in node.orelse
            )
            if has_strict_if_expression and resets_on_else:
                matching_guards.append(node)
        self.assertEqual(len(matching_guards), 1)
        self.assertEqual(
            localization.THRESHOLD_PER_S,
            validator.LEAKAGE_THRESHOLD_PER_S,
        )


class BoundaryTests(unittest.TestCase):
    def test_twenty_one_pixel_footprint_counts_image_edge_as_boundary(self) -> None:
        valid = np.full((25, 25), 255, dtype=np.uint8)
        boundary, interior, common = localization.boundary_masks(valid, valid)
        self.assertEqual(int(np.count_nonzero(common)), 625)
        self.assertEqual(int(np.count_nonzero(interior)), 25)
        self.assertEqual(int(np.count_nonzero(boundary)), 600)
        self.assertFalse(bool(interior[9, 12]))
        self.assertTrue(bool(interior[10, 10]))
        self.assertTrue(bool(interior[14, 14]))
        self.assertFalse(bool(interior[15, 12]))

    def test_internal_invalid_pixel_expands_boundary_by_chebyshev_ten(self) -> None:
        valid = np.full((45, 45), 255, dtype=np.uint8)
        valid[22, 22] = 0
        boundary, interior, common = localization.boundary_masks(valid, valid)
        self.assertFalse(bool(common[22, 22]))
        self.assertTrue(bool(boundary[12, 12]))
        self.assertFalse(bool(interior[12, 12]))
        self.assertTrue(bool(interior[11, 11]))

    def test_source_error_stratum_does_not_mutate_shared_validity(self) -> None:
        points = np.asarray([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        valid = np.asarray([True, True], dtype=np.bool_)
        before = valid.copy()
        stratum = np.zeros(
            (localization.geometry.HEIGHT, localization.geometry.WIDTH),
            dtype=np.bool_,
        )
        stratum[10, 10] = True

        localization._ols_error_cells(
            points,
            points.copy(),
            valid,
            1.0 / 60.0,
            {"minimum_tracks_per_cell": 12},
            stratum,
        )

        self.assertTrue(np.array_equal(valid, before))

    def test_boundary_and_interior_evaluability_are_independently_bound(
        self,
    ) -> None:
        recorded = {
            "evaluable": False,
            "boundary_evaluable": False,
            "interior_evaluable": True,
        }
        recomputed = {
            "evaluable": False,
            "boundary": {"evaluable": False},
            "interior": {"evaluable": True},
        }
        self.assertEqual(
            validator._boundary_evaluable_states(
                recorded, recomputed, "BOUNDARY_TEST"
            ),
            (False, False, True),
        )

        wrong_joint = dict(recorded)
        wrong_joint["evaluable"] = True
        with self.assertRaisesRegex(
            validator.InvalidExecution, "JOINT_EVALUABLE"
        ):
            validator._boundary_evaluable_states(
                wrong_joint, recomputed, "BOUNDARY_TEST"
            )

        wrong_replay = copy.deepcopy(recomputed)
        wrong_replay["interior"]["evaluable"] = False
        with self.assertRaisesRegex(
            validator.InvalidExecution, "REPLAY_EVALUABLE"
        ):
            validator._boundary_evaluable_states(
                recorded, wrong_replay, "BOUNDARY_TEST"
            )

    def test_boundary_cluster_reduction_uses_one_matched_pair_set(
        self,
    ) -> None:
        availability = (
            (True, True, 0.001, 0.001),
            (True, True, 0.002, 0.002),
            (True, False, 99.0, 0.0),
            (False, True, 0.0, 88.0),
        )
        pairs = []
        for boundary_ok, interior_ok, boundary_value, interior_value in (
            availability
        ):
            pairs.append(
                {
                    "geometry": {
                        "evaluable": True,
                        "signed_per_s": 0.0,
                        "absolute_per_s": 0.0,
                        "roundtrip_max_px": 0.0,
                    },
                    "warp": {
                        "evaluable": True,
                        "coordinate_roundtrip_p99_px": 0.0,
                        "interior_gray_absolute_p90": 0.0,
                        "overlap_fraction": 1.0,
                        "interior_support_fraction": 1.0,
                    },
                    "boundary": {
                        "boundary_evaluable": boundary_ok,
                        "interior_evaluable": interior_ok,
                        "boundary_source_error_absolute_per_s": (
                            boundary_value
                        ),
                        "interior_source_error_absolute_per_s": (
                            interior_value
                        ),
                    },
                    "flow": {
                        "accepted": {
                            "evaluable": True,
                            "signed_per_s": 0.0,
                            "absolute_per_s": 0.0,
                        },
                        "managed": {
                            "evaluable": True,
                            "signed_per_s": 0.0,
                            "absolute_per_s": 0.0,
                        },
                    },
                    "local_fit": {
                        "evaluable": True,
                        "signed_per_s": 0.02,
                        "absolute_per_s": 0.02,
                    },
                }
            )

        reduced = validator.reduce_cluster(pairs, 4)
        layer = reduced["layers"]["MASK_BOUNDARY"]
        self.assertEqual(layer["matched_evaluable_pair_count"], 2)
        self.assertEqual(
            layer["boundary_individually_evaluable_pair_count"], 3
        )
        self.assertEqual(
            layer["interior_individually_evaluable_pair_count"], 3
        )
        self.assertEqual(
            layer["comparison_pair_set"], "MATCHED_BOUNDARY_AND_INTERIOR"
        )
        self.assertEqual(layer["status"], "NOT_EVALUABLE")
        self.assertAlmostEqual(
            layer["boundary_source_error_absolute_p90_per_s"], 0.0019
        )
        self.assertAlmostEqual(
            layer["interior_source_error_absolute_p90_per_s"], 0.0019
        )

    def test_geometry_roundtrip_uses_only_geometry_evaluable_pairs(
        self,
    ) -> None:
        pairs = []
        for index in range(4):
            pairs.append(
                {
                    "geometry": {
                        "evaluable": index < 3,
                        "signed_per_s": 0.0,
                        "absolute_per_s": 0.0,
                        "roundtrip_max_px": 0.0 if index < 3 else 100.0,
                    },
                    "warp": {
                        "evaluable": True,
                        "coordinate_roundtrip_p99_px": 0.0,
                        "interior_gray_absolute_p90": 0.0,
                        "overlap_fraction": 1.0,
                        "interior_support_fraction": 1.0,
                    },
                    "boundary": {
                        "boundary_evaluable": True,
                        "interior_evaluable": True,
                        "boundary_source_error_absolute_per_s": 0.0,
                        "interior_source_error_absolute_per_s": 0.0,
                    },
                    "flow": {
                        "accepted": {
                            "evaluable": True,
                            "signed_per_s": 0.0,
                            "absolute_per_s": 0.0,
                        },
                        "managed": {
                            "evaluable": True,
                            "signed_per_s": 0.0,
                            "absolute_per_s": 0.0,
                        },
                    },
                    "local_fit": {
                        "evaluable": True,
                        "signed_per_s": 0.02,
                        "absolute_per_s": 0.02,
                    },
                }
            )

        reduced = validator.reduce_cluster(pairs, 4)
        layer = reduced["layers"]["INPUT_GEOMETRY"]
        self.assertEqual(layer["evaluable_pair_count"], 3)
        self.assertEqual(layer["roundtrip_max_px"], 0.0)
        self.assertEqual(layer["status"], "PASS")

    def test_flow_subprimitives_use_independent_coverage_cohorts(
        self,
    ) -> None:
        pairs = []
        for index in range(4):
            pairs.append(
                {
                    "geometry": {
                        "evaluable": True,
                        "signed_per_s": 0.0,
                        "absolute_per_s": 0.0,
                        "roundtrip_max_px": 0.0,
                    },
                    "warp": {
                        "evaluable": True,
                        "coordinate_roundtrip_p99_px": 0.0,
                        "interior_gray_absolute_p90": 0.0,
                        "overlap_fraction": 1.0,
                        "interior_support_fraction": 1.0,
                    },
                    "boundary": {
                        "boundary_evaluable": True,
                        "interior_evaluable": True,
                        "boundary_source_error_absolute_per_s": 0.0,
                        "interior_source_error_absolute_per_s": 0.0,
                    },
                    "flow": {
                        "accepted": {
                            "evaluable": index in (0, 1, 2),
                            "signed_per_s": 0.0,
                            "absolute_per_s": 0.0,
                        },
                        "managed": {
                            "evaluable": index in (0, 1, 3),
                            "signed_per_s": 0.0,
                            "absolute_per_s": 0.0,
                        },
                    },
                    "local_fit": {
                        "evaluable": True,
                        "signed_per_s": 0.02,
                        "absolute_per_s": 0.02,
                    },
                }
            )

        reduced = validator.reduce_cluster(pairs, 4)
        layer = reduced["layers"]["SPARSE_LK_AND_TRACK_FILTERING"]
        self.assertEqual(layer["accepted"]["evaluable_pair_count"], 3)
        self.assertEqual(layer["managed"]["evaluable_pair_count"], 3)
        self.assertEqual(layer["status"], "PASS")


class ClusterMetricTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[
        dict[str, object], dict[str, object], dict[str, object]
    ]:
        cluster = {
            "cluster_id": "PILOT_ONLY_CLUSTER",
            "sequence_id": "PILOT_ONLY_SEQUENCE",
            "block": "PILOT_ONLY",
            "ordinal": 0,
        }
        independent = {
            "pair_count": 4,
            "coverage_minimum": 3,
            "layers": {
                "INPUT_GEOMETRY": {
                    "status": "PASS",
                    "evaluable_pair_count": 4,
                    "absolute_p90_per_s": 0.0,
                    "roundtrip_max_px": 0.0,
                },
                "ROTATION_WARP": {
                    "status": "PASS",
                    "evaluable_pair_count": 4,
                    "coordinate_roundtrip_pair_p99_cluster_p90_px": 0.0,
                    "interior_gray_absolute_pair_p90_cluster_p90": 0.0,
                },
                "MASK_BOUNDARY": {
                    "status": "PASS",
                    "failure_mode": None,
                    "matched_evaluable_pair_count": 4,
                    "boundary_source_error_absolute_p90_per_s": 0.0,
                    "interior_source_error_absolute_p90_per_s": 0.0,
                },
                "SPARSE_LK_AND_TRACK_FILTERING": {
                    "status": "PASS",
                    "accepted": {
                        "evaluable_pair_count": 4,
                        "absolute_p90_per_s": 0.0,
                    },
                    "managed": {
                        "evaluable_pair_count": 4,
                        "absolute_p90_per_s": 0.0,
                    },
                },
                "LOCAL_AFFINE_AND_FINAL_AGGREGATION": {
                    "status": "FAIL",
                    "evaluable_pair_count": 4,
                    "absolute_p90_per_s": 0.02,
                },
            },
            "final_precondition": {
                "coverage_ok": True,
                "absolute_p90_per_s": 0.02,
                "boundary_failed": True,
            },
        }
        recorded = {
            "schema": "rcle.r3_rotation_leakage_localization.cluster.v1",
            "mode": "PILOT",
            "block": cluster["block"],
            "ordinal": cluster["ordinal"],
            "cluster_id": cluster["cluster_id"],
            "sequence_id": cluster["sequence_id"],
            "planned_pair_count": 4,
            "minimum_evaluable_pair_count": 3,
            "final": {
                "evaluable_pair_count": 4,
                "absolute_per_s_p90": 0.02,
                "reproduced": True,
            },
            "geometry": {},
            "warp": {},
            "mask": {},
            "flow": {},
            "local": {},
            "layers": {
                "INPUT_GEOMETRY": {
                    "status": "PASS",
                    "evaluable_pair_count": 4,
                    "absolute_per_s_p90": 0.0,
                    "roundtrip_max_px_max": 0.0,
                },
                "ROTATION_WARP": {
                    "status": "PASS",
                    "evaluable_pair_count": 4,
                    "coordinate_roundtrip_p99_px_p90": 0.0,
                    "interior_gray_absolute_p90_normalized_p90": 0.0,
                },
                "MASK_BOUNDARY": {
                    "status": "PASS",
                    "evaluable_pair_count": 4,
                    "boundary_absolute_per_s_p90": 0.0,
                    "interior_absolute_per_s_p90": 0.0,
                    "boundary_only_failure": False,
                    "interior_flow_status": "PASS",
                },
                "SPARSE_LK_AND_TRACK_FILTERING": {
                    "status": "PASS",
                    "accepted_status": "PASS",
                    "managed_status": "PASS",
                    "accepted": {
                        "evaluable_pair_count": 4,
                        "absolute_per_s_p90": 0.0,
                    },
                    "managed": {
                        "evaluable_pair_count": 4,
                        "absolute_per_s_p90": 0.0,
                    },
                },
                "LOCAL_AFFINE_AND_FINAL_AGGREGATION": {
                    "status": "FAIL",
                    "evaluable_pair_count": 4,
                    "absolute_per_s_p90": 0.02,
                    "numeric_reproduced": True,
                },
            },
            "route": "PENDING_INDEPENDENT_VALIDATION",
            "ambiguity_rules": {},
            "claim_ceiling": (
                "CONTROLLED_GENERATOR_INTERNAL_MECHANISM_LOCALIZATION_ONLY"
            ),
        }
        recorded["geometry"] = copy.deepcopy(
            recorded["layers"]["INPUT_GEOMETRY"]
        )
        recorded["warp"] = copy.deepcopy(
            recorded["layers"]["ROTATION_WARP"]
        )
        recorded["mask"] = copy.deepcopy(
            recorded["layers"]["MASK_BOUNDARY"]
        )
        recorded["mask"].pop("boundary_only_failure")
        recorded["mask"].pop("interior_flow_status")
        recorded["flow"] = {
            "accepted_status": "PASS",
            "managed_status": "PASS",
            "interior_status": "PASS",
            "accepted": copy.deepcopy(
                recorded["layers"][
                    "SPARSE_LK_AND_TRACK_FILTERING"
                ]["accepted"]
            ),
            "managed": copy.deepcopy(
                recorded["layers"][
                    "SPARSE_LK_AND_TRACK_FILTERING"
                ]["managed"]
            ),
        }
        recorded["local"] = copy.deepcopy(
            recorded["layers"]["LOCAL_AFFINE_AND_FINAL_AGGREGATION"]
        )
        recorded["ambiguity_rules"] = copy.deepcopy(
            validator.CLUSTER_METRIC_AMBIGUITY_RULES
        )
        return cluster, independent, recorded

    def test_decision_metrics_require_exact_schema_and_recomputed_values(
        self,
    ) -> None:
        cluster, independent, recorded = self._fixture()
        validator._compare_runner_metrics(
            recorded, independent, cluster
        )

        extra_key = copy.deepcopy(recorded)
        extra_key["unexpected"] = True
        with self.assertRaisesRegex(
            validator.InvalidExecution, "METRIC_TOP_LEVEL_KEYSET"
        ):
            validator._compare_runner_metrics(
                extra_key, independent, cluster
            )

        extra_layer = copy.deepcopy(recorded)
        extra_layer["layers"]["UNAUTHORIZED_LAYER"] = {
            "status": "PASS"
        }
        with self.assertRaisesRegex(
            validator.InvalidExecution, "METRIC_LAYER_KEYSET"
        ):
            validator._compare_runner_metrics(
                extra_layer, independent, cluster
            )

        reduction_drift = copy.deepcopy(recorded)
        reduction_drift["layers"]["INPUT_GEOMETRY"][
            "absolute_per_s_p90"
        ] = 0.5
        reduction_drift["geometry"]["absolute_per_s_p90"] = 0.5
        with self.assertRaisesRegex(
            validator.InvalidExecution,
            "METRIC_GEOMETRY_ABSOLUTE_P90",
        ):
            validator._compare_runner_metrics(
                reduction_drift, independent, cluster
            )

        governance_mutations = (
            ("mode", "FORMAL", "METRIC_MODE"),
            (
                "claim_ceiling",
                "UNBOUNDED_PRODUCT_SAFETY_AUTHORITY",
                "METRIC_CLAIM_CEILING",
            ),
            (
                "ambiguity_rules",
                {},
                "METRIC_AMBIGUITY_RULES",
            ),
        )
        for key, value, expected in governance_mutations:
            with self.subTest(key=key):
                mutation = copy.deepcopy(recorded)
                mutation[key] = value
                with self.assertRaisesRegex(
                    validator.InvalidExecution, expected
                ):
                    validator._compare_runner_metrics(
                        mutation, independent, cluster
                    )

        summary_drift = copy.deepcopy(recorded)
        summary_drift["geometry"]["absolute_per_s_p90"] = 0.5
        with self.assertRaisesRegex(
            validator.InvalidExecution,
            "METRIC_REDUNDANT_SUMMARY_BINDING",
        ):
            validator._compare_runner_metrics(
                summary_drift, independent, cluster
            )


class WarpPrimitiveTests(unittest.TestCase):
    def test_gray_residual_sign_is_warped_current_minus_previous(self) -> None:
        shape = (200, 200)
        mask = np.full(shape, 255, dtype=np.uint8)
        capture = localization.PairCapture(
            np.full(shape, 10, dtype=np.uint8),
            np.full(shape, 12, dtype=np.uint8),
            mask,
            mask,
            1.0 / 60.0,
        )
        capture.compensation = {
            "result": SimpleNamespace(
                image=np.full(shape, 12, dtype=np.uint8),
                valid_mask=mask,
                overlap_fraction=1.0,
            ),
            "homography": np.eye(3, dtype=np.float64),
        }
        audit, _, _ = localization._warp_audit(
            capture, np.empty((0, 2), dtype=np.float32)
        )
        self.assertTrue(audit["evaluable"])
        self.assertAlmostEqual(
            audit["interior_gray_signed_p50_normalized"],
            2.0 / 255.0,
        )
        self.assertAlmostEqual(
            audit["interior_gray_absolute_p90_normalized"],
            2.0 / 255.0,
        )


class ObservationHookTransparencyTests(unittest.TestCase):
    def test_disjoint_pair_capture_is_return_and_state_transparent(
        self,
    ) -> None:
        protocol = localization.load_json(
            localization.repo_root()
            / (
                "scripts/research/egomotion_compensated_looming/configs/"
                "phase_a_synthetic_signal_audit_r0.json"
            )
        )
        generator = np.random.default_rng(0x5A17C0DE)
        previous_rgb = generator.integers(
            0,
            256,
            size=(
                localization.geometry.HEIGHT,
                localization.geometry.WIDTH,
                3,
            ),
            dtype=np.uint8,
        )
        current_rgb = previous_rgb.copy()
        valid = np.ones(previous_rgb.shape[:2], dtype=np.bool_)
        frame_sha256 = localization.sha256_array(previous_rgb)
        mask_sha256 = localization.sha256_array(valid)
        evaluate_arguments = {
            "pair_index": 0,
            "previous_rgb": previous_rgb,
            "current_rgb": current_rgb,
            "previous_valid": valid,
            "current_valid": valid,
            "previous_timestamp_s": 0.0,
            "current_timestamp_s": 1.0 / 60.0,
            "previous_world_from_camera": np.eye(3, dtype=np.float64),
            "current_world_from_camera": np.eye(3, dtype=np.float64),
            "intrinsic": localization.geometry.K,
            "protocol": protocol,
        }
        baseline_state = localization.r3.PairState()
        instrumented_state = localization.r3.PairState()
        baseline_row = localization.transport.evaluate_pair(
            **evaluate_arguments,
            state=baseline_state,
        )
        previous_gray = localization.transport.rgb_to_gray(previous_rgb)
        current_gray = localization.transport.rgb_to_gray(current_rgb)
        valid_u8 = localization.transport.valid_mask(
            valid, previous_gray.shape
        )
        call_order: list[str] = []

        class RecordingCapture(localization.PairCapture):
            def compensate(self, *args: object, **kwargs: object) -> object:
                call_order.append("compensate")
                return super().compensate(*args, **kwargs)

            def detect(self, *args: object, **kwargs: object) -> np.ndarray:
                call_order.append("detect")
                return super().detect(*args, **kwargs)

            def track(self, *args: object, **kwargs: object) -> object:
                call_order.append("track")
                return super().track(*args, **kwargs)

            def fit(self, *args: object, **kwargs: object) -> object:
                call_order.append("fit")
                return super().fit(*args, **kwargs)

            def observable(
                self, *args: object, **kwargs: object
            ) -> object:
                call_order.append("observable")
                return super().observable(*args, **kwargs)

            def merge(self, *args: object, **kwargs: object) -> object:
                call_order.append("merge")
                return super().merge(*args, **kwargs)

            def common_cells(
                self, *args: object, **kwargs: object
            ) -> object:
                call_order.append("common")
                return super().common_cells(*args, **kwargs)

        capture = RecordingCapture(
            previous_gray,
            current_gray,
            valid_u8,
            valid_u8,
            1.0 / 60.0,
        )
        with capture.patches():
            instrumented_row = localization.transport.evaluate_pair(
                **evaluate_arguments,
                state=instrumented_state,
            )

        self.assertEqual(
            localization.canonical_bytes(instrumented_row),
            localization.canonical_bytes(baseline_row),
        )
        self.assertEqual(instrumented_row, baseline_row)
        self.assertEqual(
            call_order,
            [
                "compensate",
                "detect",
                "track",
                "track",
                "fit",
                "fit",
                "observable",
                "common",
            ],
        )
        self.assertEqual(len(capture.track_calls), 2)
        self.assertEqual(len(capture.fit_calls), 2)
        self.assertEqual(len(capture.observable_calls), 1)
        self.assertEqual(len(capture.merge_calls), 0)
        self.assertIsNotNone(capture.initial_points)
        self.assertIsNotNone(capture.compensation)
        self.assertIsNotNone(capture.common)
        self.assertTrue(baseline_row["support_manager"]["baseline_only"])
        self.assertEqual(
            baseline_row["support_manager"]["activated_cell_indices"], []
        )

        baseline_survivors = baseline_state.survivors
        instrumented_survivors = instrumented_state.survivors
        self.assertIsNotNone(baseline_survivors)
        self.assertIsNotNone(instrumented_survivors)
        self.assertEqual(
            baseline_state.dt_seconds, instrumented_state.dt_seconds
        )
        self.assertEqual(
            baseline_survivors.requested_count,
            instrumented_survivors.requested_count,
        )
        for name in (
            "previous_points",
            "current_points",
            "forward_backward_errors",
        ):
            baseline = getattr(baseline_survivors, name)
            instrumented = getattr(instrumented_survivors, name)
            self.assertEqual(baseline.dtype, instrumented.dtype)
            self.assertEqual(baseline.shape, instrumented.shape)
            self.assertTrue(np.array_equal(baseline, instrumented))

        self.assertIs(
            localization.r3.compensate_current_to_previous,
            capture._original_compensate,
        )
        self.assertIs(
            localization.r3.detect_fixed_grid_features,
            capture._original_detect,
        )
        self.assertIs(
            localization.r3.track_features, capture._original_track
        )
        self.assertIs(
            localization.r3.fit_fixed_grid_local_affine,
            capture._original_fit,
        )
        self.assertIs(
            localization.r3.track_observable_points,
            capture._original_observable,
        )
        self.assertIs(
            localization.r3.merge_path_correspondences,
            capture._original_merge,
        )
        self.assertIs(
            localization.r0_evaluation._common_cell_expansions,
            capture._original_common,
        )
        self.assertEqual(localization.sha256_array(previous_rgb), frame_sha256)
        self.assertEqual(localization.sha256_array(current_rgb), frame_sha256)
        self.assertEqual(localization.sha256_array(valid), mask_sha256)


class PrimitivePackageMutationTests(unittest.TestCase):
    def _save(
        self, directory: Path, name: str, arrays: dict[str, np.ndarray]
    ) -> Path:
        path = directory / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        return path

    def test_minimal_package_is_valid_and_exact_keyset_is_required(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays = _primitive_arrays()
            loaded = validator.load_primitives(
                self._save(directory, "valid", arrays), pair_count=1
            )
            self.assertEqual(set(loaded), validator.NPZ_KEYS)

            missing = _copy_arrays(arrays)
            missing.pop("initial_source_world")
            with self.assertRaisesRegex(
                validator.InvalidExecution, "NPZ_KEYSET"
            ):
                validator.load_primitives(
                    self._save(directory, "missing", missing), pair_count=1
                )

            extra = _copy_arrays(arrays)
            extra["unexpected"] = np.asarray([1], dtype=np.int64)
            with self.assertRaisesRegex(
                validator.InvalidExecution, "NPZ_KEYSET"
            ):
                validator.load_primitives(
                    self._save(directory, "extra", extra), pair_count=1
                )

    def test_nonfinite_and_object_arrays_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nonfinite = _primitive_arrays()
            nonfinite["initial_source_depth_m"][0] = np.nan
            with self.assertRaisesRegex(
                validator.InvalidExecution,
                "NPZ_NONFINITE:initial_source_depth_m",
            ):
                validator.load_primitives(
                    self._save(directory, "nonfinite", nonfinite),
                    pair_count=1,
                )

            object_array = _primitive_arrays()
            object_array["initial_previous"] = np.asarray(
                [[10.0, 10.0], [20.0, 20.0]], dtype=object
            )
            with self.assertRaises(
                (validator.InvalidExecution, ValueError)
            ):
                validator.load_primitives(
                    self._save(directory, "object", object_array),
                    pair_count=1,
                )

    def test_offsets_indices_and_order_mutations_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            validator.InvalidExecution, "TEST_OFFSETS:MONOTONIC"
        ):
            validator._validate_offsets(
                np.asarray([0, 2, 1], dtype=np.int64),
                expected_groups=2,
                terminal_length=1,
                label="TEST_OFFSETS",
            )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_index = _primitive_arrays()
            source_index["initial_source_index"] = np.asarray(
                [1, 0], dtype=np.int32
            )
            with self.assertRaisesRegex(
                validator.InvalidExecution, "INITIAL_SOURCE_INDEX:0"
            ):
                validator.load_primitives(
                    self._save(directory, "source-index", source_index),
                    pair_count=1,
                )

            forward_index = _primitive_arrays()
            forward_index["lk_forward_source_index"] = np.asarray(
                [1, 0], dtype=np.int32
            )
            with self.assertRaisesRegex(
                validator.InvalidExecution, "LK_FORWARD_INDEX:0"
            ):
                validator.load_primitives(
                    self._save(directory, "forward-index", forward_index),
                    pair_count=1,
                )

            accepted_order = _primitive_arrays()
            accepted_order["accepted_previous"] = accepted_order[
                "accepted_previous"
            ][::-1].copy()
            with self.assertRaisesRegex(
                validator.InvalidExecution,
                "ACCEPTED_PREVIOUS_IDENTITY:0",
            ):
                validator.load_primitives(
                    self._save(
                        directory, "accepted-order", accepted_order
                    ),
                    pair_count=1,
                )

            fit_path = _primitive_arrays()
            fit_path["fit_group_path"][[0, 1]] = fit_path[
                "fit_group_path"
            ][[1, 0]]
            with self.assertRaisesRegex(
                validator.InvalidExecution, "FIT_GROUP_PATH"
            ):
                validator.load_primitives(
                    self._save(directory, "fit-path", fit_path),
                    pair_count=1,
                )

            fit_identity = _primitive_arrays()
            fit_identity["fit_current"][2, 0] += 1.0
            with self.assertRaisesRegex(
                validator.InvalidExecution,
                "COMPENSATED_INITIAL_ACCEPTED_IDENTITY:0",
            ):
                validator.load_primitives(
                    self._save(directory, "fit-identity", fit_identity),
                    pair_count=1,
                )

            activated_order = _primitive_arrays()
            activated_order["activated_offsets"] = np.asarray(
                [0, 2], dtype=np.int64
            )
            activated_order["activated_cell_index"] = np.asarray(
                [1, 0], dtype=np.int8
            )
            with self.assertRaisesRegex(
                validator.InvalidExecution, "ACTIVATED_CELL_ORDER:0"
            ):
                validator.load_primitives(
                    self._save(
                        directory, "activated-order", activated_order
                    ),
                    pair_count=1,
                )

            merge_path = _primitive_arrays()
            merge_path["merge_group_path"] = merge_path[
                "merge_group_path"
            ][::-1].copy()
            with self.assertRaisesRegex(
                validator.InvalidExecution, "MERGE_GROUP_PATH:0"
            ):
                validator.load_primitives(
                    self._save(directory, "merge-path", merge_path),
                    pair_count=1,
                )


class RoutingTests(unittest.TestCase):
    @staticmethod
    def _passing_metrics() -> dict[str, object]:
        return {
            "final": {
                "evaluable_pairs": 601,
                "absolute_p90_per_s": 0.02,
                "reproduced": True,
            },
            "geometry": {"status": "PASS"},
            "warp": {"status": "PASS"},
            "mask": {"status": "PASS"},
            "flow": {
                "accepted_status": "PASS",
                "managed_status": "PASS",
                "interior_status": "PASS",
            },
            "local": {"status": "FAIL"},
        }

    def test_route_precedence(self) -> None:
        metrics = self._passing_metrics()
        metrics["geometry"] = {"status": "FAIL"}
        self.assertEqual(
            localization.route_cluster(metrics),
            "LEAKAGE_ALREADY_PRESENT_IN_INPUT_GEOMETRY",
        )

        metrics = self._passing_metrics()
        metrics["warp"] = {"status": "FAIL"}
        self.assertEqual(
            localization.route_cluster(metrics),
            "LEAKAGE_FIRST_VISIBLE_AT_WARP",
        )
        metrics["flow"] = {
            "accepted_status": "FAIL",
            "managed_status": "PASS",
            "interior_status": "PASS",
        }
        self.assertEqual(
            localization.route_cluster(metrics),
            "MULTIPLE_SOURCES_NOT_SEPARABLE",
        )

        metrics = self._passing_metrics()
        metrics["mask"] = {"status": "BOUNDARY_ONLY_FAIL"}
        self.assertEqual(
            localization.route_cluster(metrics),
            "LEAKAGE_FIRST_VISIBLE_AT_MASK_BOUNDARY",
        )
        metrics["flow"] = {
            "accepted_status": "PASS",
            "managed_status": "PASS",
            "interior_status": "FAIL",
        }
        self.assertEqual(
            localization.route_cluster(metrics),
            "MULTIPLE_SOURCES_NOT_SEPARABLE",
        )

        metrics = self._passing_metrics()
        metrics["flow"] = {
            "accepted_status": "FAIL",
            "managed_status": "PASS",
            "interior_status": "PASS",
        }
        self.assertEqual(
            localization.route_cluster(metrics),
            "LEAKAGE_FIRST_VISIBLE_AT_FLOW",
        )

        metrics = self._passing_metrics()
        self.assertEqual(
            localization.route_cluster(metrics),
            "LEAKAGE_FIRST_VISIBLE_AT_LOCAL_FIT",
        )

    def test_missing_coverage_or_no_final_leakage_is_not_evaluable(self) -> None:
        metrics = self._passing_metrics()
        metrics["warp"] = {"status": "NOT_EVALUABLE"}
        self.assertEqual(localization.route_cluster(metrics), "NOT_EVALUABLE")

        metrics = self._passing_metrics()
        metrics["final"] = {
            "evaluable_pairs": 601,
            "absolute_p90_per_s": 0.01,
            "reproduced": True,
        }
        self.assertEqual(localization.route_cluster(metrics), "NOT_EVALUABLE")

        metrics = self._passing_metrics()
        metrics["final"] = {
            "evaluable_pairs": 450,
            "absolute_p90_per_s": 0.02,
            "reproduced": True,
        }
        self.assertEqual(localization.route_cluster(metrics), "NOT_EVALUABLE")

    def test_all_layers_pass_with_final_failure_is_not_scientific_multiple(self) -> None:
        metrics = self._passing_metrics()
        metrics["local"] = {"status": "PASS"}
        self.assertEqual(localization.route_cluster(metrics), "NOT_EVALUABLE")


class ReceiptSchemaTests(unittest.TestCase):
    @staticmethod
    def _implementation_ready_fixture(
        root: Path,
    ) -> tuple[Path, dict[str, object]]:
        paths = {
            "runner": root / "runner.py",
            "independent_execution_validator": (
                root / localization.INDEPENDENT_VALIDATOR_RELATIVE
            ),
            "test_suite": root / localization.TEST_SUITE_RELATIVE,
            "generator": root / localization.GENERATOR_GEOMETRY_RELATIVE,
            "renderer": root / localization.MATERIAL_RENDERER_RELATIVE,
            "numeric_amendment": (
                root
                / localization.SPEC_DIRECT_BINDING_PATHS[
                    "numeric_representation_amendment"
                ]
            ),
        }
        for role, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {role}\n", encoding="utf-8")
        frozen_paths = set(localization.SPEC_DIRECT_BINDING_PATHS.values())
        frozen_paths.update(localization.SPEC_FROZEN_DEPENDENCY_PATHS)
        for relative in sorted(frozen_paths):
            path = root / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"# frozen fixture: {relative}\n", encoding="utf-8"
                )
        implementation_bindings = {
            role: {
                "path": paths[role].relative_to(root).as_posix(),
                "sha256": localization.sha256_file(paths[role]),
            }
            for role in (
                "runner",
                "independent_execution_validator",
                "test_suite",
            )
        }
        spec = {
            "schema": localization.SPEC_SCHEMA,
            "protocol_id": localization.PROTOCOL_ID,
            "task_id": localization.TASK_ID,
            "status": localization.EXECUTABLE_SPEC_STATUS,
            "terminal": localization.SPEC_TERMINAL,
            "claim_ceiling": localization.SPEC_CLAIM_CEILING,
            "authority": {
                "separate_r1_activation_exists": True,
                "this_spec_adds_execution_authority": False,
                "formal_one_shot_may_start_from_this_spec_alone": False,
                "required_status_before_formal_launch": (
                    localization.SPEC_REQUIRED_STATUS
                ),
                "stage_b_retry_replacement_or_reseed": False,
                "r3_modification": False,
                "single_variable_repair": False,
                "formal_480_plus_16": False,
            },
            "implementation_bindings": {
                "status": localization.IMPLEMENTATION_BINDINGS_STATUS,
                **implementation_bindings,
                "finalization_rule": "Frozen fixture implementations.",
            },
            "bindings": {
                **{
                    name: {
                        "path": relative,
                        "sha256": localization.sha256_file(root / relative),
                    }
                    for name, relative in (
                        localization.SPEC_DIRECT_BINDING_PATHS.items()
                    )
                },
                "frozen_execution_dependencies": [
                    {
                        "path": relative,
                        "sha256": localization.sha256_file(root / relative),
                    }
                    for relative in localization.SPEC_FROZEN_DEPENDENCY_PATHS
                ],
            },
        }
        spec_path = root / "spec.json"
        _write_json(spec_path, spec)
        pilot_manifest_path = root / "pilots" / "manifest.json"
        pilot_disjoint_path = root / "pilots" / "disjoint.json"
        _write_json(pilot_manifest_path, {"fixture": "manifest"})
        _write_json(pilot_disjoint_path, {"fixture": "disjoint"})
        pilot_artifacts: dict[str, dict[str, str]] = {}
        for prefix in ("w1", "w4"):
            run_root = root / "pilots" / prefix
            run_receipt = run_root / "run_receipt.json"
            validation_receipt = (
                run_root / "independent_validation_receipt.json"
            )
            _write_json(run_receipt, {"fixture": f"{prefix}-run"})
            _write_json(
                validation_receipt,
                {"fixture": f"{prefix}-independent-validation"},
            )
            pilot_artifacts[prefix] = {
                "run": localization.sha256_file(run_receipt),
                "validation": localization.sha256_file(
                    validation_receipt
                ),
            }
        pilot_manifest_sha256 = localization.sha256_file(
            pilot_manifest_path
        )
        pilot_receipt = {
            "schema": localization.PILOT_EQUIVALENCE_SCHEMA,
            "protocol_id": localization.PROTOCOL_ID,
            "task_id": localization.TASK_ID,
            "valid": True,
            "terminal": localization.PILOT_EQUIVALENCE_TERMINAL,
            "cluster_count": localization.PILOT_CLUSTER_COUNT,
            "formal_authority_consumed": False,
            "scientific_interpretation": False,
            "runner_source_sha256": implementation_bindings["runner"][
                "sha256"
            ],
            "validator_source_sha256": implementation_bindings[
                "independent_execution_validator"
            ]["sha256"],
            "checks": {"semantic_equivalence": "PASS"},
            "w1_root": "pilots/w1",
            "w4_root": "pilots/w4",
            "pilot_manifest_path": "pilots/manifest.json",
            "semantic_payload_sha256": "s" * 64,
            "pilot_manifest_sha256": pilot_manifest_sha256,
        }
        pilot_path = root / "pilots" / "equivalence.json"
        _write_json(pilot_path, pilot_receipt)
        preformal_receipt = {
            "schema": localization.PREFORMAL_TEST_SCHEMA,
            "protocol_id": localization.PROTOCOL_ID,
            "task_id": localization.TASK_ID,
            "status": "PASS",
            "terminal": localization.PREFORMAL_TEST_TERMINAL,
            "test_count": 25,
            "failure_count": 0,
            "formal_authority_consumed": False,
            "formal_output_root_access": False,
            "sealed_input_access": False,
            "checks": {
                "unit_and_mutation_suite": "PASS",
                "observation_hook_transparency": "PASS",
                "postclaim_failure_closure": "PASS",
                "primitive_provenance_mutations": "PASS",
            },
            "implementation_bindings": implementation_bindings,
        }
        preformal_path = root / "pilots" / "preformal-tests.json"
        _write_json(preformal_path, preformal_receipt)
        host_path = root / "host.json"
        _write_json(host_path, {"fixture": "host-preflight"})
        spec["pilot_w1_w4_equivalence"] = {
            "frozen_evidence": {
                "pilot_manifest_path": "pilots/manifest.json",
                "pilot_manifest_sha256": pilot_manifest_sha256,
                "pilot_disjoint_receipt_path": "pilots/disjoint.json",
                "pilot_disjoint_receipt_sha256": (
                    localization.sha256_file(pilot_disjoint_path)
                ),
                "w1_root": "pilots/w1",
                "w1_run_receipt_sha256": pilot_artifacts["w1"]["run"],
                "w1_independent_validation_receipt_sha256": (
                    pilot_artifacts["w1"]["validation"]
                ),
                "w4_root": "pilots/w4",
                "w4_run_receipt_sha256": pilot_artifacts["w4"]["run"],
                "w4_independent_validation_receipt_sha256": (
                    pilot_artifacts["w4"]["validation"]
                ),
                "equivalence_receipt_path": "pilots/equivalence.json",
                "equivalence_receipt_sha256": (
                    localization.sha256_file(pilot_path)
                ),
                "semantic_payload_sha256": "s" * 64,
                "status": localization.PILOT_EQUIVALENCE_TERMINAL,
                "scientific_interpretation": False,
            }
        }
        spec["preformal_test_gate"] = {
            "frozen_evidence": {
                "receipt_path": "pilots/preformal-tests.json",
                "receipt_sha256": localization.sha256_file(preformal_path),
                "test_count": 25,
                "failure_count": 0,
                "terminal": localization.PREFORMAL_TEST_TERMINAL,
            }
        }
        spec["host_preflight_contract"] = {
            "frozen_evidence": {
                "receipt_path": "host.json",
                "receipt_sha256": localization.sha256_file(host_path),
                "status": "QUALIFIED",
                "workers": localization.WORKERS,
            }
        }
        _write_json(spec_path, spec)
        ready = {
            "schema": localization.IMPLEMENTATION_READY_SCHEMA,
            "protocol_id": localization.PROTOCOL_ID,
            "task_id": localization.TASK_ID,
            "status": "PASS",
            "terminal": localization.IMPLEMENTATION_READY_TERMINAL,
            "formal_authority_consumed": False,
            "scientific_interpretation": False,
            "spec_path": "spec.json",
            "spec_sha256": localization.sha256_file(spec_path),
            "implementation_bindings": implementation_bindings,
            "pilot_w1_w4_equivalence": {
                "status": "PASS",
                "workers": [1, localization.WORKERS],
                "receipt_path": "pilots/equivalence.json",
                "receipt_sha256": localization.sha256_file(pilot_path),
                "semantic_payload_sha256": "s" * 64,
                "pilot_manifest_sha256": pilot_manifest_sha256,
            },
            "preformal_test_gate": {
                "status": "PASS",
                "receipt_path": "pilots/preformal-tests.json",
                "receipt_sha256": localization.sha256_file(preformal_path),
                "test_count": 25,
                "test_suite_sha256": implementation_bindings["test_suite"][
                    "sha256"
                ],
            },
            "host_preflight": {
                "path": "host.json",
                "sha256": localization.sha256_file(host_path),
            },
        }
        ready_path = root / "ready.json"
        _write_json(ready_path, ready)
        return ready_path, ready

    def test_implementation_ready_exact_schema_positive_and_mutations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready_path, ready = self._implementation_ready_fixture(root)
            patches = (
                mock.patch.object(localization, "__file__", root / "runner.py"),
                mock.patch.object(localization, "SPEC_RELATIVE", "spec.json"),
                mock.patch.object(
                    localization, "PILOT_PARENT_RELATIVE", "pilots"
                ),
                mock.patch.object(
                    localization, "FORMAL_ROOT_RELATIVE", "formal"
                ),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                result = localization.validate_implementation_ready(
                    root, ready_path
                )
                self.assertEqual(
                    result["implementation_hashes"],
                    {
                        role: ready["implementation_bindings"][role][
                            "sha256"
                        ]
                        for role in (
                            "runner",
                            "independent_execution_validator",
                            "test_suite",
                        )
                    },
                )

                mutations = []
                wrong_schema = copy.deepcopy(ready)
                wrong_schema["schema"] = (
                    localization.IMPLEMENTATION_READY_SCHEMA + ".unexpected"
                )
                mutations.append(("schema", wrong_schema))

                extra_binding = copy.deepcopy(ready)
                extra_binding["implementation_bindings"]["unexpected"] = {
                    "path": "unexpected.py",
                    "sha256": "0" * 64,
                }
                mutations.append(("binding-keyset", extra_binding))

                failed_pilot_check = copy.deepcopy(ready)
                pilot_path = root / "pilots" / "equivalence.json"
                pilot_receipt = copy.deepcopy(
                    localization.load_json(pilot_path)
                )
                pilot_receipt["checks"]["semantic_equivalence"] = "FAIL"
                _write_json(pilot_path, pilot_receipt)
                failed_pilot_check["pilot_w1_w4_equivalence"][
                    "receipt_sha256"
                ] = localization.sha256_file(pilot_path)
                mutations.append(("pilot-check", failed_pilot_check))

                for label, mutated in mutations:
                    with self.subTest(label=label):
                        _write_json(ready_path, mutated)
                        with self.assertRaises(
                            localization.InvalidLocalization
                        ):
                            localization.validate_implementation_ready(
                                root, ready_path
                            )

    def test_every_frozen_spec_binding_is_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready_path, _ = self._implementation_ready_fixture(root)
            with (
                mock.patch.object(
                    localization, "__file__", root / "runner.py"
                ),
                mock.patch.object(
                    localization, "SPEC_RELATIVE", "spec.json"
                ),
                mock.patch.object(
                    localization, "PILOT_PARENT_RELATIVE", "pilots"
                ),
                mock.patch.object(
                    localization, "FORMAL_ROOT_RELATIVE", "formal"
                ),
            ):
                localization.validate_implementation_ready(root, ready_path)
                (
                    root
                    / localization.SPEC_DIRECT_BINDING_PATHS[
                        "numeric_representation_amendment"
                    ]
                ).write_text(
                    "binding drift\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    localization.InvalidLocalization,
                    "SPEC_BINDING:numeric_representation_amendment",
                ):
                    localization.validate_implementation_ready(
                        root, ready_path
                    )

    def test_preformal_test_gate_rejects_failed_checks_binding_drift_and_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready_path, base_ready = self._implementation_ready_fixture(root)
            test_path = root / "pilots" / "preformal-tests.json"
            base_test = localization.load_json(test_path)
            with (
                mock.patch.object(
                    localization, "__file__", root / "runner.py"
                ),
                mock.patch.object(
                    localization, "SPEC_RELATIVE", "spec.json"
                ),
                mock.patch.object(
                    localization, "PILOT_PARENT_RELATIVE", "pilots"
                ),
                mock.patch.object(
                    localization, "FORMAL_ROOT_RELATIVE", "formal"
                ),
            ):
                localization.validate_implementation_ready(root, ready_path)
                mutations = []

                failed_check = copy.deepcopy(base_test)
                failed_check["checks"][
                    "observation_hook_transparency"
                ] = "FAIL"
                mutations.append(
                    (
                        "failed-check",
                        failed_check,
                        copy.deepcopy(base_ready),
                        "PREFORMAL_TEST_RECEIPT_INVALID",
                    )
                )

                binding_drift = copy.deepcopy(base_test)
                binding_drift["implementation_bindings"]["unexpected"] = {
                    "path": "unexpected.py",
                    "sha256": "0" * 64,
                }
                mutations.append(
                    (
                        "binding-keyset",
                        binding_drift,
                        copy.deepcopy(base_ready),
                        "PREFORMAL_TEST_SOURCE_BINDING",
                    )
                )

                summary_drift = copy.deepcopy(base_ready)
                summary_drift["preformal_test_gate"]["test_count"] = 26
                mutations.append(
                    (
                        "summary",
                        copy.deepcopy(base_test),
                        summary_drift,
                        "PREFORMAL_TEST_SUMMARY_BINDING",
                    )
                )

                for label, test_receipt, ready, error in mutations:
                    with self.subTest(label=label):
                        _write_json(test_path, test_receipt)
                        ready["preformal_test_gate"][
                            "receipt_sha256"
                        ] = localization.sha256_file(test_path)
                        _write_json(ready_path, ready)
                        with self.assertRaisesRegex(
                            localization.InvalidLocalization, error
                        ):
                            localization.validate_implementation_ready(
                                root, ready_path
                            )

    def test_preclaim_rejects_spec_omission_governance_and_leaf_rebinding(
        self,
    ) -> None:
        cases = (
            "binding-omission",
            "authority-expansion",
            "finalization-rule",
            "pilot-substitution",
            "absolute-host-path",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                ready_path, ready = self._implementation_ready_fixture(root)
                spec_path = root / "spec.json"
                spec = localization.load_json(spec_path)
                if case == "binding-omission":
                    spec["bindings"].pop("memory_gate_amendment")
                    _write_json(spec_path, spec)
                    ready["spec_sha256"] = localization.sha256_file(
                        spec_path
                    )
                    expected = "SPEC_BINDING_KEYSET"
                elif case == "authority-expansion":
                    spec["authority"]["r3_modification"] = True
                    _write_json(spec_path, spec)
                    ready["spec_sha256"] = localization.sha256_file(
                        spec_path
                    )
                    expected = "EXECUTABLE_SPEC_GOVERNANCE"
                elif case == "finalization-rule":
                    spec["implementation_bindings"][
                        "finalization_rule"
                    ] = ""
                    _write_json(spec_path, spec)
                    ready["spec_sha256"] = localization.sha256_file(
                        spec_path
                    )
                    expected = "SPEC_IMPLEMENTATION_BINDINGS_STATUS"
                elif case == "pilot-substitution":
                    substitute = root / "pilots" / "substitute.json"
                    substitute.write_bytes(
                        (root / "pilots" / "equivalence.json").read_bytes()
                    )
                    ready["pilot_w1_w4_equivalence"].update(
                        {
                            "receipt_path": "pilots/substitute.json",
                            "receipt_sha256": localization.sha256_file(
                                substitute
                            ),
                        }
                    )
                    expected = "SPEC_PILOT_EVIDENCE_BINDING"
                else:
                    ready["host_preflight"]["path"] = str(
                        (root / "host.json").resolve()
                    )
                    expected = "BINDING_PATH:HOST_PREFLIGHT_READY"
                _write_json(ready_path, ready)
                with (
                    mock.patch.object(
                        localization, "__file__", root / "runner.py"
                    ),
                    mock.patch.object(
                        localization, "SPEC_RELATIVE", "spec.json"
                    ),
                    mock.patch.object(
                        localization, "PILOT_PARENT_RELATIVE", "pilots"
                    ),
                    mock.patch.object(
                        localization, "FORMAL_ROOT_RELATIVE", "formal"
                    ),
                ):
                    with self.assertRaisesRegex(
                        localization.InvalidLocalization, expected
                    ):
                        localization.validate_implementation_ready(
                            root, ready_path
                        )

    @staticmethod
    def _host_receipt(
        runner_path: Path,
        launcher_path: Path,
        authority: dict[str, str],
    ) -> dict[str, object]:
        return {
            "schema_version": localization.HOST_PREFLIGHT_SCHEMA,
            "task_id": localization.HOST_PREFLIGHT_TASK_ID,
            "execution_class": "formal",
            "implementation": {
                "script": "runner.py",
                "sha256": localization.sha256_file(runner_path),
            },
            "workload": {
                "class": "cpu_data_parallel",
                "real_data_mechanics_match": True,
                "input_identity": (
                    f"identity_lock:{authority['identity_sha256']}"
                ),
            },
            "pilot": {
                "representative_units": 32,
                "projected_full_units": 8 * localization.PAIR_COUNT,
                "same_access_mechanics": True,
                "output_equivalence": "PASS",
                "progress_samples": 2,
            },
            "scheduler": {
                "backend": "cpu_process_pool",
                "workers": localization.WORKERS,
                "comparison_performed": True,
                "scientific_parameters_unchanged": True,
                "inject_workers": True,
                "reserve_memory_gib": 6.0,
                "estimated_gib_per_worker": 0.25,
            },
            "progress": {
                "path": "formal/progress.json",
                "fields": [
                    "phase",
                    "completed_units",
                    "total_units",
                    "throughput",
                    "eta_seconds",
                    "last_progress_at",
                    "status",
                ],
                "verified_in_pilot": True,
                "update_interval_seconds": 30.0,
            },
            "terminal": {
                "success_path": "formal/success.json",
                "failure_path": "formal/failure.json",
            },
            "formal": {
                "one_shot": True,
                "claim_created_by_runner_only": True,
                "claim_path": "formal/claim.json",
                "output_path": "formal/success.json",
                "failure_receipt_path": "formal/failure.json",
                "activation_authority": (
                    "activation.json:"
                    f"{authority['activation_sha256']}"
                ),
            },
            "guarded_launcher": {
                "script": "guard.ps1",
                "sha256": localization.sha256_file(launcher_path),
            },
        }

    def test_host_preflight_exact_schema_positive_and_mutations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner_path = root / "runner.py"
            runner_path.write_text("# runner\n", encoding="utf-8")
            launcher_path = root / "guard.ps1"
            launcher_path.write_text("# guarded launcher\n", encoding="utf-8")
            host_path = root / "host.json"
            authority = {
                "identity_sha256": "i" * 64,
                "activation_sha256": "a" * 64,
            }
            valid = self._host_receipt(
                runner_path, launcher_path, authority
            )

            def validate(value: dict[str, object]) -> dict[str, object]:
                _write_json(host_path, value)
                implementation_ready = {
                    "host_preflight_binding": {
                        "path": "host.json",
                        "sha256": localization.sha256_file(host_path),
                    }
                }
                return localization.validate_host_preflight_receipt(
                    root,
                    host_path,
                    implementation_ready,
                    authority,
                )

            def validate_raw(payload: bytes) -> dict[str, object]:
                host_path.write_bytes(payload)
                implementation_ready = {
                    "host_preflight_binding": {
                        "path": "host.json",
                        "sha256": localization.sha256_file(host_path),
                    }
                }
                return localization.validate_host_preflight_receipt(
                    root,
                    host_path,
                    implementation_ready,
                    authority,
                )

            with (
                mock.patch.object(
                    localization, "__file__", root / "runner.py"
                ),
                mock.patch.object(
                    localization, "FORMAL_ROOT_RELATIVE", "formal"
                ),
                mock.patch.object(
                    localization, "ACTIVATION_RELATIVE", "activation.json"
                ),
                mock.patch.object(
                    localization,
                    "GUARDED_HOST_LAUNCHER_RELATIVE",
                    "guard.ps1",
                ),
            ):
                self.assertEqual(
                    validate(valid)["sha256"],
                    localization.sha256_file(host_path),
                )
                mutations = []
                wrong_schema = copy.deepcopy(valid)
                wrong_schema["schema_version"] = "unexpected.v2"
                mutations.append(("schema", wrong_schema))

                extra_implementation_key = copy.deepcopy(valid)
                extra_implementation_key["implementation"]["unexpected"] = (
                    True
                )
                mutations.append(
                    ("implementation-keyset", extra_implementation_key)
                )

                wrong_workers = copy.deepcopy(valid)
                wrong_workers["scheduler"]["workers"] = 1
                mutations.append(("workers", wrong_workers))

                stale_progress = copy.deepcopy(valid)
                stale_progress["progress"]["update_interval_seconds"] = 61.0
                mutations.append(("progress-interval", stale_progress))

                string_samples = copy.deepcopy(valid)
                string_samples["pilot"]["progress_samples"] = "2"
                mutations.append(("progress-samples-type", string_samples))

                boolean_memory = copy.deepcopy(valid)
                boolean_memory["scheduler"][
                    "estimated_gib_per_worker"
                ] = True
                mutations.append(("worker-memory-type", boolean_memory))

                invalid_interval = copy.deepcopy(valid)
                invalid_interval["progress"][
                    "update_interval_seconds"
                ] = "nan"
                mutations.append(("progress-interval-type", invalid_interval))

                invalid_fields = copy.deepcopy(valid)
                invalid_fields["progress"]["fields"] = [{}]
                mutations.append(("progress-fields-type", invalid_fields))

                for label, mutated in mutations:
                    with self.subTest(label=label):
                        with self.assertRaisesRegex(
                            localization.InvalidLocalization,
                            "HOST_PREFLIGHT_RECEIPT_INVALID",
                        ):
                            validate(mutated)

                canonical = localization.canonical_bytes(valid)
                duplicate = (
                    b'{"schema_version":"unexpected",'
                    + canonical[1:]
                )
                with self.assertRaisesRegex(
                    localization.InvalidLocalization,
                    "DUPLICATE_JSON_KEY:schema_version",
                ):
                    validate_raw(duplicate)
                nonfinite = b'{"unused":NaN,' + canonical[1:]
                with self.assertRaisesRegex(
                    localization.InvalidLocalization,
                    "NONFINITE_JSON_CONSTANT:NaN",
                ):
                    validate_raw(nonfinite)


class IndependentFormalBindingTests(unittest.TestCase):
    FIXTURE_ACTIVATION_BINDING_PATHS = {
        "FROZEN_CONTRACT": "contract.json",
        "IDENTITY_INPUT_LOCK": "identity.json",
        "INDEPENDENT_PREFLIGHT_RECEIPT": "preflight.json",
    }
    FIXTURE_CONTRACT_BINDING_PATHS = {
        "identity_input_lock": "identity.json",
        "r3_parameters": "protocol.json",
    }

    def test_repository_authority_accepts_frozen_identity_placeholder(
        self,
    ) -> None:
        authority = localization.validate_authority(
            localization.repo_root()
        )
        self.assertEqual(
            authority["identity_sha256"],
            localization.sha256_file(
                localization.repo_root() / localization.IDENTITY_RELATIVE
            ),
        )

    @staticmethod
    def _enter_validator_patches(
        stack: ExitStack, root: Path
    ) -> None:
        replacements = {
            "__file__": root / localization.INDEPENDENT_VALIDATOR_RELATIVE,
            "SPEC_RELATIVE": "spec.json",
            "RUNNER_RELATIVE": "runner.py",
            "PILOT_PARENT_RELATIVE": "pilots",
            "FORMAL_ROOT_RELATIVE": "formal",
            "ACTIVATION_RELATIVE": "activation.json",
            "CONTRACT_RELATIVE": "contract.json",
            "IDENTITY_RELATIVE": "identity.json",
            "PREFLIGHT_RECEIPT_RELATIVE": "preflight.json",
            "PROTOCOL_RELATIVE": "protocol.json",
            "GUARDED_HOST_LAUNCHER_RELATIVE": "guard.ps1",
            "ACTIVATION_BINDING_PATHS": (
                IndependentFormalBindingTests
                .FIXTURE_ACTIVATION_BINDING_PATHS
            ),
            "CONTRACT_FROZEN_BINDING_PATHS": (
                IndependentFormalBindingTests
                .FIXTURE_CONTRACT_BINDING_PATHS
            ),
        }
        for name, value in replacements.items():
            stack.enter_context(mock.patch.object(validator, name, value))

    @staticmethod
    def _refresh_chain(
        root: Path, state: dict[str, object]
    ) -> None:
        pilot_path = root / "pilots" / "equivalence.json"
        test_path = root / "pilots" / "preformal-tests.json"
        host_path = root / "host.json"
        spec_path = root / "spec.json"
        ready_path = root / "ready.json"
        contract_path = root / "contract.json"
        identity_path = root / "identity.json"
        preflight_path = root / "preflight.json"
        protocol_path = root / "protocol.json"
        activation_path = root / "activation.json"
        contract = validator.load_json(contract_path)
        contract["frozen_bindings"]["r3_parameters"]["sha256"] = (
            validator.sha256_file(protocol_path)
        )
        _write_json(contract_path, contract)
        activation = validator.load_json(activation_path)
        for record in activation["bindings"]:
            role = record["role"]
            relative = (
                IndependentFormalBindingTests
                .FIXTURE_ACTIVATION_BINDING_PATHS[role]
            )
            record["sha256"] = validator.sha256_file(root / relative)
        _write_json(activation_path, activation)
        state["activation"] = activation
        host = validator.load_json(host_path)
        host["workload"]["input_identity"] = (
            f"identity_lock:{validator.sha256_file(identity_path)}"
        )
        host["formal"]["activation_authority"] = (
            f"activation.json:{validator.sha256_file(activation_path)}"
        )
        _write_json(host_path, host)
        pilot_receipt = validator.load_json(pilot_path)
        test_receipt = validator.load_json(test_path)
        spec = validator.load_json(spec_path)
        spec_pilot = spec["pilot_w1_w4_equivalence"][
            "frozen_evidence"
        ]
        spec_pilot.update(
            {
                "equivalence_receipt_path": "pilots/equivalence.json",
                "equivalence_receipt_sha256": validator.sha256_file(
                    pilot_path
                ),
                "semantic_payload_sha256": pilot_receipt.get(
                    "semantic_payload_sha256"
                ),
                "pilot_manifest_path": pilot_receipt.get(
                    "pilot_manifest_path"
                ),
                "pilot_manifest_sha256": pilot_receipt.get(
                    "pilot_manifest_sha256"
                ),
                "w1_root": pilot_receipt.get("w1_root"),
                "w4_root": pilot_receipt.get("w4_root"),
            }
        )
        spec["preformal_test_gate"]["frozen_evidence"].update(
            {
                "receipt_path": "pilots/preformal-tests.json",
                "receipt_sha256": validator.sha256_file(test_path),
                "test_count": test_receipt.get("test_count"),
                "failure_count": test_receipt.get("failure_count"),
                "terminal": test_receipt.get("terminal"),
            }
        )
        spec["host_preflight_contract"]["frozen_evidence"].update(
            {
                "receipt_path": "host.json",
                "receipt_sha256": validator.sha256_file(host_path),
            }
        )
        _write_json(spec_path, spec)
        ready = validator.load_json(ready_path)
        ready["spec_sha256"] = validator.sha256_file(spec_path)
        ready["pilot_w1_w4_equivalence"].update(
            {
                "receipt_path": "pilots/equivalence.json",
                "receipt_sha256": validator.sha256_file(pilot_path),
                "semantic_payload_sha256": pilot_receipt.get(
                    "semantic_payload_sha256"
                ),
                "pilot_manifest_sha256": pilot_receipt.get(
                    "pilot_manifest_sha256"
                ),
            }
        )
        ready["preformal_test_gate"].update(
            {
                "receipt_path": "pilots/preformal-tests.json",
                "receipt_sha256": validator.sha256_file(test_path),
                "test_count": test_receipt.get("test_count"),
            }
        )
        ready["host_preflight"] = {
            "path": "host.json",
            "sha256": validator.sha256_file(host_path),
        }
        _write_json(ready_path, ready)
        claim = state["claim"]
        run = state["run"]
        assert isinstance(claim, dict)
        assert isinstance(run, dict)
        claim.update(
            {
                "executable_spec_path": "spec.json",
                "executable_spec_sha256": validator.sha256_file(spec_path),
                "implementation_ready_receipt_path": "ready.json",
                "implementation_ready_receipt_sha256": (
                    validator.sha256_file(ready_path)
                ),
                "host_preflight_receipt_path": "host.json",
                "host_preflight_receipt_sha256": validator.sha256_file(
                    host_path
                ),
                "activation_sha256": validator.sha256_file(
                    activation_path
                ),
                "contract_sha256": validator.sha256_file(contract_path),
                "identity_lock_sha256": validator.sha256_file(
                    identity_path
                ),
                "preflight_receipt_sha256": validator.sha256_file(
                    preflight_path
                ),
                "protocol_sha256": validator.sha256_file(protocol_path),
            }
        )
        run["identity_lock_sha256"] = claim["identity_lock_sha256"]
        run.update(
            {
                "executable_spec_sha256": claim[
                    "executable_spec_sha256"
                ],
                "implementation_ready_receipt_sha256": claim[
                    "implementation_ready_receipt_sha256"
                ],
                "host_preflight_receipt_sha256": claim[
                    "host_preflight_receipt_sha256"
                ],
            }
        )
        claim_path = root / "formal" / "claim.json"
        _write_json(claim_path, claim)
        run["claim_sha256"] = validator.sha256_file(claim_path)

    @classmethod
    def _fixture(cls, root: Path) -> dict[str, object]:
        ready_path, _ = ReceiptSchemaTests._implementation_ready_fixture(
            root
        )
        identity = {"protocol_id": validator.PROTOCOL_ID}
        _write_json(root / "identity.json", identity)
        protocol = {
            "sparse_lk": {
                "grid_rows": 3,
                "grid_cols": 3,
                "max_features_per_cell": 80,
                "quality_level": 0.01,
                "min_distance_pixels": 5.0,
                "block_size": 7,
                "window_size": [21, 21],
                "max_pyramid_level": 3,
                "termination_count": 30,
                "termination_epsilon": 0.01,
                "forward_backward_max_error_pixels": 1.0,
            },
            "local_affine": {
                "grid_rows": 3,
                "grid_cols": 3,
                "minimum_tracks_per_cell": 12,
                "minimum_track_convex_hull_fraction": 0.1,
                "maximum_design_condition_number": 1000.0,
                "maximum_median_fit_residual_pixels_per_frame": 0.75,
                "minimum_common_evaluable_cells_per_pair": 5,
            },
            "metrics": {
                "sign_accuracy_zero_band_per_s": (
                    validator.LEAKAGE_THRESHOLD_PER_S
                )
            },
        }
        _write_json(root / "protocol.json", protocol)
        contract = {
            "protocol_id": validator.PROTOCOL_ID,
            "task_id": validator.TASK_ID,
            "unchanged_r3_parameters": {
                "sparse_lk": copy.deepcopy(
                    validator.EXPECTED_LK_CONTRACT
                ),
                "local_affine": copy.deepcopy(
                    validator.EXPECTED_LOCAL_AFFINE_CONTRACT
                ),
            },
            "frozen_bindings": {
                "identity_input_lock": {
                    "path": "identity.json",
                    "sha256_filled_by_validator_receipt": True,
                },
                "r3_parameters": {
                    "path": "protocol.json",
                    "sha256": validator.sha256_file(
                        root / "protocol.json"
                    ),
                },
            },
        }
        _write_json(root / "contract.json", contract)
        preflight = {
            "task_id": validator.TASK_ID,
            "protocol_status": "VALID",
            "check_count": 10,
            "checks": {
                f"check_{index:02d}": "PASS"
                for index in range(1, 11)
            },
        }
        _write_json(root / "preflight.json", preflight)
        activation = {
            "decision": (
                "AUTHORIZE_ROTATION_LEAKAGE_LOCALIZATION_ONE_SHOT_EXECUTION"
            ),
            "execution_authorized": True,
            "r3_modification_authorized": False,
            "stage_b_rerun_authorized": False,
            "single_variable_repair_authorized": False,
            "bindings": [
                {
                    "role": "FROZEN_CONTRACT",
                    "path": "contract.json",
                    "sha256": validator.sha256_file(
                        root / "contract.json"
                    ),
                },
                {
                    "role": "IDENTITY_INPUT_LOCK",
                    "path": "identity.json",
                    "sha256": validator.sha256_file(
                        root / "identity.json"
                    ),
                },
                {
                    "role": "INDEPENDENT_PREFLIGHT_RECEIPT",
                    "path": "preflight.json",
                    "sha256": validator.sha256_file(
                        root / "preflight.json"
                    ),
                },
            ],
            "execution": {
                "cluster_count": validator.CLUSTER_COUNT,
                "fixed_pairs_per_cluster": validator.PAIR_COUNT,
                "workers": validator.WORKERS,
                "one_shot": True,
                "write_once_output": True,
                "source_localization_workload_run": False,
                "output_root": "formal",
            },
            "resource_gate": {
                "in_flight_emergency_floor_bytes": (
                    validator.IN_FLIGHT_FLOOR_BYTES
                ),
                "launch_and_refill_minimum_available_ram_bytes": (
                    validator.LAUNCH_REFILL_BYTES
                ),
                "launch_check_required": True,
                "sustained_paging_stop": True,
            },
        }
        _write_json(root / "activation.json", activation)
        guard_path = root / "guard.ps1"
        guard_path.write_text("# guarded launcher\n", encoding="utf-8")
        authority = {
            "identity_sha256": validator.sha256_file(
                root / "identity.json"
            ),
            "activation_sha256": validator.sha256_file(
                root / "activation.json"
            ),
        }
        host = ReceiptSchemaTests._host_receipt(
            root / "runner.py", guard_path, authority
        )
        _write_json(root / "host.json", host)
        claim = {
            "schema": "rcle.r3_rotation_leakage_localization.claim.v1",
            "protocol_id": validator.PROTOCOL_ID,
            "task_id": validator.TASK_ID,
            "runner_id": validator.RUNNER_ID,
            "claim_id": "1" * 32,
            "claimed_utc": "2026-07-29T00:00:00Z",
            "runner_source_path": "runner.py",
            "runner_source_sha256": validator.sha256_file(
                root / "runner.py"
            ),
            "activation_sha256": authority["activation_sha256"],
            "contract_sha256": validator.sha256_file(
                root / "contract.json"
            ),
            "identity_lock_sha256": authority["identity_sha256"],
            "preflight_receipt_sha256": validator.sha256_file(
                root / "preflight.json"
            ),
            "protocol_sha256": validator.sha256_file(
                root / "protocol.json"
            ),
            "mode": "FORMAL",
            "workers": validator.WORKERS,
            "cluster_count": validator.CLUSTER_COUNT,
            "write_once": True,
            "formal_authority_consumed": True,
            "future_repair_authorized": False,
            "terminal": "EXECUTION_CLAIMED / NO_RETRY",
        }
        run = {
            "schema": "rcle.r3_rotation_leakage_localization.run.v1",
            "protocol_id": validator.PROTOCOL_ID,
            "task_id": validator.TASK_ID,
            "runner_id": validator.RUNNER_ID,
            "mode": "FORMAL",
            "identity_lock_sha256": authority["identity_sha256"],
            "formal_authority_consumed": True,
            "terminal": (
                "LOCALIZATION_EXECUTION_COMPLETE / "
                "INDEPENDENT_VALIDATION_REQUIRED"
            ),
        }
        (root / "formal").mkdir()
        state: dict[str, object] = {
            "ready_path": ready_path,
            "activation": activation,
            "claim": claim,
            "run": run,
        }
        cls._refresh_chain(root, state)
        return state

    def test_rehashed_formal_leaf_and_spec_mutations_are_rejected(
        self,
    ) -> None:
        cases = {
            "pilot-schema": "READY_PILOT_RECEIPT",
            "pilot-check": "READY_PILOT_RECEIPT",
            "test-sealed-access": "READY_TEST_RECEIPT",
            "test-check": "READY_TEST_RECEIPT",
            "host-workers": "HOST_PREFLIGHT_SEMANTICS",
            "host-progress-interval": "HOST_PREFLIGHT_SEMANTICS",
            "host-one-shot": "HOST_PREFLIGHT_SEMANTICS",
            "spec-binding-omission": "SPEC_BINDING_KEYSET",
            "spec-authority-expansion": "SPEC_GOVERNANCE",
            "absolute-ready-path": "CLAIM_READY:ABSOLUTE_PATH",
        }
        for case, expected in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                state = self._fixture(root)
                if case.startswith("pilot-"):
                    path = root / "pilots" / "equivalence.json"
                    value = validator.load_json(path)
                    if case == "pilot-schema":
                        value["schema"] = "unexpected"
                    else:
                        value["checks"]["semantic_equivalence"] = "FAIL"
                    _write_json(path, value)
                elif case.startswith("test-"):
                    path = root / "pilots" / "preformal-tests.json"
                    value = validator.load_json(path)
                    if case == "test-sealed-access":
                        value["sealed_input_access"] = True
                    else:
                        value["checks"][
                            "observation_hook_transparency"
                        ] = "FAIL"
                    _write_json(path, value)
                elif case.startswith("host-"):
                    path = root / "host.json"
                    value = validator.load_json(path)
                    if case == "host-workers":
                        value["scheduler"]["workers"] = 1
                    elif case == "host-progress-interval":
                        value["progress"][
                            "update_interval_seconds"
                        ] = 61.0
                    else:
                        value["formal"]["one_shot"] = False
                    _write_json(path, value)
                elif case.startswith("spec-"):
                    path = root / "spec.json"
                    value = validator.load_json(path)
                    if case == "spec-binding-omission":
                        value["bindings"].pop("memory_gate_amendment")
                    else:
                        value["authority"]["r3_modification"] = True
                    _write_json(path, value)
                self._refresh_chain(root, state)
                claim = state["claim"]
                run = state["run"]
                activation = state["activation"]
                assert isinstance(claim, dict)
                assert isinstance(run, dict)
                assert isinstance(activation, dict)
                if case == "absolute-ready-path":
                    claim["implementation_ready_receipt_path"] = str(
                        (root / "ready.json").resolve()
                    )
                with ExitStack() as stack:
                    self._enter_validator_patches(stack, root)
                    with self.assertRaisesRegex(
                        validator.InvalidExecution, expected
                    ):
                        validator._verify_formal_readiness_bindings(
                            root, claim, run, activation
                        )

    def test_claim_and_run_semantic_mutations_are_rejected(self) -> None:
        cases = {
            "claim-schema": "CLAIM_IDENTITY",
            "claim-preflight": "CLAIM_BINDING:preflight_receipt_sha256",
            "run-mode": "FORMAL_MODE",
            "run-identity": "FORMAL_RUN_LIFECYCLE",
            "run-authority": "FORMAL_RUN_LIFECYCLE",
            "run-terminal": "FORMAL_RUN_LIFECYCLE",
            "activation-authority": "ACTIVATION_AUTHORITY",
            "activation-workers": "ACTIVATION_EXECUTION_LOCK",
            "activation-resource": "ACTIVATION_RESOURCE_LOCK",
            "preflight-task": "PREFLIGHT_IDENTITY",
            "protocol-lk": "R3_LK_PARAMETER_DRIFT",
            "protocol-threshold": "R3_THRESHOLD_DRIFT",
            "contract-parameters": "R3_CONTRACT_PARAMETER_DRIFT",
            "contract-placeholder": (
                "CONTRACT_BINDING_PLACEHOLDER:identity_input_lock"
            ),
        }
        for case, expected in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                state = self._fixture(root)
                claim = state["claim"]
                run = state["run"]
                assert isinstance(claim, dict)
                assert isinstance(run, dict)
                if case == "claim-schema":
                    claim["schema"] = "unexpected"
                elif case == "claim-preflight":
                    claim["preflight_receipt_sha256"] = "0" * 64
                elif case == "run-mode":
                    run["mode"] = "PILOT"
                elif case == "run-identity":
                    run["identity_lock_sha256"] = "0" * 64
                elif case == "run-authority":
                    run["formal_authority_consumed"] = False
                elif case == "run-terminal":
                    run["terminal"] = "unexpected"
                elif case.startswith("activation-"):
                    path = root / "activation.json"
                    value = validator.load_json(path)
                    if case == "activation-authority":
                        value["single_variable_repair_authorized"] = True
                    elif case == "activation-workers":
                        value["execution"]["workers"] = 1
                    else:
                        value["resource_gate"][
                            "launch_check_required"
                        ] = False
                    _write_json(path, value)
                    self._refresh_chain(root, state)
                elif case == "preflight-task":
                    path = root / "preflight.json"
                    value = validator.load_json(path)
                    value["task_id"] = "unexpected"
                    _write_json(path, value)
                    self._refresh_chain(root, state)
                elif case.startswith("protocol-"):
                    path = root / "protocol.json"
                    value = validator.load_json(path)
                    if case == "protocol-lk":
                        value["sparse_lk"]["grid_rows"] = 999
                    else:
                        value["metrics"][
                            "sign_accuracy_zero_band_per_s"
                        ] = 999.0
                    _write_json(path, value)
                    self._refresh_chain(root, state)
                elif case.startswith("contract-"):
                    path = root / "contract.json"
                    value = validator.load_json(path)
                    if case == "contract-parameters":
                        value["unchanged_r3_parameters"]["sparse_lk"][
                            "grid"
                        ] = "999x3"
                    else:
                        value["frozen_bindings"]["identity_input_lock"] = {
                            "path": "identity.json",
                            "sha256": validator.sha256_file(
                                root / "identity.json"
                            ),
                        }
                    _write_json(path, value)
                    self._refresh_chain(root, state)
                if case.startswith("claim-"):
                    claim_path = root / "formal" / "claim.json"
                    _write_json(claim_path, claim)
                    run["claim_sha256"] = validator.sha256_file(
                        claim_path
                    )
                with ExitStack() as stack:
                    self._enter_validator_patches(stack, root)
                    with self.assertRaisesRegex(
                        validator.InvalidExecution, expected
                    ):
                        validator._verify_control_bindings(
                            root,
                            root / "formal",
                            root / "formal",
                            run,
                            fixture=False,
                        )


class PilotEquivalenceHelperTests(unittest.TestCase):
    def test_operational_receipt_fields_are_stripped_but_science_is_kept(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left = {
                "cluster_id": "PILOT_ONLY_CLUSTER_000",
                "route": "NOT_EVALUABLE",
                "wall_seconds": 1.0,
                "minimum_available_ram_bytes": 8,
                "pair_ledger_sha256": "a" * 64,
                "primitives_sha256": "b" * 64,
                "cluster_metrics_sha256": "c" * 64,
            }
            right = {
                **left,
                "wall_seconds": 99.0,
                "minimum_available_ram_bytes": 4,
                "pair_ledger_sha256": "d" * 64,
                "primitives_sha256": "e" * 64,
                "cluster_metrics_sha256": "f" * 64,
            }
            left_path = directory / "left.json"
            right_path = directory / "right.json"
            left_path.write_bytes(validator.canonical_bytes(left))
            right_path.write_bytes(validator.canonical_bytes(right))
            self.assertEqual(
                validator._stripped_cluster_receipt(left_path),
                validator._stripped_cluster_receipt(right_path),
            )

            right["route"] = "LEAKAGE_FIRST_VISIBLE_AT_FLOW"
            right_path.write_bytes(validator.canonical_bytes(right))
            self.assertNotEqual(
                validator._stripped_cluster_receipt(left_path),
                validator._stripped_cluster_receipt(right_path),
            )

    def test_npz_semantic_digest_ignores_archive_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays = _primitive_arrays()
            compressed = directory / "compressed.npz"
            stored = directory / "stored.npz"
            np.savez_compressed(compressed, **arrays)
            np.savez(stored, **arrays)
            self.assertNotEqual(
                validator.sha256_file(compressed),
                validator.sha256_file(stored),
            )
            self.assertEqual(
                validator._npz_semantic_digest(compressed),
                validator._npz_semantic_digest(stored),
            )

    def test_compare_pilots_accepts_only_equal_semantic_payloads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validator_path = root / "validator.py"
            validator_path.write_text("# validator\n", encoding="utf-8")
            manifest = root / "pilots" / "manifest.json"
            _write_json(manifest, {"fixture": "disjoint"})
            w1 = root / "pilots" / "w1"
            w4 = root / "pilots" / "w4"
            w1.mkdir()
            w4.mkdir()
            _write_json(
                w1 / "run_receipt.json",
                {"mode": "PILOT", "resource": {"workers": 1}},
            )
            _write_json(
                w4 / "run_receipt.json",
                {
                    "mode": "PILOT",
                    "resource": {"workers": localization.WORKERS},
                },
            )
            manifest_hash = validator.sha256_file(manifest)
            semantic_payload = {
                "mode": "PILOT",
                "cluster_count": 4,
                "route_status": "PENDING_INDEPENDENT_VALIDATION",
                "pilot_manifest_sha256": manifest_hash,
                "pilot_disjoint_receipt_sha256": "d" * 64,
                "runner_source_path": "runner.py",
                "runner_source_sha256": "r" * 64,
                "source_bindings": [],
                "clusters": [],
            }
            analysis = {"clusters": [], "route_counts": {}}

            common_patches = (
                mock.patch.object(validator, "repo_root", return_value=root),
                mock.patch.object(
                    validator, "__file__", str(validator_path)
                ),
                mock.patch.object(
                    validator,
                    "_resolve_response_root",
                    side_effect=lambda value: value,
                ),
                mock.patch.object(
                    validator,
                    "validate_execution",
                    return_value=(analysis, {}, {}),
                ),
            )
            output = root / "pilots" / "equivalence.json"
            with common_patches[0], common_patches[1], common_patches[
                2
            ], common_patches[3], mock.patch.object(
                validator,
                "_pilot_semantic_payload",
                return_value=semantic_payload,
            ):
                receipt = validator.compare_pilots(
                    w1, w4, manifest, output
                )
            self.assertTrue(receipt["valid"])
            self.assertEqual(
                receipt["semantic_payload_sha256"],
                hashlib.sha256(
                    validator.canonical_bytes(semantic_payload)
                ).hexdigest(),
            )
            self.assertTrue(output.is_file())

            mismatch_output = root / "pilots" / "mismatch.json"
            mismatched = copy.deepcopy(semantic_payload)
            mismatched["clusters"] = [{"cluster_id": "mutated"}]
            common_patches = (
                mock.patch.object(validator, "repo_root", return_value=root),
                mock.patch.object(
                    validator, "__file__", str(validator_path)
                ),
                mock.patch.object(
                    validator,
                    "_resolve_response_root",
                    side_effect=lambda value: value,
                ),
                mock.patch.object(
                    validator,
                    "validate_execution",
                    return_value=(analysis, {}, {}),
                ),
            )
            with common_patches[0], common_patches[1], common_patches[
                2
            ], common_patches[3], mock.patch.object(
                validator,
                "_pilot_semantic_payload",
                side_effect=[semantic_payload, mismatched],
            ):
                with self.assertRaisesRegex(
                    validator.InvalidExecution,
                    "PILOT_EQUIVALENCE_SEMANTIC_PAYLOAD",
                ):
                    validator.compare_pilots(
                        w1, w4, manifest, mismatch_output
                    )
            self.assertFalse(mismatch_output.exists())


class ValidatorLifecycleTests(unittest.TestCase):
    def test_progress_samples_and_owned_failure_close_truthfully(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "formal"
            formal.mkdir()
            validator_path = root / "validator.py"
            validator_path.write_text(
                "# validator lifecycle fixture\n", encoding="utf-8"
            )
            started = time.perf_counter()
            validator.write_validation_progress(
                formal,
                validation_claim_id="1" * 32,
                completed_units=0,
                total_units=4,
                started=started,
                status="running",
                initial=True,
            )
            with self.assertRaises(FileExistsError):
                validator.write_validation_progress(
                    formal,
                    validation_claim_id="2" * 32,
                    completed_units=0,
                    total_units=4,
                    started=started,
                    status="running",
                    initial=True,
                )
            validator.write_validation_progress(
                formal,
                validation_claim_id="1" * 32,
                completed_units=1,
                total_units=4,
                started=started,
                status="running",
            )
            progress = validator.load_json(
                formal / "validation_progress.json", canonical=True
            )
            self.assertEqual(progress["sample_index"], 1)
            self.assertEqual(progress["completed_units"], 1)
            self.assertEqual(progress["total_units"], 4)

            with (
                mock.patch.object(validator, "repo_root", return_value=root),
                mock.patch.object(
                    validator, "FORMAL_ROOT_RELATIVE", "formal"
                ),
                mock.patch.object(
                    validator, "__file__", str(validator_path)
                ),
            ):
                validator.ACTIVE_VALIDATION_CLAIMS[formal.resolve()] = (
                    "1" * 32
                )
                validator.write_formal_validation_failure(
                    formal, RuntimeError("VALIDATION_SENTINEL")
                )

            failure = validator.load_json(
                formal / "validation_failure.json", canonical=True
            )
            progress = validator.load_json(
                formal / "validation_progress.json", canonical=True
            )
            self.assertEqual(
                failure["terminal"],
                "INDEPENDENT_VALIDATION_INVALID / "
                "ONE_SHOT_CONSUMED / NO_RERUN",
            )
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["completed_units"], 1)
            self.assertEqual(progress["total_units"], 4)
            self.assertNotIn(
                formal.resolve(), validator.ACTIVE_VALIDATION_CLAIMS
            )

    def test_unowned_validation_cannot_write_formal_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "formal"
            formal.mkdir()
            validator_path = root / "validator.py"
            validator_path.write_text(
                "# validator ownership fixture\n", encoding="utf-8"
            )
            with (
                mock.patch.object(validator, "repo_root", return_value=root),
                mock.patch.object(
                    validator, "FORMAL_ROOT_RELATIVE", "formal"
                ),
                mock.patch.object(
                    validator, "__file__", str(validator_path)
                ),
            ):
                validator.write_formal_validation_failure(
                    formal, RuntimeError("UNOWNED_SENTINEL")
                )
            self.assertFalse(
                (formal / "validation_failure.json").exists()
            )

    def test_validation_collision_loser_cannot_close_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "formal"
            formal.mkdir()
            validator_path = root / "validator.py"
            validator_path.write_text(
                "# validator ownership token fixture\n", encoding="utf-8"
            )
            validator.write_validation_progress(
                formal,
                validation_claim_id="1" * 32,
                completed_units=0,
                total_units=4,
                started=time.perf_counter(),
                status="running",
                initial=True,
            )
            with (
                mock.patch.object(validator, "repo_root", return_value=root),
                mock.patch.object(
                    validator, "FORMAL_ROOT_RELATIVE", "formal"
                ),
                mock.patch.object(
                    validator, "__file__", str(validator_path)
                ),
            ):
                validator.ACTIVE_VALIDATION_CLAIMS[formal.resolve()] = (
                    "2" * 32
                )
                validator.write_formal_validation_failure(
                    formal, KeyboardInterrupt("LOSER_INTERRUPT")
                )
            self.assertFalse(
                (formal / "validation_failure.json").exists()
            )
            progress = validator.load_json(
                formal / "validation_progress.json", canonical=True
            )
            self.assertEqual(progress["status"], "running")
            self.assertEqual(progress["validation_claim_id"], "1" * 32)
            self.assertNotIn(
                formal.resolve(), validator.ACTIVE_VALIDATION_CLAIMS
            )


class FormalSafetyTests(unittest.TestCase):
    def test_runner_direct_script_entry_matches_guarded_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(localization.__file__).resolve()),
                    "--help",
                ],
                cwd=temporary,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("execute-formal", completed.stdout)

    def test_resource_gate(self) -> None:
        gib = 1024**3
        self.assertEqual(
            localization.resource_action(
                6 * gib,
                active_workers=0,
                refill_requested=True,
                paging_streak=0,
            ),
            "ALLOW",
        )
        self.assertEqual(
            localization.resource_action(
                6 * gib - 1,
                active_workers=0,
                refill_requested=True,
                paging_streak=0,
            ),
            "STOP_BEFORE_CLAIM",
        )
        self.assertEqual(
            localization.resource_action(
                4 * gib - 1,
                active_workers=4,
                refill_requested=False,
                paging_streak=0,
            ),
            "STOP_IN_FLIGHT",
        )
        self.assertEqual(
            localization.resource_action(
                8 * gib,
                active_workers=4,
                refill_requested=False,
                paging_streak=3,
            ),
            "STOP_IN_FLIGHT",
        )

    def test_final_rehash_barrier_rejects_stale_claim_before_claiming(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner_path = root / "runner.py"
            runner_path.write_text("# runner\n", encoding="utf-8")
            ready_path = root / "ready.json"
            ready_path.write_text("{}\n", encoding="utf-8")
            host_path = root / "host.json"
            host_path.write_text("{}\n", encoding="utf-8")
            output_root = root / "formal"
            authority = {
                "activation_sha256": "a" * 64,
                "contract_sha256": "c" * 64,
                "identity_sha256": "i" * 64,
                "preflight_receipt_sha256": "p" * 64,
                "protocol_sha256": "q" * 64,
            }
            readiness = {
                "source_bindings": [{"path": "source", "sha256": "s" * 64}],
                "spec_sha256": "e" * 64,
                "implementation_ready_receipt_path": ready_path,
                "implementation_ready_receipt_sha256": "r" * 64,
            }
            host = {"path": host_path, "sha256": "h" * 64}
            claim = {
                "runner_source_path": "runner.py",
                "runner_source_sha256": localization.sha256_file(
                    runner_path
                ),
                "activation_sha256": authority["activation_sha256"],
                "contract_sha256": authority["contract_sha256"],
                "identity_lock_sha256": authority["identity_sha256"],
                "preflight_receipt_sha256": authority[
                    "preflight_receipt_sha256"
                ],
                "protocol_sha256": authority["protocol_sha256"],
                "source_bindings": readiness["source_bindings"],
                "executable_spec_path": "spec.json",
                "executable_spec_sha256": readiness["spec_sha256"],
                "implementation_ready_receipt_path": "ready.json",
                "implementation_ready_receipt_sha256": readiness[
                    "implementation_ready_receipt_sha256"
                ],
                "host_preflight_receipt_path": "host.json",
                "host_preflight_receipt_sha256": host["sha256"],
            }
            with (
                mock.patch.object(localization, "__file__", runner_path),
                mock.patch.object(
                    localization, "SPEC_RELATIVE", "spec.json"
                ),
                mock.patch.object(
                    localization,
                    "validate_authority",
                    return_value=authority,
                ),
                mock.patch.object(
                    localization,
                    "validate_implementation_ready",
                    return_value=readiness,
                ),
                mock.patch.object(
                    localization,
                    "validate_host_preflight_receipt",
                    return_value=host,
                ),
                mock.patch.object(
                    localization, "validate_formal_target_absent"
                ) as target_check,
                mock.patch.object(
                    localization.psutil,
                    "virtual_memory",
                    return_value=SimpleNamespace(
                        available=8 * localization.GIB
                    ),
                ),
            ):
                localization._formal_claim_rehash_barrier(
                    root=root,
                    output_root=output_root,
                    claim=claim,
                    implementation_ready_receipt_path=ready_path,
                    host_preflight_receipt_path=host_path,
                )
                target_check.assert_called_once_with(root, output_root)
                stale = copy.deepcopy(claim)
                stale["activation_sha256"] = "0" * 64
                with self.assertRaisesRegex(
                    localization.InvalidLocalization,
                    "FINAL_REHASH_BARRIER:activation_sha256",
                ):
                    localization._formal_claim_rehash_barrier(
                        root=root,
                        output_root=output_root,
                        claim=stale,
                        implementation_ready_receipt_path=ready_path,
                        host_preflight_receipt_path=host_path,
                    )

    def test_formal_target_collision_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "formal"
            with (
                mock.patch.object(
                    localization,
                    "FORMAL_ROOT_RELATIVE",
                    "formal",
                ),
                mock.patch.object(
                    localization,
                    "SUCCESSOR_FORMAL_RELATIVE",
                    "successor-formal",
                ),
            ):
                localization.validate_formal_target_absent(root, target)
                target.mkdir()
                with self.assertRaises(ValueError):
                    localization.validate_formal_target_absent(root, target)
                with self.assertRaises(ValueError):
                    localization.validate_formal_target_absent(
                        root, root / "wrong-formal"
                    )

    def test_postclaim_setup_failure_closes_claim_with_failure_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot_parent = root / "pilots"
            pilot_parent.mkdir()
            output = pilot_parent / "postclaim-failure"
            runner_path = root / "runner.py"
            runner_path.write_text("# synthetic runner binding\n", encoding="utf-8")
            manifest_path = pilot_parent / "manifest.json"
            _write_json(manifest_path, {"fixture": "disjoint"})
            disjoint_path = pilot_parent / "disjoint.json"
            _write_json(disjoint_path, {"status": "PASS"})
            authority = {
                "activation_sha256": "a" * 64,
                "contract_sha256": "c" * 64,
                "identity_sha256": "i" * 64,
                "preflight_receipt_sha256": "p" * 64,
                "protocol_sha256": "q" * 64,
                "source_bindings": [],
            }
            original_mkdir = Path.mkdir

            def fail_staging(
                path: Path, *args: object, **kwargs: object
            ) -> None:
                if path == output / "staging":
                    raise RuntimeError("POSTCLAIM_SETUP_SENTINEL")
                original_mkdir(path, *args, **kwargs)

            with (
                mock.patch.object(
                    localization, "repo_root", return_value=root
                ),
                mock.patch.object(
                    localization, "__file__", runner_path
                ),
                mock.patch.object(
                    localization, "PILOT_PARENT_RELATIVE", "pilots"
                ),
                mock.patch.object(
                    localization, "FORMAL_ROOT_RELATIVE", "formal"
                ),
                mock.patch.object(
                    localization,
                    "validate_authority",
                    return_value=authority,
                ),
                mock.patch.object(
                    localization,
                    "validate_pilot_disjoint_receipt",
                    return_value={
                        "sha256": "d" * 64,
                        "receipt": {"status": "PASS"},
                    },
                ),
                mock.patch.object(
                    localization, "load_json", return_value={}
                ),
                mock.patch.object(
                    localization,
                    "pilot_tasks",
                    return_value=[
                        {
                            "cluster_id": "PILOT_ONLY_CLUSTER_000",
                            "pair_count": 1,
                        }
                    ],
                ),
                mock.patch.object(
                    localization.psutil,
                    "virtual_memory",
                    return_value=SimpleNamespace(
                        available=8 * localization.GIB
                    ),
                ),
                mock.patch.object(Path, "mkdir", new=fail_staging),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "POSTCLAIM_SETUP_SENTINEL"
                ):
                    localization.execute(
                        mode=localization.PILOT_MODE,
                        output_root=output,
                        pilot_manifest_path=manifest_path,
                        pilot_disjoint_receipt_path=disjoint_path,
                        pilot_workers=1,
                    )

            claim_path = output / "claim.json"
            failure_path = output / "failure.json"
            self.assertTrue(claim_path.is_file())
            self.assertTrue(failure_path.is_file())
            self.assertFalse((output / "success.json").exists())
            failure = localization.load_json(failure_path)
            self.assertEqual(
                failure["schema"],
                "rcle.r3_rotation_leakage_localization.failure.v1",
            )
            self.assertEqual(failure["mode"], localization.PILOT_MODE)
            self.assertEqual(
                failure["claim_sha256"],
                localization.sha256_file(claim_path),
            )
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertEqual(
                failure["error"], "POSTCLAIM_SETUP_SENTINEL"
            )
            self.assertEqual(failure["completed_clusters"], 0)
            self.assertEqual(failure["completed_cluster_ids"], [])
            self.assertEqual(failure["residual_worker_pids"], [])
            self.assertIs(failure["retry_authorized"], False)
            self.assertEqual(
                failure["terminal"],
                "INVALID_INCOMPLETE_ONE_SHOT_CONSUMED / "
                "NO_RETRY_REPLACEMENT_RESEED_RESUME_OR_OUTPUT_DELETE",
            )

    def test_atomic_claim_collision_loser_cannot_pollute_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "pilots" / "winner"

            def publish_foreign_then_collide(
                output_root: Path, claim: dict[str, object]
            ) -> None:
                foreign_claim = {**claim, "claim_id": "f" * 32}
                output_root.mkdir()
                localization.write_exclusive_json(
                    output_root / "claim.json", foreign_claim
                )
                raise FileExistsError("FOREIGN_CLAIM_COLLISION")

            with self.assertRaisesRegex(
                FileExistsError, "FOREIGN_CLAIM_COLLISION"
            ):
                _execute_pilot_claim_fixture(
                    root, output, publish_foreign_then_collide
                )
            self.assertFalse((output / "failure.json").exists())
            self.assertEqual(
                localization.load_json(output / "claim.json")["claim_id"],
                "f" * 32,
            )

    def test_atomic_claim_owner_interrupted_after_publish_writes_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "pilots" / "interrupted-owner"

            def publish_then_interrupt(
                output_root: Path, claim: dict[str, object]
            ) -> None:
                output_root.mkdir()
                localization.write_exclusive_json(
                    output_root / "claim.json", claim
                )
                raise KeyboardInterrupt("POST_RENAME_INTERRUPT")

            with self.assertRaisesRegex(
                KeyboardInterrupt, "POST_RENAME_INTERRUPT"
            ):
                _execute_pilot_claim_fixture(
                    root, output, publish_then_interrupt
                )
            failure = localization.load_json(output / "failure.json")
            self.assertEqual(failure["error_type"], "KeyboardInterrupt")
            self.assertEqual(failure["error"], "POST_RENAME_INTERRUPT")


if __name__ == "__main__":
    unittest.main()
