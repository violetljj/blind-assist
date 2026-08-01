from __future__ import annotations

import unittest
import sys
import gzip
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_geometry_teacher_canary import (
    _advected_basis,
    _bin_obstacle_support,
    _causal_tangent_velocity,
    _cell_probes_world,
    _coverage_fraction,
    _decide_terminal,
    _known_field,
    _required_denominators,
    _select_horizon_indices,
    _select_history_indices,
    _theta_edges,
)


class GeometryTeacherCanaryTest(unittest.TestCase):
    def test_nominal_horizon_binding_respects_tolerance(self) -> None:
        timestamps = [0, 143, 286, 429, 571, 714, 857]
        source_frames = [0, 2, 4, 6, 8, 10, 12]
        self.assertEqual(
            _select_horizon_indices(timestamps, source_frames, 400, 100),
            [3, 4, 5, 6, None, None, None],
        )
        self.assertEqual(
            _select_horizon_indices(timestamps, source_frames, 800, 100),
            [6, 6, None, None, None, None, None],
        )

    def test_horizon_tie_prefers_lower_source_frame(self) -> None:
        self.assertEqual(
            _select_horizon_indices(
                [0, 300, 500],
                [0, 3, 5],
                400,
                100,
            ),
            [1, None, None],
        )

    def test_history_tie_prefers_higher_source_frame(self) -> None:
        self.assertEqual(
            _select_history_indices(
                [100, 300, 500],
                [1, 3, 5],
                300,
                100,
            ),
            [None, 0, 1],
        )

    def test_history_binding_respects_strict_past_and_tolerance(self) -> None:
        self.assertEqual(
            _select_history_indices(
                [0, 200, 400, 600, 800],
                [0, 2, 4, 6, 8],
                400,
                50,
            ),
            [None, None, 0, 1, 2],
        )

    def test_advected_origin_uses_only_ground_tangent_velocity(self) -> None:
        basis = (
            np.asarray([1.0, 2.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        history = {
            "position_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        anchor = {
            "position_m": [1.0, 2.0, 1.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        velocity = _causal_tangent_velocity(
            history, anchor, 0.4, basis[3]
        )
        np.testing.assert_allclose(velocity, [2.5, 5.0, 0.0])
        shifted = _advected_basis(basis, velocity, 800)
        np.testing.assert_allclose(shifted[0], [3.0, 6.0, 0.0])
        np.testing.assert_array_equal(shifted[1], basis[1])
        np.testing.assert_array_equal(shifted[2], basis[2])
        np.testing.assert_array_equal(shifted[3], basis[3])

    def test_height_layers_are_binned_independently(self) -> None:
        points = np.asarray(
            [
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [0.1, 0.8, 1.8],
            ]
        )
        basis = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        counts, dynamic = _bin_obstacle_support(
            points,
            np.asarray([False, True, False]),
            basis,
            np.linspace(-np.pi, np.pi, 25),
            np.asarray([0.0, 1.0]),
            [(0.05, 0.35), (0.35, 1.35), (1.35, 2.05)],
        )
        self.assertEqual([1, 1, 1], counts.sum(axis=(0, 1)).tolist())
        self.assertEqual([0, 1, 0], dynamic.sum(axis=(0, 1)).tolist())

    def test_forward_sector_excludes_points_outside_frozen_angles(self) -> None:
        points = np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0, -1.0],
                [-1.0, 0.0, 1.0, 2.0, 0.0],
                [0.1, 0.1, 0.1, 0.1, 0.1],
            ]
        )
        basis = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        theta_edges = _theta_edges(
            {
                "theta_bin_count": 6,
                "theta_range_degrees": [-45.0, 45.0],
            }
        )
        counts, _ = _bin_obstacle_support(
            points,
            np.zeros(points.shape[1], dtype=bool),
            basis,
            theta_edges,
            np.asarray([0.0, 3.0]),
            [(0.05, 0.35)],
        )
        self.assertEqual(3, int(counts.sum()))
        self.assertEqual(1, int(counts[0].sum()))
        self.assertEqual(1, int(counts[-1].sum()))

    def test_probe_geometry_uses_frozen_forward_sector_edges(self) -> None:
        basis = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        theta_edges = _theta_edges(
            {
                "theta_bin_count": 6,
                "theta_range_degrees": [-45.0, 45.0],
            }
        )
        probes = _cell_probes_world(
            basis,
            theta_edges,
            np.asarray([0.0, 1.0]),
            [(0.05, 0.35)],
        )
        self.assertEqual((6, 3, 9), probes.shape)
        self.assertGreaterEqual(float(probes[:, 0, :].min()), -1e-12)
        angles = np.arctan2(probes[:, 1, :], probes[:, 0, :])
        self.assertGreaterEqual(float(angles.min()), -np.pi / 4 - 1e-12)
        self.assertLessEqual(float(angles.max()), np.pi / 4 + 1e-12)

    def test_full_circle_keeps_positive_pi_in_first_wrapped_bin(self) -> None:
        points = np.asarray([[-1.0], [0.0], [0.1]])
        basis = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        counts, _ = _bin_obstacle_support(
            points,
            np.asarray([False]),
            basis,
            np.linspace(-np.pi, np.pi, 25),
            np.asarray([0.0, 2.0]),
            [(0.05, 0.35)],
        )
        self.assertEqual(1, int(counts[0].sum()))
        self.assertEqual(1, int(counts.sum()))

    def test_unknown_cells_never_shrink_frozen_denominators(self) -> None:
        self.assertEqual(
            _required_denominators(12, 24, 6, 3),
            {
                "known_per_horizon": 5184,
                "height_disagreement": 1728,
                "future_union": 5184,
            },
        )
        self.assertEqual(0.0, _coverage_fraction(0, 0))

    def test_semantic_zero_probe_cannot_become_known_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "depth").mkdir()
            (root / "masks").mkdir()
            depth_values = np.concatenate(
                (
                    np.asarray([4, 4], dtype="<f2"),
                    np.full(16, 10.0, dtype="<f2"),
                )
            )
            depth_path = root / "depth" / "000000.float16.gz"
            depth_path.write_bytes(gzip.compress(depth_values.tobytes()))
            mask_path = root / "masks" / "000000.png"
            Image.fromarray(
                np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB"
            ).save(mask_path)
            probes = np.zeros((1, 3, 9), dtype=np.float64)
            probes[:, 2, :] = 1.0
            row = {
                "width": 4,
                "height": 4,
                "source_depth_path": "depth/000000.float16.gz",
                "source_mask_path": "masks/000000.png",
            }
            binding = {
                "position_m": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
            known, score = _known_field(
                probes,
                root,
                row,
                binding,
                {"fx": 1.0, "fy": 1.0, "cx": 1.5, "cy": 1.5},
                1,
                1,
                1,
                0.2,
                5,
            )

        self.assertFalse(bool(known[0, 0, 0]))
        self.assertEqual(0.0, float(score[0, 0, 0]))

    def test_source_failure_precedes_representation_terminals(self) -> None:
        sessions = [
            {
                "authority_validation": {"ok": False},
                "mechanics_valid": False,
                "multi_height_supported": False,
                "future_supported": False,
            }
        ]
        self.assertEqual(
            _decide_terminal(sessions),
            "H1_GEOMETRY_TEACHER_NOT_EVALUABLE",
        )

    def test_multi_height_failure_precedes_future_failure(self) -> None:
        sessions = [
            {
                "authority_validation": {"ok": True},
                "mechanics_valid": True,
                "multi_height_supported": False,
                "future_supported": False,
            }
        ]
        self.assertEqual(
            _decide_terminal(sessions),
            "H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP",
        )

    def test_future_failure_has_specific_terminal(self) -> None:
        sessions = [
            {
                "authority_validation": {"ok": True},
                "mechanics_valid": True,
                "multi_height_supported": True,
                "future_supported": False,
            }
        ]
        self.assertEqual(
            _decide_terminal(sessions),
            "H1_FUTURE_PROXY_NOT_SUPPORTED_STOP",
        )

    def test_all_session_gates_support_mechanism_only(self) -> None:
        sessions = [
            {
                "authority_validation": {"ok": True},
                "mechanics_valid": True,
                "multi_height_supported": True,
                "future_supported": True,
            }
            for _ in range(4)
        ]
        self.assertEqual(
            _decide_terminal(sessions),
            "GEOMETRY_PROXY_MECHANISM_SUPPORTED",
        )


if __name__ == "__main__":
    unittest.main()
