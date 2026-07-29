from __future__ import annotations

import copy
import hashlib
from unittest import mock
import unittest

import numpy as np

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    material_residual_contraction_r1 as operator,
)


def _scene() -> dict:
    return {
        "scene_geometry_sha256": "a" * 64,
        "world": {
            "objects": [
                {
                    "object_id": 7,
                    "linear_rgb": [0.8, 0.6, 0.4],
                    "texture": {
                        "cycles_per_m": 1.0,
                        "phase": 0.0,
                    },
                }
            ]
        },
    }


def _raycast(*_args):
    depth = np.asarray([2.0, 2.0], dtype=np.float64)
    object_id = np.asarray([7, 7], dtype=np.int32)
    world = np.asarray(
        [[0.25, 0.25, 2.0], [1.25, 0.25, 2.0]],
        dtype=np.float64,
    )
    return depth, object_id, world


def _sha256_f8(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array.astype("<f8", copy=False)).tobytes()
    ).hexdigest()


class MaterialResidualContractionR1Test(unittest.TestCase):
    def _render(self):
        with (
            mock.patch.object(operator.p1, "WIDTH", 2),
            mock.patch.object(operator.p1, "HEIGHT", 1),
            mock.patch.object(
                operator.p1,
                "_raycast",
                side_effect=_raycast,
            ) as raycast,
        ):
            result = operator.render_pair(
                _scene(),
                np.eye(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        self.assertEqual(raycast.call_count, 1)
        return result

    def test_prequantization_residual_ratio_is_exactly_frozen(self) -> None:
        result = self._render()
        base = np.asarray([0.8, 0.6, 0.4], dtype=np.float64)
        clean_modulation = np.asarray([0.65, 1.0], dtype=np.float64)
        mean_modulation = np.full(2, 0.825, dtype=np.float64)
        low_modulation = mean_modulation + 0.15 * (
            clean_modulation - mean_modulation
        )
        clean = base[None, :] * clean_modulation[:, None]
        mean = base[None, :] * mean_modulation[:, None]
        low = mean + 0.15 * (clean - mean)

        identity = result["prequantization_identity"]
        self.assertEqual(
            identity["clean_linear_rgb_sha256"],
            _sha256_f8(clean.reshape(1, 2, 3)),
        )
        self.assertEqual(
            identity["material_mean_linear_rgb_sha256"],
            _sha256_f8(mean.reshape(1, 2, 3)),
        )
        self.assertEqual(
            identity["low_linear_rgb_sha256"],
            _sha256_f8(low.reshape(1, 2, 3)),
        )
        self.assertEqual(identity["residual_relation_max_abs_error"], 0.0)

    def test_two_checker_states_preserve_material_mean(self) -> None:
        clean = np.asarray([0.65, 1.0], dtype=np.float64)
        low = 0.825 + 0.15 * (clean - 0.825)
        self.assertEqual(float(np.mean(clean)), 0.825)
        self.assertEqual(float(np.mean(low)), 0.825)

    def test_geometry_and_scene_hash_are_shared_and_unchanged(self) -> None:
        scene = _scene()
        before = copy.deepcopy(scene)
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        rotation_before = rotation.copy()
        translation_before = translation.copy()
        with (
            mock.patch.object(operator.p1, "WIDTH", 2),
            mock.patch.object(operator.p1, "HEIGHT", 1),
            mock.patch.object(operator.p1, "_raycast", side_effect=_raycast),
        ):
            first = operator.render_pair(
                scene,
                rotation,
                translation,
            )

        self.assertEqual(scene, before)
        np.testing.assert_array_equal(rotation, rotation_before)
        np.testing.assert_array_equal(translation, translation_before)
        self.assertEqual(
            first["geometry_identity"]["scene_geometry_sha256"],
            before["scene_geometry_sha256"],
        )
        self.assertEqual(first["valid_mask"].tolist(), [[True, True]])
        self.assertEqual(first["object_id"].tolist(), [[7, 7]])
        self.assertEqual(
            first["rgb_pair"]["clean"].shape,
            first["rgb_pair"]["low"].shape,
        )

    def test_deterministic_rgb_and_identity_hashes(self) -> None:
        first = self._render()
        second = self._render()
        np.testing.assert_array_equal(
            first["rgb_pair"]["clean"],
            second["rgb_pair"]["clean"],
        )
        np.testing.assert_array_equal(
            first["rgb_pair"]["low"],
            second["rgb_pair"]["low"],
        )
        self.assertEqual(
            first["prequantization_identity"],
            second["prequantization_identity"],
        )
        self.assertEqual(
            first["geometry_identity"],
            second["geometry_identity"],
        )

    def test_alpha_and_operator_parameter_mutations_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "QMS_R1_ALPHA_MUST_EQUAL_FROZEN_0_15",
        ):
            operator.render_pair(
                _scene(),
                np.eye(3),
                np.zeros(3),
                alpha=0.1500001,
            )
        for name, value in (
            ("OPERATOR_ID", "DRIFT"),
            ("CLEAN_MODULATION_LOW", 0.64),
            ("CLEAN_MODULATION_RANGE", 0.36),
            ("MATERIAL_MEAN_MODULATION", 0.824),
            ("ALPHA", 0.2),
            ("PSF_NONE", False),
        ):
            with self.subTest(name=name):
                with mock.patch.object(operator, name, value):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "QMS_R1_OPERATOR_IDENTITY_DRIFT",
                    ):
                        operator.render_pair(
                            _scene(),
                            np.eye(3),
                            np.zeros(3),
                        )

    def test_external_identity_mutation_is_rejected(self) -> None:
        identity = operator.frozen_operator_identity()
        for key, value in (
            ("alpha", 0.30),
            ("material_mean_modulation", 0.5),
            ("pairing", "TWO_RAYCASTS"),
            ("psf_none", False),
        ):
            with self.subTest(key=key):
                mutation = dict(identity)
                mutation[key] = value
                with self.assertRaisesRegex(
                    ValueError,
                    "QMS_R1_OPERATOR_IDENTITY_INVALID",
                ):
                    operator.validate_operator_identity(mutation)


if __name__ == "__main__":
    unittest.main()
