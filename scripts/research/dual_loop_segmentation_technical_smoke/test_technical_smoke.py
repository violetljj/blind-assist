from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import technical_smoke


class TechnicalSmokeContractTest(unittest.TestCase):
    def test_load_observations_only_accepts_hidden_candidate_and_prior_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "fixed_units": [
                            {
                                "unit_id": "clip-0",
                                "observations": [
                                    {
                                        "unit_id": "clip-0",
                                        "session_id": "session-0",
                                        "slot_ordinal": 0,
                                        "review_image_path": "image.png",
                                        "candidate_output_visible": False,
                                        "prior_review_visible": False,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = technical_smoke.load_observations(manifest)
            self.assertEqual(["clip-0"], [row["unit_id"] for row in rows])
            self.assertEqual(0, rows[0]["slot_ordinal"])

    def test_load_observations_rejects_visible_candidate_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "fixed_units": [
                            {
                                "unit_id": "clip-0",
                                "observations": [
                                    {
                                        "unit_id": "clip-0",
                                        "session_id": "session-0",
                                        "slot_ordinal": 0,
                                        "review_image_path": "image.png",
                                        "candidate_output_visible": True,
                                        "prior_review_visible": False,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "candidate output visibility"):
                technical_smoke.load_observations(manifest)

    def test_class_distribution_is_explicit_and_fractional(self) -> None:
        rows = technical_smoke.class_distribution([3, 1, 0, 0], 4)
        self.assertEqual("walkable", rows[0]["name"])
        self.assertAlmostEqual(0.75, rows[0]["fraction"])
        self.assertEqual(1, rows[1]["pixels"])

    def test_output_must_stay_under_artifacts_local(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "artifacts.local").mkdir()
            with self.assertRaisesRegex(ValueError, "artifacts.local"):
                technical_smoke.ensure_artifact_path(root, root / "outside.json")

    def test_quantized_rgb_shape_and_range(self) -> None:
        from PIL import Image

        tensor = technical_smoke._prepare_int8_rgb(
            Image.new("RGB", (3, 2), (255, 0, 127)),
            (1, 4, 4, 3),
            scale=1.0,
            zero_point=-128,
        )
        self.assertEqual((1, 4, 4, 3), tensor.shape)
        self.assertEqual(np.int8, tensor.dtype)
        self.assertTrue(np.all(tensor <= 127))
        self.assertTrue(np.all(tensor >= -128))


if __name__ == "__main__":
    unittest.main()
