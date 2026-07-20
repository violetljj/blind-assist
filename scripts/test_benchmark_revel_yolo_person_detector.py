import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("benchmark_revel_yolo_person_detector.py")
SPEC = importlib.util.spec_from_file_location("revel_detector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RevelYoloPersonDetectorTest(unittest.TestCase):
    def test_iou_and_score_ordered_matching(self):
        prediction = np.asarray([[0, 0, 1, 1], [.1, .1, .9, .9]], dtype=float)
        truth, matched = MODULE._match(prediction, np.asarray([.4, .9]), np.asarray([[0, 0, 1, 1]], dtype=float), .05)
        np.testing.assert_array_equal(truth, np.asarray([True, False])); np.testing.assert_array_equal(matched, np.asarray([True]))

    def test_area_strata(self):
        self.assertEqual("small", MODULE._stratum(.01)); self.assertEqual("medium", MODULE._stratum(.03)); self.assertEqual("large", MODULE._stratum(.2))

    def test_uniform_frame_selection_is_bounded_and_deterministic(self):
        self.assertEqual([0, 3, 6, 9], MODULE._select_indices(10, 4, "uniform"))
        self.assertEqual([0, 1, 2, 3], MODULE._select_indices(10, 4, "head"))
        self.assertEqual(list(range(3)), MODULE._select_indices(3, 8, "uniform"))

    def test_frame_selection_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            MODULE._select_indices(10, 0, "uniform")
        with self.assertRaises(ValueError):
            MODULE._select_indices(10, 2, "random")

    def test_runtime_guard_rejects_unsafe_values(self):
        MODULE._validate_runtime(batch=1, imgsz=256, memory_fraction=.25, inter_batch_delay_ms=0)
        for arguments in ((0, 256, .25, 0), (1, 32, .25, 0), (1, 256, 0, 0), (1, 256, 1.1, 0), (1, 256, .25, -1)):
            with self.assertRaises(ValueError):
                MODULE._validate_runtime(*arguments)
        MODULE._validate_runtime(1, 256, .15, 250, MODULE.INFERENCE_MODE_TILED, 50)
        MODULE._validate_runtime(8, 256, .35, 0, MODULE.INFERENCE_MODE_TILED, 0)

    def test_selection_contract_is_hash_bindable_and_strict(self):
        contract = {
            "format": "blindassist_revel_crop_tiling_selection_v1",
            "sample_role": "failure_enriched_crop_tiling_upper_bound",
            "source_details_receipt": {"sha256": "a" * 64},
            "selected_indices": [1, 4, 9],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            indices, loaded = MODULE._read_selection_contract(path, total=10, expected_count=3)
            self.assertEqual([1, 4, 9], indices)
            self.assertEqual(contract, loaded)
            for invalid in ([1, 1, 9], [4, 1, 9], [1, 4, 10]):
                contract["selected_indices"] = invalid
                path.write_text(json.dumps(contract), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE._read_selection_contract(path, total=10, expected_count=3)

    def test_fixed_corner_crops_cover_source_with_overlap(self):
        self.assertEqual(
            [(0, 0, 208, 156), (138, 0, 346, 156), (0, 104, 208, 260), (138, 104, 346, 260)],
            MODULE._tile_bounds(346, 260),
        )
        bounds = MODULE._tile_bounds(259, 260)
        self.assertEqual((0, 0, 156, 156), bounds[0])
        self.assertEqual((103, 104, 259, 260), bounds[-1])

    def test_tile_boxes_remap_and_cross_view_nms_are_deterministic(self):
        remapped = MODULE._remap_prediction(
            np.asarray([[10, 20, 30, 40]], dtype=float),
            full_width=100,
            full_height=80,
            offset_x=40,
            offset_y=20,
        )
        np.testing.assert_allclose([[.5, .5, .7, .75]], remapped)
        boxes = np.asarray([[.1, .1, .4, .4], [.11, .11, .39, .39], [.6, .6, .8, .8]], dtype=float)
        kept = MODULE._nms_indices(boxes, np.asarray([.8, .9, .7]), .5)
        np.testing.assert_array_equal([1, 2], kept)

    def test_frame_detail_preserves_timestamp_boxes_and_match_state(self):
        record = MODULE._frame_detail_record(
            7,
            Path("1700000000000000000.jpg"),
            np.asarray([[.1, .2, .3, .4]], dtype=float),
            np.asarray([.8], dtype=float),
            np.asarray([[.1, .2, .3, .4]], dtype=float),
            np.asarray([.04], dtype=float),
            np.asarray([True]),
            np.asarray([True]),
        )
        self.assertEqual(1700000000000000000, record["source_timestamp_ns"])
        self.assertEqual("medium", record["ground_truth"][0]["stratum"])
        self.assertTrue(record["ground_truth"][0]["matched_at_fixed_score"])
        self.assertEqual({"tp": 1, "fp": 0, "fn": 0}, record["fixed_score_counts"])
        self.assertEqual("full_frame", record["predictions_over_score_floor"][0]["view_id"])
        self.assertEqual({"mode": "full_frame", "view_count": 1, "raw_prediction_count": 1, "fused_prediction_count": 1}, record["inference"])


if __name__ == "__main__":
    unittest.main()
