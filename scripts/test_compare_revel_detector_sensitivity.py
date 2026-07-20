import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("compare_revel_detector_sensitivity.py")
SPEC = importlib.util.spec_from_file_location("revel_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _benchmark(imgsz, small, f1):
    return {
        "format": "blindassist_revel_yolo11n_person_benchmark_v2",
        "dataset": {"total_frames": 8580, "evaluated_frames": 128, "selection": "uniform", "selected_first_index": 0, "selected_last_index": 8579, "person_ground_truth_boxes": 196},
        "model": {"weights_sha256": "abc", "batch": 1, "half": False, "imgsz": imgsz},
        "fixed_score_metrics": {"precision": .9, "recall": .9, "f1": f1},
        "ap50_over_score_floor": .9,
        "recall_by_normalized_box_area": {"small": {"recall": small}, "medium": {"recall": .9}, "large": {"recall": .95}},
    }


def _guard():
    return {
        "format": "blindassist_guarded_gpu_run_v1", "exit_code": 0, "stop_reason": None, "monitor_samples": 10,
        "limits": {"max_temperature_c": 72, "max_frames": 128, "batch": 1},
        "observed": {"max_temperature_c": 50, "max_power_draw_w": 30, "relevant_system_events": 0},
    }


class RevelDetectorSensitivityTest(unittest.TestCase):
    def test_rejects_resolution_without_small_target_gain(self):
        report = MODULE.compare(_benchmark(256, .4, .90), _benchmark(320, .4, .89), _guard(), _guard())
        self.assertEqual("do_not_scale_candidate_to_512", report["recommendation"])
        self.assertEqual(0.0, report["delta_candidate_minus_baseline"]["recall_by_area"]["small"])

    def test_accepts_resolution_with_small_gain_and_non_regressing_f1(self):
        report = MODULE.compare(_benchmark(256, .4, .90), _benchmark(320, .5, .91), _guard(), _guard())
        self.assertEqual("scale_candidate_to_512", report["recommendation"])

    def test_rejects_unpaired_dataset(self):
        candidate = _benchmark(320, .5, .91)
        candidate["dataset"]["evaluated_frames"] = 64
        with self.assertRaises(ValueError):
            MODULE.compare(_benchmark(256, .4, .90), candidate, _guard(), _guard())


if __name__ == "__main__":
    unittest.main()
