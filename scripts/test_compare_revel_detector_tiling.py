import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("compare_revel_detector_tiling.py")
SPEC = importlib.util.spec_from_file_location("revel_tiling", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


CONTRACT_SHA = "c" * 64


def _guard(frames, mode, batch=1):
    views = frames * (5 if mode == MODULE.CANDIDATE_MODE else 1)
    return {
        "format": "blindassist_guarded_gpu_run_v1", "exit_code": 0, "stop_reason": None, "monitor_samples": 3,
        "limits": {"max_temperature_c": 72, "max_frames": frames, "batch": batch, "inference_mode": mode, "max_inference_views": views},
        "observed": {"max_temperature_c": 50, "relevant_system_events": 0},
    }


def _benchmark(frames, indices, mode, tp, fp, fn, batch=1):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "format": "blindassist_revel_yolo11n_person_benchmark_v2",
        "dataset": {"evaluated_frames": frames, "selected_indices": indices, "selection_receipt": {"sha256": CONTRACT_SHA}},
        "model": {"weights_sha256": "w", "imgsz": 256, "batch": batch, "half": False, "score_floor": .05, "iou_threshold": .5, "inference_mode": mode},
        "fixed_score_metrics": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)},
        "compute_backend": {"inference_views": frames * (5 if mode == MODULE.CANDIDATE_MODE else 1)},
    }


def _row(index, ground_truth, fp=0):
    tp = sum(item["matched_at_fixed_score"] for item in ground_truth)
    return {
        "selected_index": index, "image_name": f"{1000 + index}.jpg", "source_timestamp_ns": 1000 + index,
        "ground_truth": ground_truth,
        "fixed_score_counts": {"tp": tp, "fp": fp, "fn": len(ground_truth) - tp},
    }


def _gt(stratum, matched):
    area = {"small": .01, "medium": .04, "large": .2}[stratum]
    return {"xyxy_normalized": [.1, .1, .2, .2], "normalized_area": area, "stratum": stratum, "matched_at_fixed_score": matched}


def _stage8():
    indices = list(range(8))
    baseline_rows = []
    candidate_rows = []
    for index in indices:
        extra = []
        if index < 3:
            extra.append(_gt("medium", True))
        elif index < 6:
            extra.append(_gt("large", True))
        baseline_rows.append(_row(index, [_gt("small", False), *extra]))
        candidate_rows.append(_row(index, [_gt("small", index < 2), *extra], fp=1 if index < 4 else 0))
    contract = {
        "format": "blindassist_revel_crop_tiling_selection_v1", "stage_frames": 8,
        "sample_role": "failure_enriched_crop_tiling_upper_bound", "selected_indices": indices,
        "expected_baseline": {"tp": 6, "fp": 4, "fn": 8, "strata": {"small": {"ground_truth": 8, "matched": 0}, "medium": {"ground_truth": 3, "matched": 3}, "large": {"ground_truth": 3, "matched": 3}}},
        "failure_segment_by_selected_index": {str(index): index for index in indices},
    }
    return (
        _benchmark(8, indices, MODULE.BASELINE_MODE, 6, 4, 8),
        _benchmark(8, indices, MODULE.CANDIDATE_MODE, 8, 4, 6),
        baseline_rows, candidate_rows, contract,
    )


class RevelDetectorTilingComparisonTest(unittest.TestCase):
    def test_eight_frame_canary_advances_only_with_recovery_and_no_regression(self):
        baseline, candidate, baseline_rows, candidate_rows, contract = _stage8()
        report = MODULE.compare(baseline, candidate, _guard(8, MODULE.BASELINE_MODE), _guard(8, MODULE.CANDIDATE_MODE), baseline_rows, candidate_rows, contract, CONTRACT_SHA)
        self.assertTrue(report["gate_passed"])
        self.assertEqual("advance_to_32_frame_gate", report["decision"])
        self.assertEqual(2, report["paired_gt"]["small_recovered"])

    def test_rejects_same_endpoints_with_different_middle_indices(self):
        baseline, candidate, baseline_rows, candidate_rows, contract = _stage8()
        candidate["dataset"] = dict(candidate["dataset"])
        candidate["dataset"]["selected_indices"] = [0, 1, 2, 4, 3, 5, 6, 7]
        with self.assertRaises(ValueError):
            MODULE.compare(baseline, candidate, _guard(8, MODULE.BASELINE_MODE), _guard(8, MODULE.CANDIDATE_MODE), baseline_rows, candidate_rows, contract, CONTRACT_SHA)

    def test_paired_gate_accepts_matching_moderate_batch(self):
        baseline, candidate, baseline_rows, candidate_rows, contract = _stage8()
        baseline["model"]["batch"] = 8
        candidate["model"]["batch"] = 8
        report = MODULE.compare(
            baseline,
            candidate,
            _guard(8, MODULE.BASELINE_MODE, batch=8),
            _guard(8, MODULE.CANDIDATE_MODE, batch=8),
            baseline_rows,
            candidate_rows,
            contract,
            CONTRACT_SHA,
        )
        self.assertTrue(report["gate_passed"])

    def test_thirty_two_frame_gate_requires_five_independent_recoveries(self):
        indices = list(range(32))
        baseline_rows = []
        candidate_rows = []
        for index in indices:
            if index >= 28:
                baseline_rows.append(_row(index, [])); candidate_rows.append(_row(index, [])); continue
            baseline_small = index < 8
            extra = []
            if index < 13:
                extra.append(_gt("medium", index < 12))
            if index < 8:
                extra.append(_gt("large", index < 7))
            baseline_rows.append(_row(index, [_gt("small", baseline_small), *extra]))
            candidate_rows.append(_row(index, [_gt("small", baseline_small or 8 <= index < 13), *extra]))
        contract = {
            "format": "blindassist_revel_crop_tiling_selection_v1", "stage_frames": 32,
            "sample_role": "failure_enriched_crop_tiling_upper_bound", "selected_indices": indices,
            "expected_baseline": {"tp": 27, "fp": 21, "fn": 22, "strata": {"small": {"ground_truth": 28, "matched": 8}, "medium": {"ground_truth": 13, "matched": 12}, "large": {"ground_truth": 8, "matched": 7}}},
            "failure_segment_by_selected_index": {str(index): index for index in range(8, 28)},
        }
        report = MODULE.compare(
            _benchmark(32, indices, MODULE.BASELINE_MODE, 27, 21, 22),
            _benchmark(32, indices, MODULE.CANDIDATE_MODE, 32, 21, 17),
            _guard(32, MODULE.BASELINE_MODE), _guard(32, MODULE.CANDIDATE_MODE),
            baseline_rows, candidate_rows, contract, CONTRACT_SHA,
        )
        self.assertTrue(report["gate_passed"])
        self.assertLessEqual(report["paired_gt"]["one_sided_exact_mcnemar_p"], .05)
        self.assertEqual(5, report["paired_gt"]["recovered_failure_segments"])


if __name__ == "__main__":
    unittest.main()
