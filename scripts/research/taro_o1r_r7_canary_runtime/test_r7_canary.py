from __future__ import annotations

import copy
import inspect
import unittest

from scripts.research.taro_o1r_r7_canary_runtime import r7_canary as runtime


def _feature() -> dict:
    hits = [[[False for _ in range(3)] for _ in range(3)] for _ in range(4)]
    return {
        "query_receipt": {"present": True},
        "r6_state": "UNKNOWN",
        "occupied_hits": hits,
        "positive_obstacle_veto": False,
        "far_fractions": [1.0, 1.0, 1.0],
        "far_valid_anchor_count": runtime.MINIMUM_FAR_VISIBLE_ANCHORS,
        "observed_support_points": 128,
    }


def _unavailable_source() -> dict:
    queries = []
    for index in range(9):
        queries.append(
            {
                "grid_index": index,
                "query_id": f"frame:q{index}",
                "query_receipt": None,
                "r6_state": "UNKNOWN",
                "occupied_hits": None,
                "positive_obstacle_veto": None,
                "far_fractions": None,
                "far_valid_anchor_count": 0,
                "observed_support_points": 0,
                "reason_codes": ["SOURCE_QUERY_FRAME_UNAVAILABLE"],
            }
        )
    return runtime._seal(
        {
            "schema": runtime.SOURCE_FRAME_SCHEMA,
            "reducer_id": runtime.REDUCER_ID,
            "parent_id": "p",
            "video_id": "v",
            "timestamp_token": "1.0",
            "physical_frame_id": "v:1.0",
            "query_features": queries,
            "source_phase_has_label_input": False,
            "training_steps": 0,
            "network_requests": 0,
        }
    )


class R7CanaryTests(unittest.TestCase):
    def test_source_api_has_no_label_side_parameter(self) -> None:
        names = inspect.signature(runtime.build_source_frame_record).parameters
        self.assertFalse(any(token in name.lower() for name in names for token in ("faro", "truth", "label", "outcome")))

    def test_exact_candidate_grid_cardinality(self) -> None:
        configs = runtime.candidate_configs()
        self.assertEqual(len(configs), 972)
        self.assertEqual(len(set(configs)), 972)

    def test_positive_occupied_precedes_clear(self) -> None:
        feature = _feature()
        feature["occupied_hits"][0][0][0] = True
        self.assertEqual(runtime.predict_query_state(feature, (0, 0, 0, 0, 0, 0)), "OCCUPIED_OBSERVED")

    def test_weak_positive_evidence_vetoes_clear(self) -> None:
        feature = _feature()
        feature["positive_obstacle_veto"] = True
        self.assertEqual(runtime.predict_query_state(feature, (3, 2, 0, 0, 0, 0)), "UNKNOWN")

    def test_clear_requires_visible_far_support(self) -> None:
        feature = _feature()
        config = (3, 2, 0, 0, 0, 2)
        self.assertEqual(runtime.predict_query_state(feature, config), "CLEAR_OBSERVED")
        feature["far_valid_anchor_count"] -= 1
        self.assertEqual(runtime.predict_query_state(feature, config), "UNKNOWN")
        feature["far_valid_anchor_count"] += 1
        feature["observed_support_points"] = 127
        self.assertEqual(runtime.predict_query_state(feature, config), "UNKNOWN")

    def test_existing_r6_definite_state_is_preserved(self) -> None:
        feature = _feature()
        feature["r6_state"] = "OCCUPIED_OBSERVED"
        self.assertEqual(runtime.predict_query_state(feature, (3, 2, 2, 2, 2, 2)), "OCCUPIED_OBSERVED")

    def test_unknown_truth_is_not_false_clear_or_precision_negative(self) -> None:
        feature = _feature()
        rows = [("p", feature, {"state": "UNKNOWN"})]
        metrics = runtime._metrics(rows, (3, 2, 2, 0, 0, 0))
        self.assertEqual(metrics["false_clear_count"], 0)
        self.assertIsNone(metrics["occupied_precision"])

    def test_wilson_lower_is_bounded_and_penalizes_small_samples(self) -> None:
        self.assertEqual(runtime._wilson_lower(0, 0), 0.0)
        self.assertLess(runtime._wilson_lower(1, 1), runtime._wilson_lower(100, 100))
        self.assertLessEqual(runtime._wilson_lower(100, 100), 1.0)

    def test_source_record_tamper_is_rejected(self) -> None:
        record = _unavailable_source()
        self.assertEqual(runtime.validate_source_frame_record(record)["physical_frame_id"], "v:1.0")
        tampered = copy.deepcopy(record)
        tampered["training_steps"] = 1
        with self.assertRaises(runtime.R7CanaryError):
            runtime.validate_source_frame_record(tampered)

    def test_label_must_bind_sealed_source_and_keep_unknown_nonnegative(self) -> None:
        source = _unavailable_source()
        labels = [
            {
                "grid_index": index,
                "query_id": f"frame:q{index}",
                "state": "UNKNOWN",
                "obstacle_pixel_count": 0,
                "minimum_truth_obstacle_pixels": runtime.MINIMUM_TRUTH_OBSTACLE_PIXELS,
                "query_support_points": 0,
                "observed_forward_m": None,
                "local_valid_fraction": 0.0,
                "reason_codes": ["SOURCE_QUERY_FRAME_UNAVAILABLE"],
            }
            for index in range(9)
        ]
        label = runtime._seal(
            {
                "schema": runtime.LABEL_FRAME_SCHEMA,
                "reducer_id": runtime.REDUCER_ID,
                "physical_frame_id": source["physical_frame_id"],
                "source_frame_record_sha256": source["content_sha256"],
                "query_labels": labels,
                "source_phase_reselection": False,
                "unknown_is_negative": False,
            }
        )
        self.assertFalse(runtime.validate_label_frame_record(label, source)["unknown_is_negative"])
        tampered = copy.deepcopy(label)
        tampered["source_frame_record_sha256"] = "0" * 64
        tampered.pop("content_sha256")
        tampered = runtime._seal(tampered)
        with self.assertRaises(runtime.R7CanaryError):
            runtime.validate_label_frame_record(tampered, source)


if __name__ == "__main__":
    unittest.main()
