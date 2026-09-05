from __future__ import annotations

import copy
import unittest

import dtr_final_reckoning_event_metrics as metrics


def episode(truth: list, prediction: list, contact: float = 0.5) -> list[dict]:
    return [{"time_s": index / 10, "truth_risk": target, "predicted_active": alert,
             "current_contact": False, "truth_contact_time_s": contact if target is True else None}
            for index, (target, alert) in enumerate(zip(truth, prediction, strict=True))]


class EventMetricsTest(unittest.TestCase):
    def test_continuous_alert_can_match_only_one_of_two_events(self) -> None:
        result = metrics.evaluate_episode(episode([True, True, False, True, True, False], [True] * 5 + [False]))
        self.assertEqual((1, 0, 1), (result["event_tp"], result["event_fp"], result["event_fn"]))
        self.assertEqual(1, result["alert_event_count"])
        self.assertEqual(0, result["matches"][0]["truth_event_id"])
        self.assertAlmostEqual(2 / 3, result["event_f1"])

    def test_unknown_truth_splits_truth_but_never_invents_alert_onset(self) -> None:
        result = metrics.evaluate_episode(episode([True, True, None, True, True, False], [True] * 5 + [False]))
        self.assertEqual(2, result["truth_event_count"])
        self.assertEqual(1, result["alert_event_count"])
        self.assertEqual(1, result["event_fn"])
        self.assertEqual(5, result["joint_known_frames"])
        self.assertEqual("UNKNOWN", result["clear_records"][0]["censor_reason"])

    def test_predicted_unknown_does_not_erase_truth_event_or_fn(self) -> None:
        result = metrics.evaluate_episode(episode([True, True, False], [None, None, None]))
        self.assertEqual(1, result["event_fn"])
        self.assertEqual(2, result["prediction_unknown_truth_positive_frames"])
        self.assertEqual(1, result["prediction_unknown_truth_negative_frames"])
        self.assertEqual(0, result["joint_known_coverage"])
        self.assertIsNone(result["false_segments_per_minute"])
        self.assertEqual(0, result["false_segments_per_truth_known_minute"])

    def test_prediction_unknown_splits_alerts_and_reports_fragmentation(self) -> None:
        result = metrics.evaluate_episode(episode([True] * 5 + [False], [True, True, None, True, True, False]))
        self.assertEqual((1, 1, 0), (result["event_tp"], result["event_fp"], result["event_fn"]))
        self.assertEqual(1, result["fragmentation_extra_onsets"])
        self.assertEqual(1.0, result["fragmented_event_rate"])
        self.assertEqual(0, result["matches"][0]["alert_event_id"])

    def test_alert_wholly_on_unknown_truth_is_not_false_positive(self) -> None:
        result = metrics.evaluate_episode(episode([None, None, False], [True, True, False]))
        self.assertEqual(1, result["fully_unevaluable_alert_count"])
        self.assertEqual(0, result["event_fp"])
        self.assertEqual(0, result["evaluable_alert_event_count"])

    def test_half_open_boundary_touch_has_no_overlap(self) -> None:
        result = metrics.evaluate_episode(episode([False, True, True, False], [True, False, False, False]))
        self.assertEqual((0, 1, 1), (result["event_tp"], result["event_fp"], result["event_fn"]))
        self.assertAlmostEqual(0.1, result["alert_events"][0]["end_s"])

    def test_alert_starting_after_contact_is_not_matched(self) -> None:
        result = metrics.evaluate_episode(episode([True] * 4, [False, False, True, True], contact=0.1))
        self.assertEqual((0, 1, 1), (result["event_tp"], result["event_fp"], result["event_fn"]))

    def test_two_matching_alerts_yield_two_events(self) -> None:
        result = metrics.evaluate_episode(episode([True, True, False, True, True, False], [True, True, False, True, True, False]))
        self.assertEqual((2, 0, 0), (result["event_tp"], result["event_fp"], result["event_fn"]))
        self.assertAlmostEqual(0.35, result["median_first_alert_lead_s"])
        self.assertAlmostEqual(0.23, result["p10_first_alert_lead_s"])

    def test_missing_contact_time_rejected_without_future_event_inference(self) -> None:
        data = episode([True, False, False], [True, False, False])
        data[0]["truth_contact_time_s"] = None
        data[2]["current_contact"] = True
        with self.assertRaisesRegex(ValueError, "missing_truth_contact_time"):
            metrics.evaluate_episode(data)
        data[0]["current_contact"] = True
        self.assertEqual(0, metrics.evaluate_episode(data)["median_first_alert_lead_s"])

    def test_clear_delay_and_censoring(self) -> None:
        result = metrics.evaluate_episode(episode([True, True, False, False, False], [True, True, True, True, False]))
        self.assertAlmostEqual(0.2, result["median_clear_delay_s"])
        self.assertEqual(0, result["clear_right_censored_count"])
        for truth, pred, reason in (
            ([True, True, False, None], [True] * 4, "UNKNOWN"),
            ([True, True, False, False], [True, True, True, None], "UNKNOWN"),
            ([True, True, False, True], [True] * 4, "NEXT_TRUE_EVENT"),
            ([True, True, False, False], [True] * 4, "EPISODE_END"),
        ):
            with self.subTest(reason=reason):
                result = metrics.evaluate_episode(episode(truth, pred))
                self.assertTrue(result["clear_records"][0]["right_censored"])
                self.assertEqual(reason, result["clear_records"][0]["censor_reason"])

    def test_frame_denominator_coverage_and_both_false_rates(self) -> None:
        result = metrics.evaluate_episode(episode([False, False, False, None], [True, None, False, True]))
        self.assertEqual(1, result["event_fp"])
        self.assertEqual(1, result["fully_unevaluable_alert_count"])
        self.assertAlmostEqual(300, result["false_segments_per_minute"])
        self.assertAlmostEqual(200, result["false_segments_per_truth_known_minute"])
        self.assertEqual({"tp": 0, "fp": 1, "fn": 0, "tn": 1, "precision": 0, "recall": None, "f1": 0}, result["frame"])
        self.assertAlmostEqual(2 / 3, result["joint_known_coverage"])

    def test_aggregate_sums_counts_instead_of_averaging_episode_f1(self) -> None:
        good = metrics.evaluate_episode(episode([True, True, False], [True, True, False]))
        bad = metrics.evaluate_episode(episode([True, True, False], [False, False, True]))
        result = metrics.aggregate([good, bad])
        self.assertEqual((1, 1, 1), (result["event_tp"], result["event_fp"], result["event_fn"]))
        self.assertEqual(0.5, result["event_f1"])
        self.assertEqual(6, result["frame_count"])

    def test_paired_bootstrap_is_deterministic_and_identical_arms_have_zero_delta(self) -> None:
        good = metrics.evaluate_episode(episode([True, True, False], [True, True, False]))
        bad = metrics.evaluate_episode(episode([True, True, False], [False, False, True]))
        arm = {"episode_b": bad, "episode_a": good}
        same = metrics.paired_episode_bootstrap(arm, arm, replicates=40)
        self.assertEqual([0, 0], same["ci95"])
        self.assertFalse(same["candidate_significant_win"])
        candidate = {key: good for key in arm}
        first = metrics.paired_episode_bootstrap(arm, candidate, replicates=40)
        self.assertEqual(first, metrics.paired_episode_bootstrap(arm, candidate, replicates=40))
        self.assertGreater(first["delta_event_f1"], 0)
        mismatched = copy.deepcopy(candidate)
        mismatched["episode_a"]["truth_event_count"] += 1
        with self.assertRaisesRegex(ValueError, "paired_truth_denominator"):
            metrics.paired_episode_bootstrap(arm, mismatched, replicates=2)

    def test_invalid_states_and_sampling_are_rejected(self) -> None:
        data = episode([False, False], [False, False])
        data[0]["truth_risk"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "invalid_tristate"):
            metrics.evaluate_episode(data)
        data[0]["truth_risk"] = False
        data[1]["time_s"] = 0.3
        with self.assertRaisesRegex(ValueError, "nonuniform_sampling"):
            metrics.evaluate_episode(data)

    def test_all_unknown_is_unevaluable_not_measured_zero(self) -> None:
        result = metrics.evaluate_episode(episode([None, None], [None, None]))
        for key in ("event_precision", "event_recall", "event_f1", "frame_precision", "frame_recall", "frame_f1",
                    "false_segments_per_minute", "false_segments_per_truth_known_minute", "joint_known_coverage",
                    "fragmented_event_rate"):
            with self.subTest(key=key):
                self.assertIsNone(result[key])
        bootstrap = metrics.paired_episode_bootstrap({"a": result}, {"a": result}, replicates=5)
        self.assertEqual(5, bootstrap["discarded_unevaluable_replicates"])
        self.assertEqual(0, bootstrap["evaluable_replicates"])
        self.assertIsNone(bootstrap["ci95"])
        self.assertIsNone(bootstrap["delta_event_f1"])
        self.assertFalse(bootstrap["candidate_significant_win"])

    def test_bootstrap_reports_partially_unevaluable_resamples(self) -> None:
        empty = metrics.evaluate_episode(episode([False, False], [False, False]))
        good = metrics.evaluate_episode(episode([True, False], [True, False]))
        values = {"empty": empty, "good": good}
        result = metrics.paired_episode_bootstrap(values, values, replicates=100)
        self.assertGreater(result["discarded_unevaluable_replicates"], 0)
        self.assertGreater(result["evaluable_replicates"], 0)
        self.assertEqual(100, result["discarded_unevaluable_replicates"] + result["evaluable_replicates"])
        self.assertEqual([0, 0], result["ci95"])


if __name__ == "__main__":
    unittest.main()
