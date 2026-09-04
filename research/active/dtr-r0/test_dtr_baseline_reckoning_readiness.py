from __future__ import annotations

import unittest

import dtr_baseline_reckoning_readiness as readiness


class BaselineReckoningReadinessTest(unittest.TestCase):
    def test_x21_event_metric_reconstruction(self) -> None:
        source = {
            "schema": "blindassist-dtr-x21-track-carried-component-ancestry-replay-v1",
            "metrics": {
                "X21": {
                    "contact_recall": 5,
                    "contact_events": 6,
                    "false_alert_segments": 11,
                    "event_f1": 5 / 11,
                    "median_first_alert_lead_s": 3.0,
                }
            },
        }
        row = readiness.normalize_jrdb_x21(source)[0]
        self.assertAlmostEqual(5 / 16, row["event_precision"])
        self.assertAlmostEqual(5 / 6, row["event_recall"])
        self.assertAlmostEqual(5 / 11, row["event_f1"])

    def test_panels_refuse_false_comparability(self) -> None:
        audit = readiness.build_audit(
            readiness.DEFAULT_CARLA,
            readiness.DEFAULT_JRDB_NATIVE,
            readiness.DEFAULT_JRDB_X21,
        )
        self.assertEqual(
            "PRELIMINARY_EXISTING_EVIDENCE_ONLY_NOT_BASELINE_RECKONING",
            audit["status"],
        )
        panels = {row["panel"] for row in audit["rows"]}
        self.assertEqual(
            {
                "CARLA_11_CONSUMED",
                "JRDB_27_NATIVE_TRACK_CEILING",
                "JRDB_6_DETECTOR_DERIVED",
            },
            panels,
        )
        self.assertTrue(
            audit["comparability"]["do_not_rank_jrdb_native_ceiling_against_detector_x21"]
        )


if __name__ == "__main__":
    unittest.main()
