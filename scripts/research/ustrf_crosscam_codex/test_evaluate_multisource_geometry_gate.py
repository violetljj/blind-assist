from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from evaluate_multisource_geometry_gate import run
from contract import sha256_file


class MultisourceGeometryGateTest(unittest.TestCase):
    def test_gate_is_source_level_and_pexels_cannot_participate(self) -> None:
        root = Path(tempfile.mkdtemp())
        prereg = root / "prereg.json"
        events = []
        for index, expected_class in enumerate(("positive", "positive", "positive", "negative", "negative", "negative")):
            events.append({"event_id": f"e{index}", "source_id": f"s{index}",
                           "expected_class": expected_class, "window_ms": [index, index + 1]})
        prereg.write_text(json.dumps({
            "schema": "blindassist_ustrf_crosscam_geometry_multisource_preregistration_v1",
            "held_out_events": events,
            "stopping_rules": {"minimum_positive_sources_with_robust_inside": 2,
                               "maximum_negative_sources_with_robust_inside": 0,
                               "maximum_unresolved_sources": 1},
            "authority": {"human_event_truth_present": False, "metric_geometry_present": False,
                          "training_authorized": False, "u0_authority_granted": False,
                          "android_runtime_change_authorized": False,
                          "production_model_replacement_authorized": False},
        }), encoding="utf-8")
        sources = []
        for index, event in enumerate(events):
            robust = index in (0, 1)
            sources.append({**event, "status": "resolved", "unresolved_reason": None,
                            "profile_summaries": [{"uncertainty_frame_ratio": ratio, "inside_count": int(robust),
                                                   "outside_count": int(not robust), "uncertain_count": 0}
                                                  for ratio in (0.01, 0.02, 0.03)],
                            "detection_relation_counts": {"inside": int(robust), "outside": int(not robust), "uncertain": 0},
                            "robust_inside": robust,
                            "event_recall": int(robust) if event["expected_class"] == "positive" else None,
                            "false_alarm": robust if event["expected_class"] == "negative" else None})
        android = root / "android.json"
        android.write_text(json.dumps({
            "schema": "blindassist_ustrf_crosscam_multisource_android_output_v1",
            "preregistration_sha256": sha256_file(prereg), "uncertainty_frame_ratios": [0.01, 0.02, 0.03],
            "threshold_fit": False, "parameter_search": False, "thresholds_changed": False,
            "training_performed": False, "pexels_used_for_gate": False, "sources": sources,
        }), encoding="utf-8")
        result = run(Namespace(preregistration=prereg, android_output=android, output=root / "report.json"))
        self.assertTrue(result["passed"])
        self.assertEqual(2, result["gate_counts"]["positive_sources_with_robust_inside"])
        self.assertFalse(result["pooled_frame_average_reported"])


if __name__ == "__main__":
    unittest.main()
