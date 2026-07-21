from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from contract import sha256_file
from evaluate_r11_attribution import run as run_attribution
from evaluate_target_oracle_geometry import run as run_oracle


FALSE_AUTHORITY = {"metric_geometry_present": False, "human_event_truth_present": False,
                   "training_authorized": False, "u0_authority_granted": False,
                   "android_runtime_change_authorized": False,
                   "production_model_replacement_authorized": False}


class R11DiagnosticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.ledger_path = self.root / "ledger.json"
        self.projection_path = self.root / "projection.json"
        self.ledger = {
            "schema": "blindassist_ustrf_crosscam_target_instance_ledger_v1",
            "diagnostic_set_role": "seen_diagnostic_not_held_out",
            "uncertainty_frame_ratios": [0.01, 0.02, 0.03], "authority": FALSE_AUTHORITY,
            "events": [{"event_id": "japan", "source_id": "source", "window_ms": [1000, 2000],
                        "target_instance": {"target_instance_id": "cone-1", "semantic_type": "traffic_cone",
                                            "expected_route_relation": "inside", "detector_label_allowlist": ["traffic cone"],
                                            "frames": [{"frame_id": "f1", "timestamp_ms": 1000, "frame_sha256": "a" * 64,
                                                        "frame_width": 1000, "frame_height": 500, "visibility": "visible",
                                                        "bbox_xyxy_norm": [0.45, 0.50, 0.55, 0.82],
                                                        "contact_xy_norm": [0.50, 0.82]}]}}]}
        self.ledger_path.write_text(json.dumps(self.ledger), encoding="utf-8")
        self.projection = {
            "schema": "blindassist_ustrf_crosscam_frame_projection_receipt_v2",
            "diagnostic_set_role": "seen_diagnostic_not_held_out", "target_ledger_sha256": sha256_file(self.ledger_path),
            "authority": FALSE_AUTHORITY,
            "events": [{"event_id": "japan", "projection_mode": "per_frame",
                        "frames": [{"frame_id": "f1", "timestamp_ms": 1000, "frame_sha256": "a" * 64,
                                    "status": "admitted", "route_polygon_xy_norm": [[0.3, 0.95], [0.7, 0.95], [0.65, 0.3], [0.35, 0.3]]}]}]}
        self.projection_path.write_text(json.dumps(self.projection), encoding="utf-8")

    def test_oracle_recomputes_target_contact_geometry(self) -> None:
        output = self.root / "oracle.json"
        result = run_oracle(Namespace(target_ledger=self.ledger_path, projection_receipt=self.projection_path, output=output))
        self.assertTrue(result["sources"][0]["oracle_geometry_passed"])
        self.assertEqual("inside", result["sources"][0]["frames"][0]["robust_relation"])

    def test_static_full_window_projection_is_rejected(self) -> None:
        self.projection["events"][0]["projection_mode"] = "static_full_window"
        self.projection_path.write_text(json.dumps(self.projection), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "static full-window projection is forbidden"):
            run_oracle(Namespace(target_ledger=self.ledger_path, projection_receipt=self.projection_path,
                                 output=self.root / "oracle.json"))

    def test_oracle_accepts_new_held_out_unscored_role(self) -> None:
        self.ledger["diagnostic_set_role"] = "new_held_out_unscored"
        self.ledger_path.write_text(json.dumps(self.ledger), encoding="utf-8")
        self.projection["diagnostic_set_role"] = "new_held_out_unscored"
        self.projection["target_ledger_sha256"] = sha256_file(self.ledger_path)
        self.projection_path.write_text(json.dumps(self.projection), encoding="utf-8")
        result = run_oracle(Namespace(target_ledger=self.ledger_path, projection_receipt=self.projection_path,
                                      output=self.root / "oracle-held-out.json"))
        self.assertTrue(result["sources"][0]["oracle_geometry_passed"])

    def test_attribution_separates_unsupported_taxonomy_and_cooccurrence_alert(self) -> None:
        oracle_path = self.root / "oracle.json"
        oracle = run_oracle(Namespace(target_ledger=self.ledger_path, projection_receipt=self.projection_path, output=oracle_path))
        android_path = self.root / "android.json"
        source = {"event_id": "japan", "source_id": "source", "target_instance_id": "cone-1",
                  "detector_coverage": {"status": "unsupported_taxonomy", "eligible_labels": []},
                  "visible_target_frame_count": 1, "target_match_frame_count": 0, "zero_detection_frame_count": 1,
                  "android_oracle_geometry_parity": None, "cooccurrence_runtime_alert_count": 0}
        android = {"schema": "blindassist_ustrf_crosscam_target_aware_android_output_v2",
                   "diagnostic_set_role": "seen_diagnostic_not_held_out",
                   "target_ledger_sha256": oracle["target_ledger_sha256"], "sources": [source]}
        android_path.write_text(json.dumps(android), encoding="utf-8")
        result = run_attribution(Namespace(oracle_output=oracle_path, android_output=android_path,
                                           output=self.root / "attribution.json"))
        self.assertEqual("detector_class_unsupported", result["sources"][0]["attribution"])

        source.update({"detector_coverage": {"status": "supported", "eligible_labels": ["traffic cone"]},
                       "target_match_frame_count": 1, "zero_detection_frame_count": 0,
                       "android_oracle_geometry_parity": True, "cooccurrence_runtime_alert_count": 3})
        android_path.write_text(json.dumps(android), encoding="utf-8")
        result = run_attribution(Namespace(oracle_output=oracle_path, android_output=android_path,
                                           output=self.root / "attribution-2.json"))
        self.assertEqual("risk_semantics_or_target_association", result["sources"][0]["attribution"])


if __name__ == "__main__":
    unittest.main()
