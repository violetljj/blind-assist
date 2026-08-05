#!/usr/bin/env python3
"""Tests for the P3 public RGB-D source-admission auditor."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import audit_p3_public_rgbd_source_admission_r0 as audit


class PublicSourceAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol = {
            "schema": audit.PROTOCOL_SCHEMA,
            "source_universe": {
                "bonn": {"official_url": "https://official.test/bonn", "official_origin": "https://official.test/", "parent_unit": "sequence", "higher_cluster_unit": "sequence"}
            },
            "capacity_gate": {"minimum_holdout_parents": 8, "target_holdout_parents": 12},
        }
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")
        self.catalog = {
            "schema": audit.CATALOG_SCHEMA,
            "protocol_sha256": audit.sha256_file(self.protocol_path),
            "sources": [self._source()],
            "runtime_state": {"model_outputs_read": False, "transition_labels_read": False, "candidate_performance_read": False},
        }
        self.catalog_path = self.root / "catalog.json"
        self.catalog_path.write_text(json.dumps(self.catalog), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _source(self) -> dict:
        return {
            "dataset_id": "bonn", "official_url": "https://official.test/bonn", "access_mode": "public",
            "license_reviewed": True, "parent_unit": "sequence", "higher_cluster_unit": "sequence",
            "published_parent_count": 26, "rgb_available": True, "source_native_timestamps_available": True,
            "intrinsics_available": True, "independent_metric_sensor_types": ["registered_rgbd"],
            "independent_metric_sensor_validity_available": True, "pose_available": True,
            "download_or_registration_state": "DIRECT_DOWNLOAD_AVAILABLE", "identity_inventory": [],
            "evidence": [{"official_url": "https://official.test/page", "claim": "RGB-D", "retrieved_on": "2026-08-05"}],
        }

    def test_preliminary_source_is_not_capacity_ready(self) -> None:
        result = audit.audit(self.protocol_path, self.catalog_path)
        self.assertFalse(result["capacity_ready"])
        self.assertEqual("IDENTITY_AUDIT_ELIGIBLE", result["admissions"][0]["status"])

    def test_forbidden_label_field_fails_closed(self) -> None:
        bad = copy.deepcopy(self.catalog)
        bad["sources"][0]["transition_counts"] = {}
        path = self.root / "bad.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            audit.audit(self.protocol_path, path)

    def test_official_url_and_overwrite_are_fail_closed(self) -> None:
        bad = copy.deepcopy(self.catalog)
        bad["sources"][0]["official_url"] = "https://other.test/"
        path = self.root / "url.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "URL drift"):
            audit.audit(self.protocol_path, path)
        output = self.root / "existing.json"
        output.write_text("existing", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            audit.write_new(output, {})


if __name__ == "__main__":
    unittest.main()
