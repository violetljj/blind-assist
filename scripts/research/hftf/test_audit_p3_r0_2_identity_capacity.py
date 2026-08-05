#!/usr/bin/env python3
"""Tests for the label-blind P3 R0.2 identity-capacity audit."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import audit_p3_r0_2_identity_capacity as audit


class IdentityCapacityAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.closure = self._json("closure.json", {"schema": "closure"})
        self.ledger = self._json("ledger.json", {"schema": "ledger"})
        self.development = self._json("development.json", {"schema": "development"})
        videos = []
        for index in range(12):
            role = "train" if index < 8 else "validation"
            videos.append({
                "video_id": f"dev-{index}", "role": role,
                "selected_frame_stems": [f"dev-{index}_{1.0 + frame * 0.1:.3f}" for frame in range(4)],
                "extracted": {"lowres_wide": [], "lowres_depth": [], "confidence": []},
            })
        for index in range(4):
            videos.append({
                "video_id": f"attempted-{index}", "role": "sealed_identity_only",
                "selected_frame_stems": [f"attempted-{index}_{1.0 + frame * 0.1:.3f}" for frame in range(4)],
                "extracted": {"lowres_wide": []}, "sealed_metric_assets_read": False,
            })
        self.arkit = self._json("arkit.json", {
            "schema": "blindassist_spatial_calibration_head_r1_scoped_media_manifest",
            "videos": videos,
        })
        self.bonn = []
        for index in range(2):
            folder = self.root / f"bonn-{index}"
            folder.mkdir()
            rgb_lines, depth_lines = [], []
            for frame in range(4):
                timestamp = 1.0 + frame * 0.1
                rgb = folder / f"rgb-{frame}.png"
                depth = folder / f"depth-{frame}.png"
                rgb.write_bytes(b"rgb" + bytes([frame]))
                depth.write_bytes(b"depth" + bytes([frame]))
                rgb_lines.append(f"{timestamp:.3f} {rgb.name}")
                depth_lines.append(f"{timestamp:.3f} {depth.name}")
            rgb_index = folder / "rgb.txt"
            depth_index = folder / "depth.txt"
            rgb_index.write_text("\n".join(rgb_lines), encoding="utf-8")
            depth_index.write_text("\n".join(depth_lines), encoding="utf-8")
            self.bonn.append({
                "parent_id": f"bonn-{index}",
                "rgb_index": self._binding(rgb_index),
                "depth_index": self._binding(depth_index),
            })
        self.protocol = {
            "schema": audit.PROTOCOL_SCHEMA,
            "auditor": {"sha256": audit.sha256_file(Path(audit.__file__))},
            "source_universe": {"arkitscenes": {"scoped_manifest": self.arkit}, "bonn_rgbd": self.bonn},
            "exclusions": {
                "r0_1_closure": self.closure, "legacy_p1_ledger": self.ledger,
                "r0_1_attempted_holdout_parent_ids": [f"attempted-{index}" for index in range(4)],
            },
            "existing_development_ancestry": self.development,
            "capacity_gate": {
                "minimum_train_parent_count": 8, "minimum_validation_parent_count": 4,
                "minimum_new_holdout_parent_count": 8, "preferred_new_holdout_parent_count": 12,
            },
        }
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _binding(self, path: Path) -> dict[str, str]:
        return {"path": path.relative_to(self.root).as_posix(), "sha256": audit.sha256_file(path)}

    def _json(self, name: str, value: dict) -> dict[str, str]:
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return self._binding(path)

    def test_reports_data_not_ready_without_labels(self) -> None:
        result = audit.build_result(self.root, self.protocol_path)
        self.assertEqual(audit.NOT_READY, result["terminal"])
        self.assertEqual(2, result["capacity"]["eligible_new_holdout_parent_count"])
        self.assertFalse(result["forbidden_fields_read"])

    def test_forbidden_label_field_fails_closed(self) -> None:
        bad = json.loads((self.root / "arkit.json").read_text())
        bad["videos"][0]["geometry_state"] = ["CLEAR"]
        bad_binding = self._json("bad-arkit.json", bad)
        protocol = copy.deepcopy(self.protocol)
        protocol["source_universe"]["arkitscenes"]["scoped_manifest"] = bad_binding
        bad_protocol = self.root / "bad-protocol.json"
        bad_protocol.write_text(json.dumps(protocol), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            audit.build_result(self.root, bad_protocol)

    def test_output_overwrite_is_forbidden(self) -> None:
        path = self.root / "result.json"
        path.write_text("existing", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overwrite forbidden"):
            audit.exclusive_write(path, {})


if __name__ == "__main__":
    unittest.main()
