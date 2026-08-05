#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("materialize_p3_public_bonn_identity_inventory_r0.py")
SPEC = importlib.util.spec_from_file_location("bonn_identity", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BonnIdentityMaterializerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.data = self.root / "data"
        self.repo.mkdir()
        self.data.mkdir()
        self.archive = self.root / "archive.zip"
        self.archive.write_bytes(b"archive")
        self.basis = self.repo / "basis.json"
        self.basis.write_text("{}\n", encoding="utf-8")
        self.base = self.root / "base.json"
        self.base.write_text(json.dumps({
            "schema": MODULE.CATALOG_SCHEMA,
            "protocol_sha256": "A",
            "sources": [{"dataset_id": "bonn_rgbd_dynamic", "identity_inventory": []}],
            "runtime_state": {"model_outputs_read": False, "transition_labels_read": False, "candidate_performance_read": False},
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_sequence(self, name: str, missing_depth: bool = False) -> None:
        sequence = self.data / name
        (sequence / "rgb").mkdir(parents=True)
        (sequence / "depth").mkdir()
        rgb_rows, depth_rows = [], []
        for index in range(4):
            timestamp = 1.0 + index * 0.1
            rgb_rel = f"rgb/{index}.png"
            depth_rel = f"depth/{index}.png"
            (sequence / rgb_rel).write_bytes(f"rgb{index}".encode())
            if not (missing_depth and index == 3):
                (sequence / depth_rel).write_bytes(f"depth{index}".encode())
            rgb_rows.append(f"{timestamp:.1f} {rgb_rel}")
            depth_rows.append(f"{timestamp:.1f} {depth_rel}")
        (sequence / "rgb.txt").write_text("\n".join(rgb_rows) + "\n", encoding="utf-8")
        (sequence / "depth.txt").write_text("\n".join(depth_rows) + "\n", encoding="utf-8")

    def exclusions(self, ids: list[str]) -> Path:
        path = self.root / "exclusions.json"
        path.write_text(json.dumps({
            "schema": MODULE.EXCLUSION_SCHEMA,
            "basis": [{"path": "basis.json", "sha256": MODULE.sha256_file(self.basis)}],
            "excluded_parent_ids": ids,
            "selection_rule": "identity only",
            "forbidden_reads": ["labels"],
        }), encoding="utf-8")
        return path

    def test_materializes_complete_and_excluded_identities(self) -> None:
        self.make_sequence("a")
        self.make_sequence("b")
        catalog, receipt = MODULE.materialize(self.repo, self.data, self.archive, self.base, self.exclusions(["a"]))
        rows = catalog["sources"][0]["identity_inventory"]
        self.assertEqual([True, False], [row["ancestry_excluded"] for row in rows])
        self.assertTrue(all(row["four_frame_continuity_confirmed"] for row in rows))
        self.assertEqual(1, receipt["eligible_identity_count"])
        self.assertFalse(receipt["label_or_model_data_read"])

    def test_missing_referenced_depth_is_disclosed_and_not_admitted(self) -> None:
        self.make_sequence("a", missing_depth=True)
        catalog, receipt = MODULE.materialize(self.repo, self.data, self.archive, self.base, self.exclusions([]))
        identity = catalog["sources"][0]["identity_inventory"][0]
        self.assertFalse(identity["raw_metric_sensor_assets_present"])
        self.assertFalse(identity["four_frame_continuity_confirmed"])
        self.assertEqual(1, receipt["parents"][0]["missing_depth_reference_count"])
        self.assertEqual(3, receipt["parents"][0]["paired_rgb_depth_identity_count"])

    def test_member_escape_fails_closed(self) -> None:
        self.make_sequence("a")
        (self.data / "a" / "rgb.txt").write_text("1.0 ../escape.png\n1.1 rgb/1.png\n1.2 rgb/2.png\n1.3 rgb/3.png\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escaped sequence"):
            MODULE.materialize(self.repo, self.data, self.archive, self.base, self.exclusions([]))


if __name__ == "__main__":
    unittest.main()
