import json
import tempfile
import unittest
from pathlib import Path

import audit_public_video_prospective_source_inventory as audit
import run_public_silver_frozen_feature_probe as common


class ProspectiveSourceInventoryAuditTest(unittest.TestCase):
    def make_fixture(self, root: Path, *, positive: bool = True) -> tuple[Path, Path]:
        video = root / "video.webm"
        video.write_bytes(b"licensed-video-fixture")
        registry = root / "registry.json"
        registry.write_text(
            json.dumps({
                "schema": "blindassist_public_video_discovery_registry_v1",
                "sources": [{
                    "source_id": "fixture_source",
                    "local_video_path": str(video),
                }],
            }),
            encoding="utf-8",
        )
        contract = root / "contract.json"
        contract.write_text('{"contract_id":"fixture"}\n', encoding="utf-8")
        inventory = root / "inventory.json"
        inventory.write_text(
            json.dumps({
                "schema": audit.INVENTORY_SCHEMA,
                "frozen_contract_sha256": common.sha256_file(contract),
                "source_registry_paths": [str(registry)],
                "prospective_gate": {
                    "minimum_independent_positive_exit_sources": 1,
                    "minimum_pedestrian_positive_exit_sources": 1,
                    "forbidden_influence_contract_ids": ["r711"],
                },
                "sources": [{
                    "source_id": "fixture_source",
                    "local_video_path": str(video),
                    "video_sha256": common.sha256_file(video),
                    "item_level_license_usable": True,
                    "original_temporal_order": True,
                    "temporal_continuity": "continuous",
                    "viewpoint": "pedestrian",
                    "positive_exit_status": (
                        "held_out_positive_exit" if positive else "no_positive_exit_observed"
                    ),
                    "influenced_contract_ids": [],
                    "review_basis": "fixture",
                }],
            }),
            encoding="utf-8",
        )
        return inventory, contract

    def test_held_out_pedestrian_positive_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inventory, contract = self.make_fixture(Path(tmp))
            report = audit.audit_inventory(inventory, contract)
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(
            report["summary"]["eligible_prospective_positive_exit_source_count"], 1
        )

    def test_no_positive_keeps_gate_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inventory, contract = self.make_fixture(Path(tmp), positive=False)
            report = audit.audit_inventory(inventory, contract)
        self.assertFalse(report["gate"]["passed"])
        self.assertEqual(
            report["summary"]["eligible_prospective_positive_exit_source_count"], 0
        )

    def test_forbidden_derivation_influence_disqualifies_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inventory, contract = self.make_fixture(Path(tmp))
            data = json.loads(inventory.read_text(encoding="utf-8"))
            data["sources"][0]["influenced_contract_ids"] = ["r711"]
            inventory.write_text(json.dumps(data), encoding="utf-8")
            report = audit.audit_inventory(inventory, contract)
        self.assertFalse(report["gate"]["passed"])
        self.assertEqual(
            report["sources"][0]["forbidden_influence_contract_ids"], ["r711"]
        )

    def test_registry_coverage_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inventory, contract = self.make_fixture(Path(tmp))
            data = json.loads(inventory.read_text(encoding="utf-8"))
            data["sources"] = []
            inventory.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                audit.audit_inventory(inventory, contract)

    def test_video_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inventory, contract = self.make_fixture(Path(tmp))
            data = json.loads(inventory.read_text(encoding="utf-8"))
            data["sources"][0]["video_sha256"] = "0" * 64
            inventory.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "video hash mismatch"):
                audit.audit_inventory(inventory, contract)


if __name__ == "__main__":
    unittest.main()
