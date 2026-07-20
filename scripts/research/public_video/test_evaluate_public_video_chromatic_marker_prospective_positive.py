import copy
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import evaluate_public_video_chromatic_marker_prospective_positive as evaluator
import run_public_silver_frozen_feature_probe as common


CONTRACT_ATTESTATION = {"path": "contract.json", "sha256": "c" * 64}
FEATURE_SHA = "f" * 64
INVENTORY_SHA = "i" * 64
VIDEO_SHA = "v" * 64


def detection(active: bool) -> list[dict]:
    if not active:
        return []
    return [{
        "class_name": "traffic cone",
        "features": {
            "high_saturation_fraction": 0.3,
            "dark_fraction": 0.1,
        },
    }]


def fixture() -> tuple[dict, dict, dict, dict]:
    samples = [
        {
            "timestamp_ms": timestamp,
            "detections": detection(timestamp <= 3000),
        }
        for timestamp in range(0, 9000, 1000)
    ]
    contract = {
        "risk_evidence_policy": {
            "policy_id": "chromatic_construction_marker_v1",
            "target_classes": ["barricade", "traffic cone"],
            "detection_acceptance": "high_saturation_fraction > dark_fraction",
            "minimum_accepted_detections_per_active_frame": 1,
            "absolute_color_threshold_used": False,
            "geometry_gate_used": False,
        },
        "lifecycle": {
            "selected_groups": ["surface_material", "barrier_structure"],
            "entry_window_samples": 3,
            "entry_min_active_samples": 2,
            "clear_absent_samples": 5,
        },
        "acceptance": {
            "minimum_risk_present_active_fraction": 0.4,
            "maximum_stable_clear_active_fraction": 0.1,
        },
    }
    features = {
        "schema": evaluator.FEATURE_SCHEMA,
        "prospective_contract": CONTRACT_ATTESTATION,
        "sources": [{
            "source_id": "held_out_source",
            "video_sha256": VIDEO_SHA,
            "samples": samples,
        }],
    }
    inventory = {
        "schema": evaluator.INVENTORY_AUDIT_SCHEMA,
        "frozen_contract": CONTRACT_ATTESTATION,
        "sources": [{
            "source_id": "held_out_source",
            "video_sha256": VIDEO_SHA,
            "viewpoint": "pedestrian",
            "eligible_prospective_positive_exit": True,
            "eligible_prospective_pedestrian_exit": True,
        }],
    }
    review = {
        "schema": evaluator.REVIEW_SCHEMA,
        "prospective_attestation": {
            "frozen_contract_sha256": CONTRACT_ATTESTATION["sha256"],
            "feature_report_sha256": FEATURE_SHA,
            "source_inventory_audit_sha256": INVENTORY_SHA,
            "policy_frozen_before_visual_review": True,
            "original_temporal_order_reviewed": True,
            "hard_cut_detected": False,
        },
        "evidence": {"video_sha256": VIDEO_SHA},
        "review": {
            "source_id": "held_out_source",
            "decision": evaluator.ACCEPT_DECISION,
            "risk_present_window_ms": [0, 3000],
            "stable_clear_window_ms": [4000, 8000],
            "candidate_boundary": {
                "present_timestamp_ms": 3000,
                "absent_timestamp_ms": 4000,
            },
        },
    }
    return features, review, inventory, contract


def run_fixture(features: dict, review: dict, inventory: dict, contract: dict) -> dict:
    return evaluator.evaluate(
        features=features,
        review=review,
        inventory_audit=inventory,
        contract=contract,
        contract_attestation=CONTRACT_ATTESTATION,
        feature_sha256=FEATURE_SHA,
        inventory_audit_sha256=INVENTORY_SHA,
    )


class ChromaticProspectivePositiveTest(unittest.TestCase):
    def test_valid_held_out_pedestrian_exit_passes(self) -> None:
        result = run_fixture(*fixture())
        self.assertTrue(result["acceptance"]["passed"])
        self.assertEqual(result["lifecycle"]["terminal_state"], "clear")
        self.assertEqual(len(result["lifecycle"]["intervals"]), 1)

    def test_derivation_contaminated_source_fails_lineage_gate(self) -> None:
        features, review, inventory, contract = fixture()
        inventory["sources"][0]["eligible_prospective_positive_exit"] = False
        result = run_fixture(features, review, inventory, contract)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertFalse(result["acceptance"]["source_lineage_gate_passed"])

    def test_visual_boundary_not_contained_fails(self) -> None:
        features, review, inventory, contract = fixture()
        review["review"]["candidate_boundary"] = {
            "present_timestamp_ms": 1000,
            "absent_timestamp_ms": 2000,
        }
        result = run_fixture(features, review, inventory, contract)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertFalse(
            result["acceptance"]["exact_single_reference_containing_exit_passed"]
        )

    def test_stable_clear_false_activation_fails_even_without_reopen(self) -> None:
        features, review, inventory, contract = fixture()
        features["sources"][0]["samples"][6]["detections"] = detection(True)
        result = run_fixture(features, review, inventory, contract)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertFalse(result["acceptance"]["stable_clear_false_activation_passed"])

    def test_feature_contract_mismatch_is_rejected(self) -> None:
        features, review, inventory, contract = fixture()
        features["prospective_contract"] = {"path": "other", "sha256": "x" * 64}
        with self.assertRaisesRegex(ValueError, "not bound"):
            run_fixture(features, review, inventory, contract)

    def test_review_must_attest_frozen_feature_report(self) -> None:
        features, review, inventory, contract = fixture()
        review["prospective_attestation"]["feature_report_sha256"] = "x" * 64
        with self.assertRaisesRegex(ValueError, "feature report"):
            run_fixture(features, review, inventory, contract)

    def test_hard_cut_is_rejected(self) -> None:
        features, review, inventory, contract = fixture()
        review["prospective_attestation"]["hard_cut_detected"] = True
        with self.assertRaisesRegex(ValueError, "Hard-cut|hard-cut"):
            run_fixture(features, review, inventory, contract)

    def test_run_verifies_inputs_and_writes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = (
                Path(__file__).resolve().parents[3]
                / "configs"
                / "public_video_chromatic_marker_lifecycle_contract_r711.json"
            )
            _, contract_attestation = evaluator.prospective.load_contract(contract_path)
            features, review, inventory, _ = fixture()
            features["prospective_contract"] = contract_attestation
            inventory["frozen_contract"] = contract_attestation

            feature_path = root / "features.json"
            feature_path.write_text(json.dumps(features), encoding="utf-8")
            Path(str(feature_path) + ".sha256").write_text(
                common.sha256_file(feature_path) + "\n", encoding="ascii"
            )
            inventory_path = root / "inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            Path(str(inventory_path) + ".sha256").write_text(
                common.sha256_file(inventory_path) + "\n", encoding="ascii"
            )
            review["prospective_attestation"].update({
                "frozen_contract_sha256": contract_attestation["sha256"],
                "feature_report_sha256": common.sha256_file(feature_path),
                "source_inventory_audit_sha256": common.sha256_file(inventory_path),
            })
            review_path = root / "review.json"
            review_path.write_text(json.dumps(review), encoding="utf-8")
            Path(str(review_path) + ".sha256").write_text(
                common.sha256_file(review_path) + "\n", encoding="ascii"
            )
            output = root / "result.json"
            report = evaluator.run(argparse.Namespace(
                features=feature_path,
                review=review_path,
                inventory_audit=inventory_path,
                contract=contract_path,
                output=output,
            ))
            self.assertTrue(report["acceptance"]["passed"])
            self.assertTrue(output.is_file())
            self.assertEqual(
                Path(str(output) + ".sha256").read_text(encoding="ascii").strip(),
                common.sha256_file(output),
            )


if __name__ == "__main__":
    unittest.main()
