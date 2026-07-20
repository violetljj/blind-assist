import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import evaluate_public_video_dual_evidence_prospective_lifecycle as evaluator
import public_video_dual_evidence_lifecycle_contract as prospective
import run_public_silver_frozen_feature_probe as common


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "public_video_dual_evidence_lifecycle_contract_r717.json"
)
CONTRACT, CONTRACT_ATTESTATION = prospective.load_contract(CONTRACT_PATH)
FEATURE_SHA = "f" * 64
INVENTORY_SHA = "e" * 64
VIDEO_SHA = "a" * 64


def fixture(*, weak_close: bool = False) -> tuple[dict, dict, dict]:
    pre = [0.10, 0.11, 0.10]
    risk = [0.10, 0.30, 0.50] if not weak_close else [0.10, 0.15, 0.20]
    post = [0.05, 0.06, 0.05] if not weak_close else [0.10, 0.148, 0.196]
    stable = [0.05, 0.051, 0.05] if not weak_close else [0.10, 0.147, 0.195]
    occupancies = pre + risk + post + stable
    samples = []
    for index, value in enumerate(occupancies):
        timestamp = index * 1000
        samples.append({
            "timestamp_ms": timestamp,
            "dynamic_occupancy": value,
            "static_residual": value,
            "static_residual_reliable": True,
            "semantic_risk_count": 1 if 3000 <= timestamp <= 5000 else 0,
        })
    features = {
        "schema": evaluator.FEATURE_SCHEMA,
        "prospective_contract": CONTRACT_ATTESTATION,
        "feature_generation": {
            "complete_video_processed": True,
            "review_windows_known_during_feature_generation": False,
            "feature_values_immutable": True,
            "sample_interval_ms": 1000,
            "dynamic_weights_sha256": CONTRACT["feature_contract"]["dynamic_channel"]["weights_sha256"],
            "semantic_weights_sha256": CONTRACT["feature_contract"]["semantic_exit_channel"]["weights_sha256"],
        },
        "sources": [{"source_id": "new-source", "video_sha256": VIDEO_SHA, "samples": samples}],
    }
    review = {
        "schema": evaluator.REVIEW_SCHEMA,
        "prospective_attestation": {
            "frozen_contract_sha256": CONTRACT_ATTESTATION["sha256"],
            "full_feature_report_sha256": FEATURE_SHA,
            "source_inventory_audit_sha256": INVENTORY_SHA,
            "contract_frozen_before_source_visual_review": True,
            "full_feature_report_frozen_before_visual_review": True,
            "reviewer_did_not_edit_feature_values": True,
            "original_temporal_order_reviewed": True,
            "hard_cut_detected": False,
        },
        "evidence": {
            "video_sha256": VIDEO_SHA,
            "contact_sheet_manifest_sha256": "b" * 64,
        },
        "review": {
            "source_id": "new-source",
            "decision": evaluator.ACCEPT_DECISION,
            "mechanism": "dynamic_agent_approach",
            "pre_risk_clear_window_ms": [0, 2000],
            "risk_present_window_ms": [3000, 5000],
            "stable_post_clear_window_ms": [6000, 11000],
            "visual_boundary": {
                "risk_present_timestamp_ms": 5000,
                "clear_timestamp_ms": 6000,
            },
        },
        "human_event_truth_present": False,
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    inventory = {
        "schema": evaluator.INVENTORY_SCHEMA,
        "sources": [{
            "source_id": "new-source",
            "video_sha256": VIDEO_SHA,
            "viewpoint": "pedestrian",
            "eligible_prospective_positive_exit": True,
            "eligible_prospective_pedestrian_exit": True,
        }],
    }
    return features, review, inventory


def run_fixture(value: tuple[dict, dict, dict]) -> dict:
    features, review, inventory = value
    return evaluator.evaluate(
        features=features,
        review=review,
        inventory_audit=inventory,
        contract=CONTRACT,
        contract_attestation=CONTRACT_ATTESTATION,
        feature_sha256=FEATURE_SHA,
        inventory_audit_sha256=INVENTORY_SHA,
    )


class DualEvidenceProspectiveLifecycleTest(unittest.TestCase):
    def test_valid_strong_open_and_close_pass(self) -> None:
        result = run_fixture(fixture())
        self.assertTrue(result["acceptance"]["passed"])
        self.assertEqual(result["transitions"]["open"]["predicted_transition"], "open_event")
        self.assertEqual(result["transitions"]["close"]["predicted_transition"], "close_event")

    def test_valid_weak_close_needs_and_uses_semantic_exit(self) -> None:
        result = run_fixture(fixture(weak_close=True))
        self.assertTrue(result["acceptance"]["passed"])
        self.assertEqual(result["transitions"]["close"]["reason"], "weak_relative_decrease_corroborated_by_semantic_exit")

    def test_weak_close_without_semantic_exit_fails(self) -> None:
        value = fixture(weak_close=True)
        for row in value[0]["sources"][0]["samples"]:
            row["semantic_risk_count"] = 0
        result = run_fixture(value)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertEqual(result["transitions"]["close"]["predicted_transition"], "uncertain")

    def test_feature_contract_substitution_is_rejected(self) -> None:
        value = fixture()
        value[0]["prospective_contract"] = {"path": "other", "sha256": "x" * 64}
        with self.assertRaisesRegex(ValueError, "not bound"):
            run_fixture(value)

    def test_review_must_attest_exact_feature_report(self) -> None:
        value = fixture()
        value[1]["prospective_attestation"]["full_feature_report_sha256"] = "x" * 64
        with self.assertRaisesRegex(ValueError, "feature report"):
            run_fixture(value)

    def test_lineage_ineligible_source_fails_acceptance(self) -> None:
        value = fixture()
        value[2]["sources"][0]["eligible_prospective_positive_exit"] = False
        result = run_fixture(value)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertFalse(result["acceptance"]["source_lineage_gate_passed"])

    def test_hard_cut_is_rejected(self) -> None:
        value = fixture()
        value[1]["prospective_attestation"]["hard_cut_detected"] = True
        with self.assertRaisesRegex(ValueError, "Hard-cut|hard-cut"):
            run_fixture(value)

    def test_visual_boundary_must_be_contained(self) -> None:
        value = fixture()
        value[1]["review"]["visual_boundary"]["clear_timestamp_ms"] = 12000
        result = run_fixture(value)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertFalse(result["acceptance"]["visual_boundary_contained"])

    def test_stable_clear_strong_reopen_fails(self) -> None:
        value = fixture()
        stable_values = [0.0, 0.4, 0.0]
        for row, new_value in zip(value[0]["sources"][0]["samples"][-3:], stable_values):
            row["dynamic_occupancy"] = new_value
        result = run_fixture(value)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertFalse(result["acceptance"]["stable_post_clear_no_reopen_passed"])

    def test_missing_full_video_sample_is_rejected(self) -> None:
        value = fixture()
        value[0]["sources"][0]["samples"].pop(4)
        with self.assertRaisesRegex(ValueError, "irregular"):
            run_fixture(value)

    def test_feature_generation_must_precede_known_review_windows(self) -> None:
        value = fixture()
        value[0]["feature_generation"]["review_windows_known_during_feature_generation"] = True
        with self.assertRaisesRegex(ValueError, "after review windows"):
            run_fixture(value)

    def test_contact_sheet_hash_must_be_real_hex(self) -> None:
        value = fixture()
        value[1]["evidence"]["contact_sheet_manifest_sha256"] = "z" * 64
        with self.assertRaisesRegex(ValueError, "contact sheet"):
            run_fixture(value)

    def test_run_verifies_sidecars_and_writes_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features, review, inventory = fixture()
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
                "full_feature_report_sha256": common.sha256_file(feature_path),
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
                contract=CONTRACT_PATH,
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
