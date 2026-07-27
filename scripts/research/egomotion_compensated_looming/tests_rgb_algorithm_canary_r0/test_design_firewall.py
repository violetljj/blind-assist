from __future__ import annotations

import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.egomotion_compensated_looming.rgb_algorithm_canary_r0.validate_design_package import (
    _canonical_manifest_hash,
    _inventory_firewall_violations,
    validate_design_objects,
    validate_synthetic_fixture,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_CONTRACT_2026-07-27.json"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "docs/research/rcle/RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_DATA_ROLE_MANIFEST_2026-07-27.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _member(ordinal: int, path: str, content: bytes) -> dict:
    return {
        "archive_ordinal": ordinal,
        "member_path": path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "synthetic_content_b64": base64.b64encode(content).decode("ascii"),
    }


def _record(window: int, pair: int, raw: float, compensated: float) -> dict:
    score = raw - compensated
    return {
        "window_index": window,
        "pair_index": pair,
        "raw_rotation_leakage_s_inverse": raw,
        "compensated_rotation_leakage_s_inverse": compensated,
        "paired_leakage_reduction_s_inverse": score,
        "raw_hex": raw.hex(),
        "compensated_hex": compensated.hex(),
        "score_hex": score.hex(),
        "evaluable": True,
        "abstention_reason": None,
    }


def _fixture() -> dict:
    cache = {
        "schema_version": "synthetic.cache.v1",
        "source_sha256": "1" * 64,
        "members": [
            _member(0, "rgb/0001.png", b"synthetic-one"),
            _member(1, "rgb/0002.png", b"synthetic-two"),
        ],
        "member_count": 2,
        "manifest_sha256": "",
    }
    cache["manifest_sha256"] = _canonical_manifest_hash(cache)
    records = [
        _record(0, 0, 0.04, 0.01),
        _record(0, 1, 0.05, 0.02),
    ]
    return {
        "schema_version": "rcle.rgb_algorithm_canary.synthetic_validator_fixture.v1",
        "synthetic_only": True,
        "cache_manifest": cache,
        "pair_records": records,
        "summary": {
            "evaluable_pair_count": 2,
            "score_sum": sum(
                item["paired_leakage_reduction_s_inverse"] for item in records
            ),
        },
        "progress_samples": [
            {
                "phase": "PRODUCER",
                "completed": 1,
                "total": 2,
                "throughput_per_s": 1.0,
                "eta_s": 1.0,
                "last_progress_at": "2026-07-27T00:00:01Z",
                "pid": 123,
                "input_sha256": "1" * 64,
                "implementation_sha256": "2" * 64,
                "status": "RUNNING",
            },
            {
                "phase": "PRODUCER",
                "completed": 2,
                "total": 2,
                "throughput_per_s": 1.0,
                "eta_s": 0.0,
                "last_progress_at": "2026-07-27T00:00:02Z",
                "pid": 123,
                "input_sha256": "1" * 64,
                "implementation_sha256": "2" * 64,
                "status": "RUNNING",
            },
        ],
    }


class DesignFirewallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = _load(CONTRACT_PATH)
        self.manifest = _load(MANIFEST_PATH)

    def test_current_design_package_is_valid(self) -> None:
        self.assertEqual(
            [], validate_design_objects(REPO_ROOT, self.contract, self.manifest)
        )

    def test_rejects_data_role_overlap(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0]["roles"].append("CONFIRMATION")
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertTrue(any("CONFIRMATION_ROLE_OVERLAP" in item for item in errors))

    def test_rejects_content_identity_drift(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0]["content_identity"] = "sha256:" + "f" * 64
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertTrue(any("PARTITION_POLICY" in item for item in errors))

    def test_rejects_independence_group_drift(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0]["independence_group"] = "FAKE_GROUP"
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertTrue(any("PARTITION_POLICY" in item for item in errors))

    def test_rejects_ancestry_drift(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0]["ancestry"] = ["FAKE_ANCESTOR"]
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertTrue(any("PARTITION_ANCESTRY" in item for item in errors))

    def test_rejects_access_level_drift(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0]["access_level"] = "FULL"
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertTrue(any("PARTITION_POLICY" in item for item in errors))

    def test_rejects_reuse_policy_drift(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0]["reuse_policy"] = "MAY_CONFIRM"
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertTrue(any("PARTITION_POLICY" in item for item in errors))

    def test_rejects_source_binding_inventory_truncation(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["source_bindings"] = mutated["source_bindings"][:1]
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertIn("SOURCE_BINDING_INVENTORY", errors)

    def test_rejects_content_identity_status_drift(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0]["content_identity_status"] = "NOT_BOUND"
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertTrue(any("PARTITION_POLICY" in item for item in errors))

    def test_rejects_identity_basis_drift(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0]["identity_basis"] = "UNVERIFIABLE"
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertTrue(any("PARTITION_POLICY" in item for item in errors))

    def test_rejects_future_execution_access_drift(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0]["future_execution_access"] = "FULL_NOW"
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertTrue(any("PARTITION_POLICY" in item for item in errors))

    def test_rejects_common_pair_ledger_binding_drift(self) -> None:
        mutated_contract = copy.deepcopy(self.contract)
        mutated_manifest = copy.deepcopy(self.manifest)
        for partition in mutated_contract["data_partitions"]:
            if "pair_ledger_sha256" in partition:
                partition["pair_ledger_sha256"] = "f" * 64
        for partition in mutated_manifest["partitions"]:
            if "pair_ledger_sha256" in partition:
                partition["pair_ledger_sha256"] = "f" * 64
        errors = validate_design_objects(
            REPO_ROOT, mutated_contract, mutated_manifest
        )
        self.assertTrue(any("PARTITION_POLICY" in item for item in errors))

    def test_rejects_partition_order_change(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["partitions"][0], mutated["partitions"][1] = (
            mutated["partitions"][1],
            mutated["partitions"][0],
        )
        errors = validate_design_objects(REPO_ROOT, self.contract, mutated)
        self.assertIn("PARTITION_ORDER", errors)

    def test_rejects_contract_manifest_conflict(self) -> None:
        mutated_contract = copy.deepcopy(self.contract)
        mutated_contract["data_partitions"][0]["pair_identity_sha256"] = "f" * 64
        errors = validate_design_objects(REPO_ROOT, mutated_contract, self.manifest)
        self.assertTrue(any("CONTRACT_MANIFEST_FIELD" in item for item in errors))

    def test_rejects_algorithm_outcome_leakage(self) -> None:
        mutated = _fixture()
        mutated["observed_window_score_s_inverse"] = 123.0
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("FIXTURE_OUTCOME_LEAKAGE" in item for item in errors))

    def test_rejects_algorithm_outcome_filename_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = (
                root
                / "artifacts.local/evidence/RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0"
            )
            target.mkdir(parents=True)
            (target / "Formal_Run_R0.CLAIM.JSON").write_text(
                "synthetic filename fixture", encoding="utf-8"
            )
            self.assertTrue(_inventory_firewall_violations(root))

    def test_rejects_missing_field(self) -> None:
        mutated = _fixture()
        del mutated["pair_records"][0]["raw_hex"]
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PAIR_FIELDS" in item for item in errors))

    def test_rejects_record_order_change(self) -> None:
        mutated = _fixture()
        mutated["pair_records"].reverse()
        errors = validate_synthetic_fixture(mutated)
        self.assertIn("PAIR_ORDER_OR_DUPLICATE", errors)

    def test_rejects_numeric_drift(self) -> None:
        mutated = _fixture()
        mutated["pair_records"][0]["paired_leakage_reduction_s_inverse"] += 0.001
        mutated["pair_records"][0]["score_hex"] = mutated["pair_records"][0][
            "paired_leakage_reduction_s_inverse"
        ].hex()
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PAIR_NUMERIC_DRIFT" in item for item in errors))

    def test_rejects_forged_summary(self) -> None:
        mutated = _fixture()
        mutated["summary"]["score_sum"] = 99.0
        errors = validate_synthetic_fixture(mutated)
        self.assertIn("SUMMARY_NUMERIC_FORGED", errors)

    def test_rejects_cache_member_tamper(self) -> None:
        mutated = _fixture()
        mutated["cache_manifest"]["members"][0]["synthetic_content_b64"] = base64.b64encode(
            b"tampered"
        ).decode("ascii")
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("CACHE_MEMBER_" in item for item in errors))

    def test_rejects_missing_progress_contract(self) -> None:
        mutated = _fixture()
        del mutated["progress_samples"][0]["eta_s"]
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PROGRESS_FIELDS" in item for item in errors))

    def test_rejects_progress_freshness(self) -> None:
        mutated = _fixture()
        mutated["progress_samples"][1]["last_progress_at"] = "2026-07-27T00:10:02Z"
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PROGRESS_FRESHNESS" in item for item in errors))

    def test_rejects_progress_timestamp_format(self) -> None:
        mutated = _fixture()
        mutated["progress_samples"][0]["last_progress_at"] = "not-rfc3339"
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PROGRESS_TIMESTAMP" in item for item in errors))

    def test_rejects_progress_phase(self) -> None:
        mutated = _fixture()
        mutated["progress_samples"][0]["phase"] = "UNDECLARED_PHASE"
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PROGRESS_PHASE" in item for item in errors))

    def test_rejects_progress_status(self) -> None:
        mutated = _fixture()
        mutated["progress_samples"][0]["status"] = "SECRET_SUCCESS"
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PROGRESS_STATUS" in item for item in errors))

    def test_rejects_progress_pid(self) -> None:
        mutated = _fixture()
        mutated["progress_samples"][1]["pid"] = 0
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PROGRESS_PID" in item for item in errors))

    def test_rejects_progress_eta(self) -> None:
        mutated = _fixture()
        mutated["progress_samples"][0]["eta_s"] = -1.0
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PROGRESS_ETA" in item for item in errors))

    def test_rejects_progress_input_hash_drift(self) -> None:
        mutated = _fixture()
        mutated["progress_samples"][1]["input_sha256"] = "3" * 64
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(any("PROGRESS_INPUT_HASH_DRIFT" in item for item in errors))

    def test_rejects_progress_implementation_hash_drift(self) -> None:
        mutated = _fixture()
        mutated["progress_samples"][1]["implementation_sha256"] = "4" * 64
        errors = validate_synthetic_fixture(mutated)
        self.assertTrue(
            any("PROGRESS_IMPLEMENTATION_HASH_DRIFT" in item for item in errors)
        )


if __name__ == "__main__":
    unittest.main()
