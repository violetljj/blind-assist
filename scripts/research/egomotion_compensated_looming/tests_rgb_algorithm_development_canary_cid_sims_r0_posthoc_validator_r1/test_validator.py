from __future__ import annotations

import copy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.research.egomotion_compensated_looming.rgb_algorithm_development_canary_cid_sims_r0_posthoc_validator_r1.run import (
    EXPECTED_LOCK_PATHS,
    verify_implementation_lock,
)
from scripts.research.egomotion_compensated_looming.rgb_algorithm_development_canary_cid_sims_r0_posthoc_validator_r1.validator import (
    canonical_sha,
    digest_file,
    expected_window_timestamps,
    load_jsonl,
    load_object,
    validate,
    validate_invalid_r0_validation,
    validate_manifest_inventory,
    validate_result_and_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = REPO_ROOT / (
    "docs/research/rcle/"
    "RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_POSTHOC_VALIDATOR_R1_"
    "CONTRACT_2026-07-27.json"
)


class PosthocValidatorIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_object(CONTRACT_PATH)
        binding = {
            item["id"]: REPO_ROOT / item["path"]
            for item in cls.contract["immutable_r0_bindings"]
        }
        cls.r0_contract = load_object(binding["r0-contract"])
        cls.result = load_object(binding["r0-result"])
        cls.rows = load_jsonl(binding["r0-pair-ledger"])
        archive = REPO_ROOT / cls.contract["source"]["archive_path"]
        with zipfile.ZipFile(archive) as handle:
            cls.expected, errors = expected_window_timestamps(
                (item.filename for item in handle.infolist()),
                cls.contract["windows"],
                cls.contract["source"]["sequence_id"],
            )
        if errors:
            raise AssertionError(errors)

    def validate_rows(
        self,
        rows: list[dict],
        result: dict,
    ) -> list[str]:
        _, errors = validate_result_and_ledger(
            self.contract,
            self.r0_contract,
            result,
            rows,
            self.expected,
        )
        return errors

    def test_live_validator_passes(self) -> None:
        payload = validate(REPO_ROOT, CONTRACT_PATH)
        self.assertEqual(payload["status"], "VALID", payload["errors"])
        self.assertFalse(payload["algorithm_reexecution_performed"])
        self.assertEqual(
            payload["immutable_r0_evidence_status"],
            "INVALID_R0_EVIDENCE / INVALID",
        )

    def test_zip_omission_and_duplication_are_rejected(self) -> None:
        windows = [
            {
                "window_index": 0,
                "start_timestamp_s": "0",
                "end_timestamp_s": "1",
                "expected_frame_count": 3,
                "expected_pair_count": 2,
            },
            {
                "window_index": 1,
                "start_timestamp_s": "1",
                "end_timestamp_s": "2",
                "expected_frame_count": 3,
                "expected_pair_count": 2,
            },
        ]
        names = [
            f"floor3_1/{kind}/{timestamp}.png"
            for timestamp in ("0", "0.1", "0.2", "1", "1.1", "1.2")
            for kind in ("color", "depth")
        ]
        _, baseline = expected_window_timestamps(names, windows, "floor3_1")
        self.assertEqual(baseline, [])
        _, omission = expected_window_timestamps(
            [name for name in names if name != "floor3_1/depth/0.1.png"],
            windows,
            "floor3_1",
        )
        self.assertTrue(any("ZIP_WINDOW_FRAME_COUNT" in error for error in omission))
        _, duplication = expected_window_timestamps(
            [*names, "floor3_1/color/0.1.png"], windows, "floor3_1"
        )
        self.assertTrue(
            any("ZIP_COLOR_TIMESTAMP_DUPLICATE" in error for error in duplication)
        )

    def test_ledger_identity_mutations_are_rejected(self) -> None:
        mutations = {}
        rows = copy.deepcopy(self.rows)
        rows[0], rows[1] = rows[1], rows[0]
        mutations["record_order"] = rows
        rows = copy.deepcopy(self.rows)
        rows[0]["pair_index"] = 7
        mutations["pair_index"] = rows
        rows = copy.deepcopy(self.rows)
        rows[0]["previous_timestamp_s"] = Decimal("1673419222.281299")
        mutations["timestamp"] = rows
        rows = copy.deepcopy(self.rows)
        rows[0]["dt_s"] += Decimal("0.000001")
        mutations["dt"] = rows
        for name, candidate in mutations.items():
            with self.subTest(name=name):
                self.assertTrue(self.validate_rows(candidate, copy.deepcopy(self.result)))

    def test_trigger_and_abstention_mutations_are_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["trigger_threshold_per_s"] = Decimal("0.02")
        self.assertTrue(self.validate_rows(rows, copy.deepcopy(self.result)))
        rows = copy.deepcopy(self.rows)
        rows[0]["trigger"] = not rows[0]["trigger"]
        self.assertTrue(self.validate_rows(rows, copy.deepcopy(self.result)))
        rows = copy.deepcopy(self.rows)
        rows[0]["evaluable"] = False
        rows[0]["reason"] = ""
        rows[0]["trigger"] = False
        rows[0].pop("compensated_expansion_median_per_s", None)
        self.assertTrue(self.validate_rows(rows, copy.deepcopy(self.result)))

    def test_aggregate_terminal_and_authority_mutations_are_rejected(self) -> None:
        def refresh(value: dict) -> dict:
            value = copy.deepcopy(value)
            value.pop("result_payload_sha256", None)
            value["result_payload_sha256"] = canonical_sha(value)
            return value

        result = copy.deepcopy(self.result)
        result["windows"][0]["trigger_count"] += 1
        errors = self.validate_rows(copy.deepcopy(self.rows), refresh(result))
        self.assertIn("RESULT_WINDOW_FIELD:0:trigger_count", errors)
        result = copy.deepcopy(self.result)
        result["terminal"] = "FORGED / VALID"
        errors = self.validate_rows(copy.deepcopy(self.rows), refresh(result))
        self.assertIn("RESULT_PRODUCER_TERMINAL", errors)
        result = copy.deepcopy(self.result)
        result["separation"]["independent_confirmation"] = True
        errors = self.validate_rows(copy.deepcopy(self.rows), refresh(result))
        self.assertIn("RESULT_SEPARATION_AUTHORITY", errors)
        result = copy.deepcopy(self.result)
        result["archive_sha256"] = "0" * 64
        errors = self.validate_rows(copy.deepcopy(self.rows), refresh(result))
        self.assertIn("RESULT_ARCHIVE_SHA256", errors)

    def test_original_invalid_validation_inventory_is_exact(self) -> None:
        errors = [
            f"PAIR_TIMESTAMP:{window}:{pair}:dt_s"
            for window in (0, 1)
            for pair in range(299)
        ]
        value = {
            "status": "INVALID",
            "algorithm_reexecution_performed": False,
            "errors": errors,
        }
        self.assertEqual(validate_invalid_r0_validation(value), [])
        value["errors"][0] = "FORGED"
        self.assertTrue(validate_invalid_r0_validation(value))

    def test_immutable_binding_and_posthoc_authority_mutations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            value = copy.deepcopy(self.contract)
            value["immutable_r0_bindings"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            payload = validate(REPO_ROOT, path)
            self.assertIn("IMMUTABLE_BINDING:r0-contract", payload["errors"])
            value = copy.deepcopy(self.contract)
            value["authority"]["independent_confirmation_authorized"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            payload = validate(REPO_ROOT, path)
            self.assertIn("POSTHOC_AUTHORITY_ESCALATION", payload["errors"])


class CacheMutationTest(unittest.TestCase):
    def test_cache_mutations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            (cache / "color").mkdir(parents=True)
            timestamps = [
                Decimal("0"),
                Decimal("0.1"),
                Decimal("0.2"),
                Decimal("1"),
                Decimal("1.1"),
                Decimal("1.2"),
            ]
            expected = {
                0: timestamps[:3],
                1: timestamps[3:],
            }
            archive_path = root / "source.zip"
            with zipfile.ZipFile(archive_path, "w") as handle:
                for index, timestamp in enumerate(timestamps):
                    content = f"rgb-{timestamp}".encode()
                    handle.writestr(
                        f"floor3_1/color/{timestamp}.png", content
                    )
                    (cache / f"color/{timestamp}.png").write_bytes(content)
                control_content = b"groundtruth"
                handle.writestr("floor3_1/groundtruth.txt", control_content)
            (cache / "groundtruth.txt").write_bytes(control_content)
            with zipfile.ZipFile(archive_path) as handle:
                infos = {item.filename: item for item in handle.infolist()}
            members = []
            for ordinal, timestamp in enumerate(timestamps):
                path = cache / f"color/{timestamp}.png"
                members.append(
                    {
                        "archive_ordinal": ordinal,
                        "timestamp_s": str(timestamp),
                        "member_path": f"floor3_1/color/{timestamp}.png",
                        "cache_relative_path": f"color/{timestamp}.png",
                        "size_bytes": path.stat().st_size,
                        "sha256": digest_file(path),
                        "window_indices": [0 if timestamp < 1 else 1],
                    }
                )
            control_path = cache / "groundtruth.txt"
            manifest = {
                "schema_version": "rcle.rgb_algorithm_cache.v1",
                "protocol_id": (
                    "RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_"
                    "CID_SIMS_FLOOR3_1"
                ),
                "archive_sha256": "synthetic",
                "control": {
                    "member_path": "floor3_1/groundtruth.txt",
                    "cache_relative_path": "groundtruth.txt",
                    "size_bytes": control_path.stat().st_size,
                    "sha256": digest_file(control_path),
                },
                "members": members,
                "member_count": len(members),
                "network_request_count": 0,
                "downloaded_bytes": 0,
            }

            def refresh(value: dict) -> dict:
                value = copy.deepcopy(value)
                value.pop("payload_sha256", None)
                value["payload_sha256"] = canonical_sha(value)
                (cache / "manifest.json").write_text(
                    json.dumps(value), encoding="utf-8"
                )
                return value

            baseline = refresh(manifest)

            def errors(value: dict) -> list[str]:
                value = refresh(value)
                return validate_manifest_inventory(
                    value,
                    [],
                    expected,
                    infos,
                    cache,
                    "floor3_1",
                    "floor3_1/groundtruth.txt",
                    8,
                    "synthetic",
                )

            self.assertEqual(errors(baseline), [])
            candidate = copy.deepcopy(baseline)
            candidate["members"].pop()
            candidate["member_count"] -= 1
            self.assertTrue(errors(candidate))
            candidate = copy.deepcopy(baseline)
            candidate["members"][1] = copy.deepcopy(candidate["members"][0])
            self.assertTrue(
                any("DUPLICATE" in item for item in errors(candidate))
            )
            candidate = copy.deepcopy(baseline)
            candidate["members"][0]["size_bytes"] += 1
            self.assertTrue(any("CACHE_SIZE" in item for item in errors(candidate)))
            candidate = copy.deepcopy(baseline)
            candidate["members"][0]["sha256"] = "0" * 64
            self.assertTrue(
                any("CACHE_SHA256" in item for item in errors(candidate))
            )
            candidate = copy.deepcopy(baseline)
            candidate["members"][0], candidate["members"][1] = (
                candidate["members"][1],
                candidate["members"][0],
            )
            self.assertTrue(
                any("CACHE_MEMBER_FIELD" in item for item in errors(candidate))
            )
            candidate = copy.deepcopy(baseline)
            candidate["members"][0]["window_indices"] = [1]
            self.assertTrue(
                any("CACHE_MEMBER_FIELD" in item for item in errors(candidate))
            )
            candidate = copy.deepcopy(baseline)
            candidate["archive_sha256"] = "0" * 64
            self.assertIn("CACHE_MANIFEST_ARCHIVE_SHA256", errors(candidate))
            first_path = cache / baseline["members"][0]["cache_relative_path"]
            original = first_path.read_bytes()
            first_path.write_bytes(b"X" * len(original))
            self.assertTrue(any("CACHE_CRC" in item for item in errors(baseline)))


class RunnerLockMutationTest(unittest.TestCase):
    def test_runner_lock_and_activation_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in EXPECTED_LOCK_PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            contract_relative = next(
                path for path in EXPECTED_LOCK_PATHS if path.startswith("docs/")
            )
            contract_path = root / contract_relative
            lock_path = root / "IMPLEMENTATION_LOCK_R1.json"
            lock = {
                "schema_version": (
                    "rcle.rgb_algorithm_development_canary."
                    "posthoc_implementation_lock.v1"
                ),
                "protocol_id": (
                    "RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_"
                    "POSTHOC_VALIDATOR_R1"
                ),
                "status": "LOCKED_BEFORE_POSTHOC_VALIDATION",
                "files": [
                    {
                        "path": relative,
                        "sha256": digest_file(root / relative),
                    }
                    for relative in sorted(EXPECTED_LOCK_PATHS)
                ],
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            activation_path = root / "ACTIVATION_LOCK_R1.json"
            activation = {
                "schema_version": (
                    "rcle.rgb_algorithm_development_canary.posthoc_activation.v1"
                ),
                "protocol_id": lock["protocol_id"],
                "status": "AUTHORIZED_FOR_POSTHOC_VALIDATOR_ONLY",
                "implementation_lock_sha256": digest_file(lock_path),
                "contract_sha256": digest_file(contract_path),
                "maximum_authority": (
                    "POSTHOC_R0_IDENTITY_CACHE_LEDGER_AGGREGATE_AUDIT_ONLY"
                ),
                "algorithm_reexecution_authorized": False,
                "r0_evidence_revalidation_authorized": False,
                "outcome_blind_claim_authorized": False,
                "threshold_tuning_authorized": False,
                "independent_confirmation_authorized": False,
                "performance_qualification_authorized": False,
                "product_or_safety_claim_authorized": False,
                "network_access_authorized": False,
                "download_authorized": False,
            }
            activation_path.write_text(json.dumps(activation), encoding="utf-8")
            self.assertEqual(
                verify_implementation_lock(
                    root, contract_path, lock_path, activation_path
                ),
                [],
            )
            drift_relative = sorted(EXPECTED_LOCK_PATHS)[0]
            (root / drift_relative).write_text(
                "drift", encoding="utf-8"
            )
            self.assertTrue(
                any(
                    error.startswith("IMPLEMENTATION_FILE:")
                    for error in verify_implementation_lock(
                        root, contract_path, lock_path, activation_path
                    )
                )
            )
            (root / drift_relative).write_text(
                drift_relative, encoding="utf-8"
            )
            activation["implementation_lock_sha256"] = "0" * 64
            activation_path.write_text(json.dumps(activation), encoding="utf-8")
            self.assertIn(
                "ACTIVATION_IMPLEMENTATION_LOCK_SHA",
                verify_implementation_lock(
                    root, contract_path, lock_path, activation_path
                ),
            )
            activation["implementation_lock_sha256"] = digest_file(lock_path)
            mutations = {
                "status": ("FORGED", "ACTIVATION_STATUS"),
                "contract_sha256": (
                    "0" * 64,
                    "ACTIVATION_CONTRACT_SHA",
                ),
                "maximum_authority": (
                    "INDEPENDENT_CONFIRMATION",
                    "ACTIVATION_AUTHORITY",
                ),
                "network_access_authorized": (
                    True,
                    "ACTIVATION_FORBIDDEN:network_access_authorized",
                ),
                "r0_evidence_revalidation_authorized": (
                    True,
                    "ACTIVATION_FORBIDDEN:r0_evidence_revalidation_authorized",
                ),
                "outcome_blind_claim_authorized": (
                    True,
                    "ACTIVATION_FORBIDDEN:outcome_blind_claim_authorized",
                ),
                "product_or_safety_claim_authorized": (
                    True,
                    "ACTIVATION_FORBIDDEN:product_or_safety_claim_authorized",
                ),
            }
            for field, (value, expected_error) in mutations.items():
                with self.subTest(activation_field=field):
                    candidate = copy.deepcopy(activation)
                    candidate[field] = value
                    activation_path.write_text(
                        json.dumps(candidate), encoding="utf-8"
                    )
                    self.assertIn(
                        expected_error,
                        verify_implementation_lock(
                            root,
                            contract_path,
                            lock_path,
                            activation_path,
                        ),
                    )
            duplicate_lock = copy.deepcopy(lock)
            duplicate_lock["files"].append(
                copy.deepcopy(duplicate_lock["files"][0])
            )
            lock_path.write_text(json.dumps(duplicate_lock), encoding="utf-8")
            activation["implementation_lock_sha256"] = digest_file(lock_path)
            activation_path.write_text(json.dumps(activation), encoding="utf-8")
            self.assertIn(
                "IMPLEMENTATION_LOCK_ALLOWLIST",
                verify_implementation_lock(
                    root, contract_path, lock_path, activation_path
                ),
            )


if __name__ == "__main__":
    unittest.main()
