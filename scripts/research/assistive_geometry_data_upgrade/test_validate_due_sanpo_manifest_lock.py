from __future__ import annotations

import copy
import unittest

from scripts.research.assistive_geometry_data_upgrade import validate_due_sanpo_manifest_lock as lock


class SanpoManifestLockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = lock.load_json(lock.PROTOCOL_PATH)
        bindings = cls.protocol["bindings"]
        cls.bootstrap = lock.load_json(lock.REPO_ROOT / bindings["bootstrap_receipt"]["path"])
        cls.real = lock.load_json(lock.REPO_ROOT / bindings["sanpo_real_manifest"]["path"])

    def test_full_lock_validates(self) -> None:
        result = lock.validate_protocol(self.protocol)
        self.assertEqual("VALID", result["status"])

    def test_bootstrap_payload_access_rejected(self) -> None:
        mutated = copy.deepcopy(self.bootstrap)
        mutated["payload_opened"] = True
        with self.assertRaisesRegex(lock.LockError, "opened payload"):
            lock.validate_bootstrap(mutated)

    def test_bootstrap_split_sha_drift_rejected(self) -> None:
        mutated = copy.deepcopy(self.bootstrap)
        mutated["sources"]["SANPO_REAL"]["official_train_split_sha256"] = "0" * 64
        with self.assertRaisesRegex(lock.LockError, "split SHA drift"):
            lock.validate_bootstrap(mutated)

    def test_nonzero_metadata_frame_assertion_rejected(self) -> None:
        mutated = copy.deepcopy(self.real)
        capability = next(iter(mutated["capabilities"].values()))
        capability["total_frames"] = 1
        capability["orientation_frame_counts"]["portrait"] = 1
        parent = mutated["identity_roster"]["parent_ids"][0]
        capability["parent_frame_counts"][parent] = 1
        with self.assertRaisesRegex(lock.LockError, "pre-payload frame count asserted"):
            lock.validate_manifest_lock(
                mutated,
                expected_source_id="sanpo_real_v0_train_discovery",
                expected_session_id="HfUrNcBq2jQ_Q5m5cp373cUtJPtFactP",
            )

    def test_metadata_cannot_be_validated_for_claim(self) -> None:
        mutated = copy.deepcopy(self.real)
        next(iter(mutated["capabilities"].values()))["quality_status"] = "VALIDATED_FOR_CLAIM"
        with self.assertRaisesRegex(lock.LockError, "metadata upgraded to claim truth"):
            lock.validate_manifest_lock(
                mutated,
                expected_source_id="sanpo_real_v0_train_discovery",
                expected_session_id="HfUrNcBq2jQ_Q5m5cp373cUtJPtFactP",
            )

    def test_selected_identity_drift_rejected(self) -> None:
        mutated = copy.deepcopy(self.bootstrap)
        mutated["sources"]["SANPO_SYNTHETIC"]["selected_session_id"] = "different"
        with self.assertRaisesRegex(lock.LockError, "selected identity drift"):
            lock.validate_bootstrap(mutated)

    def test_formal_prescreen_authority_stays_false(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["execution_authority"]["formal_source_prescreen"] = True
        with self.assertRaisesRegex(lock.LockError, "execution authority drift"):
            lock.validate_protocol(mutated)


if __name__ == "__main__":
    unittest.main()
