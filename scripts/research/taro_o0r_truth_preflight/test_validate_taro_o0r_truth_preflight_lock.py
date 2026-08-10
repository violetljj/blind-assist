from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.research.taro_o0r_truth_preflight.validate_taro_o0r_truth_preflight_lock import (
    canonical_sha256,
    expanded_requests,
    repo_root_from_validator,
    validate_document,
)


LOCK_PATH = Path(
    "docs/research/taro/"
    "TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_2026-08-10.json"
)


class TaroO0RTruthPreflightLockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = repo_root_from_validator()
        cls.lock = json.loads((cls.repo_root / LOCK_PATH).read_text(encoding="utf-8"))

    def errors_for(self, value: dict) -> list[str]:
        return validate_document(value, repo_root=self.repo_root, check_filesystem=False)

    def test_frozen_lock_is_valid_with_live_bindings_and_absent_roots(self) -> None:
        self.assertEqual(
            [],
            validate_document(self.lock, repo_root=self.repo_root, check_filesystem=True),
        )

    def test_expands_exactly_72_unique_training_head_targets(self) -> None:
        rows = expanded_requests(self.lock)
        self.assertEqual(72, len(rows))
        self.assertEqual(72, len({row["url"] for row in rows}))
        self.assertTrue(all(row["official_fold"] == "Training" for row in rows))
        self.assertEqual(
            self.lock["asset_plan"]["expanded_requests_sha256"],
            canonical_sha256(rows),
        )

    def test_parent_or_role_mutation_fails_closed(self) -> None:
        value = copy.deepcopy(self.lock)
        value["asset_plan"]["selected_parents"][0]["role"] = "O0R_EVAL_CANDIDATE"
        self.assertTrue(any("parent roster" in error for error in self.errors_for(value)))

    def test_url_template_mutation_fails_closed(self) -> None:
        value = copy.deepcopy(self.lock)
        value["asset_plan"]["asset_templates"][0]["url_template"] = (
            "https://example.test/{official_fold}/{video_id}.zip"
        )
        self.assertTrue(any("asset template" in error for error in self.errors_for(value)))

    def test_request_digest_cannot_be_self_resealed_after_template_mutation(self) -> None:
        value = copy.deepcopy(self.lock)
        value["asset_plan"]["asset_templates"][0]["url_template"] = (
            "https://example.test/{official_fold}/{video_id}.zip"
        )
        value["asset_plan"]["expanded_requests_sha256"] = canonical_sha256(
            expanded_requests(value)
        )
        self.assertTrue(any("asset template" in error for error in self.errors_for(value)))

    def test_authorization_gap_cannot_be_promoted_inside_lock(self) -> None:
        value = copy.deepcopy(self.lock)
        value["authorization_gate"]["source_body_execution_authorized"] = True
        self.assertTrue(any("authorization gate" in error for error in self.errors_for(value)))

    def test_network_or_truth_authority_mutation_fails_closed(self) -> None:
        for key in (
            "network_or_head_execution",
            "source_payload_download_or_open",
            "truth_materialization",
            "selected_source_uncertainty_fit",
            "depthart_inference",
            "factorial_execution",
        ):
            with self.subTest(key=key):
                value = copy.deepcopy(self.lock)
                value["execution_authority"][key] = True
                self.assertTrue(any("execution authority" in error for error in self.errors_for(value)))

    def test_root_or_budget_mutation_fails_closed(self) -> None:
        root_value = copy.deepcopy(self.lock)
        root_value["exclusive_roots"][0]["path"] += "-drift"
        self.assertTrue(any("root set" in error for error in self.errors_for(root_value)))

        budget_value = copy.deepcopy(self.lock)
        budget_value["resource_budget"]["maximum_compressed_source_bytes"] += 1
        self.assertTrue(any("resource budget" in error for error in self.errors_for(budget_value)))

        redirect_value = copy.deepcopy(self.lock)
        redirect_value["asset_plan"]["off_host_redirect_allowed"] = True
        self.assertTrue(any("off-host redirect" in error for error in self.errors_for(redirect_value)))

        one_shot_value = copy.deepcopy(self.lock)
        one_shot_value["one_shot_rule"]["rerun"] = True
        self.assertTrue(any("one-shot rule" in error for error in self.errors_for(one_shot_value)))


if __name__ == "__main__":
    unittest.main()
