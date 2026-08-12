from __future__ import annotations

import copy
import unittest

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r10_clear_runtime import run_top8_selection as runner


def feature(eligible: bool) -> dict:
    return {
        "query_receipt": {} if eligible else None,
        "r6_state": "UNKNOWN",
        "positive_obstacle_veto": False if eligible else None,
        "occupied_hits": [[[False]]] if eligible else None,
        "far_valid_anchor_count": 6 if eligible else 0,
        "far_fractions": [0.0, 0.0, 0.0] if eligible else None,
        "observed_support_points": 0,
    }


def synthetic_sources() -> list[dict]:
    rows = []
    for parent_index, ((parent_id, video_id), frame_count) in enumerate(
        zip(runner.EXPECTED_PARENT_IDENTITIES, runner.phase_a.FROZEN_FRAME_COUNTS, strict=True)
    ):
        eligible_count = parent_index % 13
        emitted = 0
        for frame_index in range(frame_count):
            queries = []
            for _ in range(9):
                allowed = emitted < eligible_count
                queries.append(feature(allowed))
                emitted += allowed
            token = f"{frame_index:06d}"
            rows.append(
                {
                    "parent_id": parent_id,
                    "video_id": video_id,
                    "timestamp_token": token,
                    "physical_frame_id": f"{video_id}:{token}",
                    "query_features": queries,
                    "source_phase_has_label_input": False,
                    "training_steps": 0,
                    "network_requests": 0,
                    "content_sha256": adapter.canonical_sha256([parent_id, video_id, token]),
                }
            )
    return rows


class Top8SelectionTests(unittest.TestCase):
    def test_phase_a_contract_is_exact_normal_r1(self) -> None:
        self.assertEqual(runner.PARENT_COUNT, 32)
        self.assertEqual(runner.FRAME_COUNT, 710)
        self.assertEqual(runner.QUERY_COUNT, 6390)
        self.assertEqual(runner.PHASE_A_FILE_COUNT, 3554)
        self.assertEqual(runner.PHASE_A_ROOT, "artifacts.local/evidence/taro/o1r-r10-fresh-pool-phase-a-r1")
        self.assertEqual(runner.phase_a_r1.PASS_TERMINAL, "TARO_O1R_R10_FRESH_POOL_PHASE_A_R1_SOURCE_ONLY_SEALED_PASS")

    def test_frozen_selector_and_rule_seals_reject_mutation(self) -> None:
        protocol = runner._read_json(runner._repo_path(runner.EXPECTED_BINDINGS["R10_PROTOCOL"]))
        self.assertEqual(runner.validate_protocol(protocol)["frozen_selector"], runner.FROZEN_PROTOCOL_SELECTOR)
        selector = runner._read_json(runner._repo_path(runner.EXPECTED_BINDINGS["R9_FROZEN_SELECTOR"]))
        validated = runner.validate_frozen_selector(selector)
        self.assertEqual(validated["content_sha256"], runner.FROZEN_SELECTOR_CONTENT_SHA256)
        self.assertEqual(validated["chosen_rule"]["rule_id"], runner.FROZEN_RULE_ID)

        mutated_selector = copy.deepcopy(selector)
        mutated_selector["chosen_rule"]["maximum_far_fraction"] = 0.05
        with self.assertRaises(runner.FreshTop8SelectionError):
            runner.validate_frozen_selector(mutated_selector)

        mutated_rule = copy.deepcopy(runner.FROZEN_RULE)
        mutated_rule["rule_id"] = "0000000000000000"
        with self.assertRaises(runner.FreshTop8SelectionError):
            runner.validate_frozen_rule(mutated_rule)

    def test_deterministic_32_to_8_and_zero_result_side_reads(self) -> None:
        sources = synthetic_sources()
        completion = {
            "content_sha256": adapter.canonical_sha256("completion"),
            "source_frame_hash_sequence_sha256": adapter.canonical_sha256([row["content_sha256"] for row in sources]),
        }
        scores, selection = runner.build_selection(completion, sources)
        ranked_forward = runner.rank_parent_scores(scores["parent_scores"])
        ranked_reverse = runner.rank_parent_scores(list(reversed(scores["parent_scores"])))

        self.assertEqual(
            [(row["parent_id"], row["video_id"]) for row in ranked_forward],
            [(row["parent_id"], row["video_id"]) for row in ranked_reverse],
        )
        self.assertEqual(len(selection["selected_parent_identities"]), 8)
        self.assertEqual(
            selection["selected_parent_identities"],
            [[row["parent_id"], row["video_id"]] for row in ranked_forward[:8]],
        )
        for record in (scores, selection, *scores["parent_scores"]):
            self.assertEqual(record["faro_reads"], 0)
            self.assertEqual(record["truth_reads"], 0)
            self.assertEqual(record["label_reads"], 0)
            self.assertEqual(record["outcome_reads"], 0)
            self.assertFalse(record["clear_output_emitted"])
        self.assertTrue(scores["all_32_scores_sealed_before_faro"])
        self.assertTrue(selection["selection_sealed_before_faro"])
        self.assertFalse(selection["unknown_is_negative"])

    def test_score_selector_mutation_is_rejected_even_with_resealed_parent_record(self) -> None:
        sources = synthetic_sources()
        completion = {
            "content_sha256": adapter.canonical_sha256("completion"),
            "source_frame_hash_sequence_sha256": adapter.canonical_sha256([row["content_sha256"] for row in sources]),
        }
        scores, _ = runner.build_selection(completion, sources)
        mutated = copy.deepcopy(scores)
        mutated["parent_scores"][0]["rule_id"] = "0000000000000000"
        mutated.pop("content_sha256")
        mutated["content_sha256"] = adapter.canonical_sha256(mutated)
        with self.assertRaises(runner.FreshTop8SelectionError):
            runner.validate_parent_scores(mutated)

    def test_public_api_has_no_result_side_parameter(self) -> None:
        runner.assert_public_api_source_only()
        self.assertFalse(runner.EXPECTED_AUTHORITY["faro_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["truth_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["label_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["outcome_read"])


if __name__ == "__main__":
    unittest.main()
