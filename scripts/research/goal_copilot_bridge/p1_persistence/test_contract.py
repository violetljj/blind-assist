from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.research.goal_copilot_bridge.p1_persistence import baseline, evaluator


FIXTURES = Path(__file__).with_name("scenarios.json")


def episodes() -> list[dict]:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


class P1PersistenceContractTest(unittest.TestCase):
    def test_all_eight_scenarios_validate_and_replay_deterministically(self) -> None:
        fixtures = episodes()
        self.assertEqual(8, len(fixtures))
        first = evaluator.run_fixture_file(FIXTURES)
        second = evaluator.run_fixture_file(FIXTURES)
        self.assertEqual(first, second)
        self.assertTrue(first["valid"])

    def test_no_referent_hard_guard_never_binds_visible_door(self) -> None:
        episode = next(item for item in episodes() if item["scenario_class"] == "NO_REFERENT_GUARD")
        output = baseline.run_baseline(baseline.extract_public_input(episode))
        result = evaluator.evaluate_episode(episode, output)
        self.assertEqual(0, result["metrics"]["illegal_bind_frames"])
        self.assertIsNone(output["referent_id"])
        self.assertEqual({"UNBOUND"}, {frame["state"] for frame in output["frames"]})

    def test_output_cannot_create_a_referent_after_no_referent_handoff(self) -> None:
        episode = next(item for item in episodes() if item["scenario_class"] == "NO_REFERENT_GUARD")
        output = baseline.run_baseline(baseline.extract_public_input(episode))
        output["referent_id"] = "invented-target"
        result = evaluator.evaluate_episode(episode, output)
        self.assertFalse(result["valid_system_output"])
        self.assertIn("cannot create or replace", result["contract_error"])

    def test_output_cannot_replace_established_referent(self) -> None:
        episode = episodes()[0]
        output = baseline.run_baseline(baseline.extract_public_input(episode))
        output["referent_id"] = "different-physical-target"
        result = evaluator.evaluate_episode(episode, output)
        self.assertFalse(result["valid_system_output"])

    def test_nontracking_states_cannot_assert_current_location(self) -> None:
        episode = episodes()[0]
        output = baseline.run_baseline(baseline.extract_public_input(episode))
        output["frames"][1]["state"] = "UNCERTAIN"
        result = evaluator.evaluate_episode(episode, output)
        self.assertFalse(result["valid_system_output"])

    def test_candidate_truth_map_must_cover_exact_public_surface(self) -> None:
        episode = copy.deepcopy(episodes()[0])
        episode["truth"]["frames"][0]["candidate_instance_map"] = {}
        with self.assertRaises(evaluator.EpisodeContractError):
            evaluator.validate_episode(episode)

    def test_negative_evidence_can_reduce_tracking_to_uncertain(self) -> None:
        episode = copy.deepcopy(episodes()[0])
        episode["frames"][1]["candidates"][0].update({
            "identity_support": 0.70,
            "identity_contradiction": 0.80,
            "stability": 0.30,
            "oscillation": 0.70,
        })
        episode["truth"]["frames"][1]["allowed_states"] = ["UNCERTAIN"]
        output = baseline.run_baseline(baseline.extract_public_input(episode))
        self.assertEqual("UNCERTAIN", output["frames"][1]["state"])
        self.assertIsNone(output["frames"][1]["current_candidate_id"])

    def test_simple_baseline_exposes_safety_headroom(self) -> None:
        result = evaluator.run_fixture_file(FIXTURES)
        aggregate = result["aggregate"]
        self.assertTrue(aggregate["illegal_bind_rate_hard_gate_pass"])
        self.assertGreater(aggregate["wrong_instance_asserted_frames"], 0)
        self.assertGreater(aggregate["identity_switches"], 0)
        self.assertGreater(aggregate["false_reacquisitions"], 0)
        self.assertGreater(aggregate["wrong_lock_persistence_max_frames"], 0)
        self.assertEqual(1.0, aggregate["temporary_occlusion_recovery_rate"]["value"])

    def test_baseline_rejects_evaluator_truth_envelope(self) -> None:
        with self.assertRaisesRegex(ValueError, "without evaluator truth"):
            baseline.run_baseline(episodes()[0])

    def test_json_schemas_are_well_formed_and_bound_to_protocol(self) -> None:
        for name in ("p1_episode_schema.json", "p1_input_schema.json", "p1_output_schema.json"):
            schema = json.loads(Path(__file__).with_name(name).read_text(encoding="utf-8"))
            self.assertEqual("https://json-schema.org/draft/2020-12/schema", schema["$schema"])
            self.assertEqual("BA-P1-TARGET-PERSISTENCE-R0-V1", schema["properties"]["protocol_id"]["const"])

    def test_lexicographic_vector_prioritizes_identity_safety(self) -> None:
        result = evaluator.run_fixture_file(FIXTURES)
        aggregate = result["aggregate"]
        self.assertEqual(
            [
                aggregate["illegal_bind_frames"],
                aggregate["wrong_instance_asserted_frames"],
                aggregate["identity_switches"],
                aggregate["false_reacquisitions"],
                -aggregate["correct_identity_coverage"]["numerator"],
            ],
            aggregate["lexicographic_vector"],
        )


if __name__ == "__main__":
    unittest.main()
