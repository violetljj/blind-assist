from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.core import (
    Attribution,
    ContractError,
    EpisodeState,
    FAIL_CLOSED_MESSAGE,
    HUMAN_ASSISTANCE_MESSAGE,
    Policy,
    State,
    adjudicate_episode,
    apply_observation,
    summarize_field_run,
)
from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.provider_adapter import _live_episode
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_silver_b_brain_baseline as brain


def observation(
    frame_id: str,
    index: int,
    *,
    status: str = "GROUNDED",
    x_min: float = 0.35,
    x_max: float = 0.65,
    y_min: float = 0.20,
    y_max: float = 0.60,
) -> dict:
    timestamp = 1_000 + index * 100
    candidate_id = f"candidate-{index}"
    evidence_id = f"evidence-{index}"
    grounded = status == "GROUNDED"
    decision = {
        "status": status,
        "selected_candidate_id": candidate_id if grounded else None,
        "ranked_candidate_ids": [candidate_id] if grounded else [],
        "source_frame_id": frame_id if grounded else None,
        "decision_timestamp_ms": timestamp + 10,
        "spatial_region": None,
        "goal_identity_support": "SUPPORTED" if grounded else "INSUFFICIENT",
        "spatial_support": "SUPPORTED" if grounded else "INSUFFICIENT",
        "confidence": 0.9 if grounded else None,
        "supporting_evidence_ids": [evidence_id] if grounded else [],
        "competing_candidate_ids": [],
        "abstention_reason": None if grounded else "NO_CANDIDATE",
        "persistence_handoff_token": (
            {
                "handoff_id": f"handoff-{index}",
                "candidate_id": candidate_id,
                "source_frame_id": frame_id,
                "spatial_region": {
                    "frame_id": frame_id,
                    "coordinate_space": "NORMALIZED_XYXY",
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                },
                "evidence_ids": [evidence_id],
            }
            if grounded
            else None
        ),
    }
    return {
        "schema_version": 1,
        "episode_id": "site-a-e01",
        "observation_id": f"observation-{index}",
        "frame_id": frame_id,
        "frame_sha256": f"{index:064x}",
        "captured_at_ms": timestamp,
        "processed_at_ms": timestamp + 10,
        "p0_output": {
            "schema_version": 1,
            "episode_id": "p0-independent-call",
            "provider_runs": [
                {
                    "provider_id": "frozen-current-p0",
                    "status": "RUN_SUCCESS",
                    "source_frame_ids": [frame_id],
                    "evidence_ids": [evidence_id] if grounded else [],
                    "candidate_ids": [candidate_id] if grounded else [],
                    "failure_reason": None,
                }
            ],
            "evidence": (
                [
                    {
                        "evidence_id": evidence_id,
                        "provider_id": "frozen-current-p0",
                        "evidence_type": "ENTRANCE_STRUCTURE",
                        "source_frame_id": frame_id,
                        "region": {
                            "frame_id": frame_id,
                            "coordinate_space": "NORMALIZED_XYXY",
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                        },
                        "confidence": 0.9,
                        "source_timestamp_ms": timestamp,
                        "expiry_timestamp_ms": timestamp + 1_000,
                        "identity_claim": {"target_name": "建筑 A", "relation": "entrance_of"},
                        "provenance": {
                            "implementation_id": "unchanged",
                            "config_id": "unchanged",
                            "source_kind": "RGB_PROVIDER",
                        },
                    }
                ]
                if grounded
                else []
            ),
            "candidates": (
                [
                    {
                        "candidate_id": candidate_id,
                        "region": {
                            "frame_id": frame_id,
                            "coordinate_space": "NORMALIZED_XYXY",
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                        },
                        "category_label": "building entrance",
                        "identity_hypothesis": "建筑 A",
                        "confidence": 0.9,
                        "provider_rank": 1,
                        "provider_ids": ["frozen-current-p0"],
                        "evidence_ids": [evidence_id],
                    }
                ]
                if grounded
                else []
            ),
            "decision": decision,
        },
    }


class RegroundingCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = EpisodeState.start(
            episode_id="site-a-e01", location_id="site-a", goal_name="建筑 A", started_at_ms=900
        )

    def test_candidate_is_current_frame_only_and_not_persisted(self) -> None:
        result = apply_observation(
            self.state,
            observation("frame-1", 1, x_min=0.02, x_max=0.30),
        )
        self.assertEqual(State.ADVANCE_AND_REOBSERVE.value, result.state.state)
        self.assertIn("左", result.instruction)
        self.assertEqual("candidate-1", result.event["candidate"]["candidate_id"])
        persisted = result.state.to_dict()
        self.assertFalse(any("candidate" in key or "region" in key or "identity" in key for key in persisted))
        self.assertNotIn("前方安全", result.message)
        with self.assertRaisesRegex(ContractError, "fresh observation"):
            apply_observation(result.state, observation("frame-1", 2))

    def test_arrival_requires_a_second_fresh_current_frame(self) -> None:
        first = apply_observation(
            self.state,
            observation("frame-1", 1, y_min=0.10, y_max=0.80),
        )
        self.assertEqual(State.ARRIVAL_CONFIRM.value, first.state.state)
        self.assertIsNone(first.state.completed_at_ms)
        second = apply_observation(
            first.state,
            observation("frame-2", 2, y_min=0.12, y_max=0.78),
        )
        self.assertEqual(State.COMPLETE.value, second.state.state)
        self.assertTrue(second.event["completion"])
        self.assertEqual("candidate-2", second.event["candidate"]["candidate_id"])

    def test_three_unreliable_observations_fail_closed_and_offer_human_exit(self) -> None:
        policy = Policy(max_consecutive_unreliable=3)
        result = None
        for index in range(1, 4):
            result = apply_observation(
                self.state,
                observation(f"frame-{index}", index, status="ABSTAIN_NO_RELIABLE_EVIDENCE"),
                policy,
            )
            self.assertIn(FAIL_CLOSED_MESSAGE, result.message)
        assert result is not None
        self.assertEqual(State.ABSTAIN.value, result.state.state)
        self.assertIn(HUMAN_ASSISTANCE_MESSAGE, result.message)
        self.assertEqual(3, result.state.rescan_count)

    def test_cross_frame_and_stale_support_are_recorded_fail_closed(self) -> None:
        cross_frame = observation("frame-1", 1)
        cross_frame["p0_output"]["evidence"][0]["source_frame_id"] = "old-frame"
        result = apply_observation(self.state, cross_frame)
        self.assertEqual(State.RESCAN.value, result.state.state)
        self.assertIn("current-frame-only", result.event["contract_error"])
        self.assertEqual(["INVALID_OUTPUT"], result.event["provider_failure_classes"])

        stale = observation("frame-2", 2)
        stale["p0_output"]["evidence"][0]["expiry_timestamp_ms"] = 1_000
        result = apply_observation(self.state, stale)
        self.assertEqual(State.RESCAN.value, result.state.state)
        self.assertIn("stale", result.event["contract_error"])

    def test_instruction_limit_stops_automatic_guidance(self) -> None:
        policy = Policy(max_instructions=1)
        first = apply_observation(
            self.state,
            observation("frame-1", 1, x_min=0.02, x_max=0.30),
            policy,
        )
        second = apply_observation(
            first.state,
            observation("frame-2", 2, x_min=0.70, x_max=0.98),
            policy,
        )
        self.assertEqual(State.ABSTAIN.value, second.state.state)
        self.assertEqual(HUMAN_ASSISTANCE_MESSAGE, second.message)

    def test_grounded_output_for_a_different_goal_fails_closed(self) -> None:
        wrong_goal = observation("frame-1", 1)
        wrong_goal["p0_output"]["evidence"][0]["identity_claim"]["target_name"] = "另一栋建筑"
        result = apply_observation(self.state, wrong_goal)
        self.assertEqual(State.RESCAN.value, result.state.state)
        self.assertIn("requested entrance goal", result.event["contract_error"])

    def test_actual_frozen_p0_output_shape_is_accepted(self) -> None:
        episode = _live_episode(
            episode_id="site-a-e01",
            goal_name="建筑 A",
            image_path=Path("unused.jpg"),
            frame_id="frame-live",
            captured_at_ms=1_000,
            proposals=[{"bbox_xyxy": [10.0, 10.0, 90.0, 90.0], "score": 0.8, "label": "entrance"}],
            width=100,
            height=100,
        )
        raw = {
            "episode_id": "site-a-e01",
            "model_case_id": "case-current",
            "action": "SELECT",
            "selected_candidate_ids": ["gdino-frame-live-001"],
            "confidence": 0.9,
            "rationale": "fixture",
        }
        frozen = brain._frozen_output(episode, raw, brain.POLICY_ID)
        result = apply_observation(
            self.state,
            {
                "schema_version": 1,
                "episode_id": "site-a-e01",
                "observation_id": "observation-live",
                "frame_id": "frame-live",
                "frame_sha256": "a" * 64,
                "captured_at_ms": 1_000,
                "processed_at_ms": 1_010,
                "p0_output": frozen,
            },
        )
        self.assertEqual(State.ARRIVAL_CONFIRM.value, result.state.state)
        self.assertIsNone(result.event["contract_error"])

    def test_false_confirmation_is_primary_and_forces_grounding_attribution(self) -> None:
        first = apply_observation(self.state, observation("frame-1", 1, y_min=0.1, y_max=0.8))
        complete = apply_observation(first.state, observation("frame-2", 2, y_min=0.1, y_max=0.8))
        summary = adjudicate_episode(
            complete.state,
            adjudicated_at_ms=1_300,
            false_entrance_confirmation=True,
            failure_attribution=Attribution.INTERACTION_OR_CONTROL_BOTTLENECK,
        )
        self.assertTrue(summary["false_entrance_confirmation"])
        self.assertEqual(Attribution.CURRENT_FRAME_GROUNDING_BOTTLENECK.value, summary["failure_attribution"])
        self.assertEqual(210, summary["first_discovery_time_ms"])
        self.assertEqual(310, summary["completion_time_ms"])

    def test_summary_requires_exactly_three_by_five(self) -> None:
        summaries = []
        for location_index in range(3):
            for episode_index in range(5):
                summaries.append(
                    {
                        "episode_id": f"site-{location_index}-e{episode_index}",
                        "location_id": f"site-{location_index}",
                        "terminal_state": "COMPLETE",
                        "false_entrance_confirmation": location_index == 0 and episode_index == 0,
                        "failure_attribution": (
                            Attribution.CURRENT_FRAME_GROUNDING_BOTTLENECK.value
                            if location_index == 0 and episode_index == 0
                            else Attribution.REGROUNDING_LOOP_MECHANICALLY_USEFUL.value
                        ),
                        "completion_time_ms": 10_000 + episode_index,
                        "first_discovery_time_ms": 1_000 + episode_index,
                        "instruction_count": 3,
                        "rescan_count": 1,
                    }
                )
        report = summarize_field_run(summaries)
        self.assertEqual("MECHANICAL_EXECUTION_COMPLETE", report["status"])
        self.assertEqual(1, report["primary_safety_metric"]["value"])
        self.assertEqual(1.0, report["task_completion_rate"]["value"])
        self.assertEqual(45, report["instruction_count"]["total"])
        self.assertEqual(15, report["rescan_count"]["total"])

        incomplete = summarize_field_run(copy.deepcopy(summaries[:-1]))
        self.assertEqual("MECHANICAL_EXECUTION_INCOMPLETE", incomplete["status"])


if __name__ == "__main__":
    unittest.main()
