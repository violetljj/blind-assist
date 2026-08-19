from __future__ import annotations

import unittest

from scripts.research.l10m_b0.closure import VERDICT as B0_CLOSURE_VERDICT

from .evaluator import evaluate_spec, load_hidden_cohort
from .policy_space import (
    INITIAL_SPEC,
    all_specs,
    canonical_spec,
    changed_components,
    parse_raw,
    parse_structured,
    render_raw,
    render_structured,
)
from .protocol import PROTOCOL_ID, STATUS, build_protocol_manifest


class L10MB1ProtocolTest(unittest.TestCase):
    def test_b0_is_closed_and_b1_execution_is_not_started(self) -> None:
        result = build_protocol_manifest()
        self.assertEqual(result["protocol_id"], PROTOCOL_ID)
        self.assertEqual(result["status"], STATUS)
        self.assertEqual(result["parent_freeze"]["verdict"], B0_CLOSURE_VERDICT)
        self.assertFalse(result["parent_freeze"]["b0_semantics_mutable"])
        self.assertEqual(result["execution_boundary"]["model_calls_made"], 0)
        self.assertFalse(result["execution_boundary"]["large_scale_structured_search_authorized"])

    def test_raw_and_structured_interfaces_cover_exactly_the_same_space(self) -> None:
        specs = all_specs()
        self.assertEqual(len(specs), 162)
        for spec in specs:
            self.assertEqual(canonical_spec(parse_raw(render_raw(spec))), canonical_spec(spec))
            self.assertEqual(canonical_spec(parse_structured(render_structured(spec))), canonical_spec(spec))
        receipt = build_protocol_manifest()["candidate_space"]["equivalence_receipt"]
        self.assertTrue(receipt["same_initial_candidate"])
        self.assertTrue(receipt["same_canonical_space"])

    def test_candidate_cannot_break_frozen_terminal_unknown_or_safety_invariants(self) -> None:
        cohort = load_hidden_cohort()
        for spec in all_specs():
            result = evaluate_spec(spec, cohort)
            self.assertTrue(result["semantic_valid"], spec)
            self.assertFalse(result["unsafe_candidate"], spec)
            self.assertEqual(result["invariant_counts"]["terminal_violations"], 0, spec)
            self.assertEqual(result["invariant_counts"]["unknown_stuck_increments"], 0, spec)

    def test_component_ledger_reports_mechanism_location(self) -> None:
        candidate = INITIAL_SPEC.__class__(
            action_selection_turn_threshold=0.10,
            fallback_min_quality=INITIAL_SPEC.fallback_min_quality,
            fallback_action=INITIAL_SPEC.fallback_action,
            stuck_response=INITIAL_SPEC.stuck_response,
            recovery_transition_action=INITIAL_SPEC.recovery_transition_action,
        )
        self.assertEqual(changed_components(INITIAL_SPEC, candidate), ["action_selection"])


if __name__ == "__main__":
    unittest.main()
