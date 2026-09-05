from __future__ import annotations

from copy import deepcopy
import unittest

from tools import knowledge


class HistoricalScopeTest(unittest.TestCase):
    def mechanism(self, history=None):
        return {
            "id": "paper-temporal#belief",
            "name": "Temporal correspondence",
            "description": "Temporal correspondence for proposal generation.",
            "route_history": history,
            "score": 20,
        }

    def terminal(self):
        return {
            "kind": "current_terminal",
            "id": "old-identity-test",
            "status": "closed",
            "question": "Temporal correspondence for identity ownership.",
            "decision": "IDENTITY_GATE_NOT_MET",
            "inheritance_role": "DEAD_FOR_THIS_ROLE",
            "role_scope": "v1 x identity owner x indoor paired views",
            "failure_signature": "Wrong identity despite local correspondence.",
            "revisit_trigger": "A separately justified target-conditioned hypothesis.",
            "successor_requires": "Historical suggestion: use a new sensor.",
            "forbidden_repeats": ["Retune temporal correspondence on sealed outcomes."],
            "evidence": ["old-result.json"],
            "terminal_markers": ["IDENTITY_GATE_NOT_MET"],
            "do_not_repeat": True,
        }

    def plan(self, mechanisms, attempts):
        return knowledge._build_minimum_experiment(
            {"failure_layers": [], "global_guardrails": ["No protected outcome access."]},
            "smoke-route", "temporal correspondence", {"layers": []},
            mechanisms, attempts,
        )

    def test_keyword_collision_keeps_candidate_and_scoped_failure_evidence(self):
        mechanisms = [self.mechanism(), {**self.mechanism(), "id": "other-candidate"}]
        terminal = self.terminal()
        before = deepcopy((mechanisms, terminal))
        plan = self.plan(mechanisms, [terminal])
        self.assertEqual(mechanisms[0]["id"], plan["selected_mechanism"]["id"])
        self.assertEqual("proposal", plan["status"])
        self.assertEqual([], plan["blocked_candidates"])
        note = plan["history_scope_notes"][0]
        self.assertEqual("unassessed", note["applicability"])
        for field in ("role_scope", "failure_signature", "forbidden_repeats",
                      "revisit_trigger", "evidence", "inheritance_role"):
            self.assertEqual(terminal[field], note[field])
        self.assertEqual(terminal["successor_requires"], note["historical_successor"])
        self.assertEqual(["IDENTITY_GATE_NOT_MET"], plan["prior_terminals_to_preserve"])
        self.assertEqual(["No protected outcome access."], plan["guardrails"])
        self.assertEqual(before, (mechanisms, terminal))

    def test_failure_or_retirement_labels_do_not_ban_a_future_proposal(self):
        for state, verdict in (("rejected", "negative"), ("retired", "positive"),
                               ("candidate", "falsified"), ("candidate", "not_evaluable")):
            with self.subTest(state=state, verdict=verdict):
                history = {
                    "id": "old-use", "use_state": state, "verdict": verdict,
                    "claim_boundary": "Only the old version on the old source.",
                    "evidence": ["source-receipt.json"],
                }
                mechanism = self.mechanism(history)
                plan = self.plan([mechanism], [])
                self.assertEqual(mechanism["id"], plan["selected_mechanism"]["id"])
                self.assertEqual("proposal", plan["status"])
                note = plan["history_scope_notes"][0]
                self.assertEqual(verdict, note["historical_outcome"])
                self.assertEqual(history["claim_boundary"], note["role_scope"])
                self.assertEqual(history["evidence"], note["evidence"])

    def test_not_evaluable_history_stays_an_evidence_gap(self):
        terminal = self.terminal()
        terminal.update(status="not_evaluable", decision="SOURCE_NOT_EVALUABLE",
                        inheritance_role="NEGATIVE_CONTROL",
                        terminal_markers=["SOURCE_NOT_EVALUABLE"])
        plan = self.plan([self.mechanism()], [terminal])
        self.assertEqual("proposal", plan["status"])
        note = plan["history_scope_notes"][0]
        self.assertEqual("SOURCE_NOT_EVALUABLE", note["historical_outcome"])
        self.assertEqual("NEGATIVE_CONTROL", note["inheritance_role"])
        self.assertEqual("unassessed", note["applicability"])

    def test_no_retrieval_hit_does_not_force_the_historical_successor(self):
        plan = self.plan([], [self.terminal()])
        self.assertEqual("hypothesis_needed", plan["status"])
        self.assertIsNone(plan["selected_mechanism"])
        self.assertNotIn("new sensor", plan["single_change"])
        self.assertEqual([self.terminal()["successor_requires"]], plan["successor_requirements"])

    def test_ranked_history_does_not_infer_a_repeat_prohibition(self):
        history = {
            "id": "old-use", "use_state": "rejected", "verdict": "not_evaluable",
            "observed_effect": "SOURCE_NOT_EVALUABLE", "expected_effect": "Support identity.",
            "claim_boundary": "v1 paired-view source only.", "metrics": [], "evidence": [],
        }
        terminal = {**self.terminal(), "routes": ["smoke-route"], "layer_scores": {},
                    "search_text": "temporal correspondence"}
        attempts = knowledge._rank_prior_attempts(
            {"experiments": [terminal]}, "smoke-route", "temporal correspondence",
            {"layers": []}, [self.mechanism(history)], 4,
        )
        self.assertEqual({"old-use", "old-identity-test"}, {row["id"] for row in attempts})
        self.assertTrue(all(row["do_not_repeat"] is None for row in attempts))
        plan = self.plan([self.mechanism(history)], attempts)
        self.assertEqual(2, len(plan["history_scope_notes"]))
        note = next(row for row in plan["history_scope_notes"] if row["id"] == "old-use")
        self.assertEqual(history["claim_boundary"], note["role_scope"])


if __name__ == "__main__":
    unittest.main()
