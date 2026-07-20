import unittest

import build_public_video_actionability_manifest as subject


class ActionabilityManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = {
            "minimum_intervention_events": 2,
            "minimum_context_only_events": 2,
            "minimum_independent_intervention_sources": 2,
            "minimum_independent_context_only_sources": 2,
            "maximum_event_class_ratio": 2.0,
        }

    @staticmethod
    def item(identifier: str, source: str, positive: bool) -> dict:
        return {
            "item_id": identifier,
            "parent_source_id": source,
            "intervention_required": positive,
            "label_basis": "frozen_current_or_past_only_causal_trace",
        }

    def test_balanced_independent_sources_pass(self) -> None:
        items = [
            self.item("p1", "ps1", True), self.item("p2", "ps2", True),
            self.item("n1", "ns1", False), self.item("n2", "ns2", False),
        ]
        checks, _ = subject.coverage_checks(items, self.gate)
        self.assertTrue(all(checks.values()))

    def test_duplicate_item_fails(self) -> None:
        items = [
            self.item("same", "ps1", True), self.item("same", "ps2", True),
            self.item("n1", "ns1", False), self.item("n2", "ns2", False),
        ]
        checks, _ = subject.coverage_checks(items, self.gate)
        self.assertFalse(checks["all_item_ids_unique"])

    def test_single_positive_source_fails(self) -> None:
        items = [
            self.item("p1", "ps1", True), self.item("p2", "ps1", True),
            self.item("n1", "ns1", False), self.item("n2", "ns2", False),
        ]
        checks, _ = subject.coverage_checks(items, self.gate)
        self.assertFalse(checks["minimum_independent_intervention_sources"])


if __name__ == "__main__":
    unittest.main()
