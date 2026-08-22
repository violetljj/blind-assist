from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.acquire_pa3_entrance_anchor_observations import resolve_entrance


class ResolveEntranceTest(unittest.TestCase):
    def test_main_entrance_precedes_closer_yes_entrance(self) -> None:
        raw = [
            {"id": 1, "lat": 0.0, "lon": 0.00001, "tags": {"entrance": "yes"}},
            {"id": 2, "lat": 0.0, "lon": 0.00010, "tags": {"entrance": "main"}},
            {"id": 3, "lat": 0.0, "lon": 0.00001, "tags": {"entrance": "main", "access": "private"}},
        ]
        selected = resolve_entrance(raw, {"lat": 0.0, "lon": 0.0})
        self.assertEqual(2, selected["osm_node_id"])

    def test_returns_none_without_eligible_entrance(self) -> None:
        self.assertIsNone(resolve_entrance([
            {"id": 1, "lat": 0.0, "lon": 0.0, "tags": {"entrance": "yes", "access": "no"}},
        ], {"lat": 0.0, "lon": 0.0}))


if __name__ == "__main__":
    unittest.main()
