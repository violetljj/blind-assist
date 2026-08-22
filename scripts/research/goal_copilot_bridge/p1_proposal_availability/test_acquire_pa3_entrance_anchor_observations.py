from __future__ import annotations

import unittest
from unittest.mock import Mock

from scripts.research.goal_copilot_bridge.p1_proposal_availability.acquire_pa3_entrance_anchor_observations import osm_map_entrance_nodes, resolve_entrance


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

    def test_osm_map_backend_extracts_only_in_radius_entrance_nodes(self) -> None:
        response = Mock()
        response.content = b'''<osm>
          <node id="1" lat="0.0" lon="0.0005"><tag k="entrance" v="main"/></node>
          <node id="2" lat="0.0" lon="0.0020"><tag k="entrance" v="yes"/></node>
          <node id="3" lat="0.0" lon="0.0004"><tag k="amenity" v="bench"/></node>
        </osm>'''
        response.raise_for_status = Mock()
        session = Mock()
        session.get.return_value = response
        rows = osm_map_entrance_nodes(session, {"lat": 0.0, "lon": 0.0}, 120, "https://example.test/map")
        self.assertEqual([1], [row["id"] for row in rows])
        session.get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
