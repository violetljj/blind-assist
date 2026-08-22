from __future__ import annotations

import unittest
from unittest.mock import Mock

from scripts.research.goal_copilot_bridge.p1_proposal_availability.acquire_pa3_entrance_anchor_observations import apply_geocoder_amendment, osm_map_entrance_nodes, resolve_entrance, resolve_parent_bound_entrance_xml, resolve_public_spatial_candidates_xml


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

    def test_parent_binding_rejects_neighbor_main_entrance(self) -> None:
        payload = b'''<osm>
          <node id="1" lat="0.0" lon="0.0"/>
          <node id="2" lat="0.0" lon="0.001"><tag k="entrance" v="yes"/></node>
          <node id="3" lat="0.001" lon="0.001"/>
          <node id="4" lat="0.001" lon="0.0"/>
          <node id="5" lat="0.0004" lon="0.0004"><tag k="entrance" v="main"/></node>
          <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="building" v="yes"/><tag k="name" v="Goal Library"/></way>
          <way id="20"><nd ref="5"/><tag k="building" v="yes"/><tag k="name" v="Neighbor Theater"/></way>
        </osm>'''
        entrance, parent, count = resolve_parent_bound_entrance_xml(payload, {
            "lat": 0.0005, "lon": 0.0005, "osm_type": "node", "osm_id": 99,
        })
        self.assertEqual(10, parent["osm_id"])
        self.assertEqual(2, entrance["osm_node_id"])
        self.assertEqual(1, count)

    def test_public_spatial_contract_keeps_entrance_set_and_uses_frontage_fallback(self) -> None:
        with_entrances = b'''<osm>
          <node id="1" lat="0.0" lon="0.0"><tag k="entrance" v="yes"/></node>
          <node id="2" lat="0.0" lon="0.001"><tag k="entrance" v="main"/></node>
          <node id="3" lat="0.001" lon="0.001"/>
          <node id="4" lat="0.001" lon="0.0"/>
          <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="building" v="yes"/></way>
        </osm>'''
        candidates, parent, count = resolve_public_spatial_candidates_xml(with_entrances, {
            "lat": 0.0005, "lon": 0.0005, "osm_type": "way", "osm_id": 10,
        })
        self.assertEqual(2, count)
        self.assertEqual(10, parent["osm_id"])
        self.assertEqual([2, 1], [candidate["osm_node_id"] for candidate in candidates])

        without_entrances = with_entrances.replace(b'<tag k="entrance" v="yes"/>', b'').replace(b'<tag k="entrance" v="main"/>', b'')
        fallbacks, _, count = resolve_public_spatial_candidates_xml(without_entrances, {
            "lat": 0.0005, "lon": 0.0005, "osm_type": "way", "osm_id": 10,
        }, maximum_frontage_fallbacks=2)
        self.assertEqual(0, count)
        self.assertEqual(2, len(fallbacks))
        self.assertTrue(all(candidate["candidate_type"] == "OSM_BUILDING_EDGE_MIDPOINT_FALLBACK" for candidate in fallbacks))

    def test_geocoder_amendment_changes_query_only_before_metadata_or_truth(self) -> None:
        queries = [{"episode_id": "goal-a", "query": "old"}, {"episode_id": "goal-b", "query": "stable"}]
        amendment = {
            "schema_version": "blindassist_p1_pa3_geocoder_query_amendment_v1",
            "original_plan_sha256": "a" * 64,
            "diagnostic": {
                "mapillary_metadata_accessed": False,
                "pixel_content_accessed": False,
                "truth_accessed": False,
                "provider_run": False,
            },
            "query_replacements": [{
                "episode_id": "goal-a", "effective_query": "new", "same_public_place_goal": True,
            }],
        }
        self.assertEqual([{"episode_id": "goal-a", "query": "new"}, queries[1]], apply_geocoder_amendment(queries, amendment, "a" * 64))
        amendment["diagnostic"]["pixel_content_accessed"] = True
        with self.assertRaisesRegex(ValueError, "pixel access"):
            apply_geocoder_amendment(queries, amendment, "a" * 64)


if __name__ == "__main__":
    unittest.main()
