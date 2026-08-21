from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p0_s0_materialization import source_slice


class SourceSliceGeometryTest(unittest.TestCase):
    def test_containment_and_metric_boundary_distance(self) -> None:
        ring = [[3.72, 51.05], [3.7202, 51.05], [3.7202, 51.0502], [3.72, 51.0502], [3.72, 51.05]]
        inside = {"lon": 3.7201, "lat": 51.0501}
        self.assertTrue(source_slice.point_in_ring(inside, ring))
        distance = source_slice.point_to_ring_distance_m({"lon": 3.72, "lat": 51.0501}, ring)
        self.assertLess(distance, 0.001)

    def test_raw_degree_delta_is_not_used_as_meters(self) -> None:
        left = {"lon": 3.72, "lat": 51.05}
        right = {"lon": 3.7201, "lat": 51.05}
        self.assertGreater(source_slice.metric_distance_m(left, right), 6.0)
        self.assertLess(source_slice.metric_distance_m(left, right), 8.0)

    def test_overture_osm_source_record_bridge_is_explicit(self) -> None:
        feature = {"properties": {"sources": [
            {"dataset": "OpenStreetMap", "license": "ODbL-1.0", "record_id": "w426158361@2"},
            {"dataset": "Other", "license": "other", "record_id": "123"},
        ]}}
        self.assertEqual({"426158361"}, source_slice.overture_osm_way_ids(feature))
        summary = source_slice.overture_license_summary([feature])
        self.assertEqual(2, len(summary))


if __name__ == "__main__":
    unittest.main()
