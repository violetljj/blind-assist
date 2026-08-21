from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as runner


class GroundingDinoS0R1Test(unittest.TestCase):
    def test_target_building_filter_keeps_only_requested_eligible_anchors(self) -> None:
        report = {
            "place_building_crosswalk_candidates": [
                {"status": "CANDIDATE_ONLY", "building_ids": ["b1"]},
                {"status": "CANDIDATE_ONLY", "building_ids": ["b2"]},
            ],
            "osm_entrance_building_crosswalk_candidates": [
                {"status": "CANDIDATE_ONLY", "entrance": "yes", "overture_building_id": "b1", "osm_entrance_id": "n1"},
                {"status": "CANDIDATE_ONLY", "entrance": "yes", "overture_building_id": "b2", "osm_entrance_id": "n2"},
            ],
        }
        selected = runner._eligible_target_anchors(report, ["b2"])
        self.assertEqual(["n2"], [item["osm_entrance_id"] for item in selected])
        with self.assertRaises(runner.RunError):
            runner._eligible_target_anchors(report, ["missing"])

    def test_source_bbox_is_read_from_each_slice(self) -> None:
        report = {
            "source_files": {"osm": {"bounds": {
                "minlon": "4.347", "minlat": "50.844", "maxlon": "4.355", "maxlat": "50.850",
            }}},
        }
        self.assertEqual((4.347, 50.844, 4.355, 50.850), runner.source_bbox(report))

    def test_source_bbox_rejects_inverted_slice(self) -> None:
        report = {"source_files": {"osm": {"bounds": {
            "minlon": "4.0", "minlat": "51.0", "maxlon": "3.0", "maxlat": "50.0",
        }}}}
        with self.assertRaises(runner.RunError):
            runner.source_bbox(report)

    def test_polygon_ring_accepts_current_overture_top_level_id_shape(self) -> None:
        feature = {
            "id": "building-1",
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[4.0, 51.0], [4.1, 51.0], [4.0, 51.0]]]},
            "properties": {"sources": []},
        }
        self.assertEqual(feature["geometry"]["coordinates"][0], runner._polygon_ring(feature))

    def test_nms_is_stable_and_class_agnostic(self) -> None:
        proposals = [
            {"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "score": 0.8, "label": "door"},
            {"bbox_xyxy": [1.0, 1.0, 10.0, 10.0], "score": 0.9, "label": "entrance"},
            {"bbox_xyxy": [20.0, 20.0, 30.0, 30.0], "score": 0.7, "label": "gate"},
        ]
        kept = runner.deterministic_nms(proposals)
        self.assertEqual(["entrance", "gate"], [item["label"] for item in kept])

    def test_metric_distance_is_not_raw_degree_distance(self) -> None:
        distance = runner.metric_distance([3.72, 51.05], [3.7201, 51.05])
        self.assertGreater(distance, 6.0)
        self.assertLess(distance, 8.0)

    def test_anchor_facing_selection_rejects_back_facing_view(self) -> None:
        anchor = {"osm_entrance_id": "node/1", "point": {"lon": 3.72, "lat": 51.0501}}
        base = {
            "coordinates": [3.72, 51.05], "camera_parameters": [0.5], "captured_at": 1,
            "sequence_id": "s1",
        }
        selected, counts = runner.select_anchor_facing_images(
            [dict(base, id="north", heading_deg=0.0), dict(base, id="south", heading_deg=180.0)],
            [anchor],
            requested_count=1,
        )
        self.assertEqual(["north"], [item["id"] for item in selected])
        self.assertEqual(1, counts["node/1"])

    def test_ray_segment_intersection(self) -> None:
        hit = runner._ray_segment((0.0, 0.0), (1.0, 0.0), (5.0, -2.0), (5.0, 2.0))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(5.0, hit[0])
        self.assertEqual((5.0, 0.0), hit[1])

    def test_generator_is_proposal_only_and_pinned(self) -> None:
        self.assertEqual("VISUAL_PROPOSAL_ONLY", runner.GENERATOR_AUTHORITY)
        self.assertEqual(64, len(runner.WEIGHTS_SHA256))
        self.assertIn("not a truth-authority gate", runner.TRAINING_PROVENANCE_LIMITATION)


if __name__ == "__main__":
    unittest.main()
