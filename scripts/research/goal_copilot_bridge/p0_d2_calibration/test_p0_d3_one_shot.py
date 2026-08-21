from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p0_d2_calibration import p0_d3_one_shot as d3


def polygon(size: float = 0.0003) -> dict:
    return {"type": "Polygon", "coordinates": [[[0, 0], [size, 0], [size, size], [0, size], [0, 0]]]}


def frozen_slice(city: str, index: int) -> dict:
    building_id, place_id = f"building-{index}", f"place-{index}"
    return {
        "city": city,
        "source_report": {
            "overture_release": "2026-08-19.0", "report_sha256": f"report-{index}",
            "source_files": {"overture_places": {"sha256": f"p-{index}"}, "overture_buildings": {"sha256": f"b-{index}"}},
            "place_building_crosswalk_candidates": [{
                "status": "CANDIDATE_ONLY", "place_id": place_id, "building_ids": [building_id],
            }],
            # If this list accidentally affects selection, deterministic equality below changes.
            "osm_entrance_building_crosswalk_candidates": [{"entrance": "service", "overture_building_id": building_id}],
        },
        "places": {"features": [{
            "id": place_id, "geometry": {"type": "Point", "coordinates": [0.0001, 0.0001]},
            "properties": {"names": {"primary": f"Venue {index}"}, "confidence": 0.9 + index / 100,
                           "basic_category": "shopping_mall", "taxonomy": {"hierarchy": ["shopping", "shopping_mall"], "primary": "shopping_mall"}},
        }]},
        "buildings": {"features": [{"id": building_id, "geometry": polygon()}]},
    }


class D3RosterTest(unittest.TestCase):
    def test_freezes_six_without_using_osm_entrances(self) -> None:
        slices = [frozen_slice(city, index) for index, city in enumerate(("Antwerp", "Bruges", "Brussels", "Ghent", "Leuven", "Mechelen"))]
        first = d3.plan_roster(slices, [], [])
        for item in slices:
            item["source_report"]["osm_entrance_building_crosswalk_candidates"] = []
        second = d3.plan_roster(slices, [], [])
        self.assertEqual(first, second)
        self.assertEqual(6, len(first["parents"]))
        self.assertFalse(first["batch_semantics"]["replacement_allowed"])
        self.assertFalse(first["batch_semantics"]["second_batch_allowed"])
        self.assertFalse(first["criteria"]["osm_entrance_fields_used"])

    def test_excludes_consumed_name_and_roster_identity(self) -> None:
        slices = [frozen_slice(city, index) for index, city in enumerate(("Antwerp", "Bruges", "Brussels", "Ghent", "Leuven", "Mechelen"))]
        cohort = {"episodes": [{"target_building_id": "unused", "evaluator_episode": {"goal_spec": {"target_name": "Venue 0"}}}]}
        roster = {"parents": [{"building_id": "building-1", "place_id": "place-1", "place_name": "Venue 1"}]}
        with self.assertRaisesRegex(d3.D3Error, "insufficient"):
            d3.plan_roster(slices, [cohort], [roster])

    def test_metadata_frame_selection_is_sequence_diverse_and_spaced(self) -> None:
        target = {"lon": 0.0, "lat": 0.0}
        raw = []
        for index, (lon, sequence) in enumerate(((0.0001, "a"), (0.00011, "a"), (0.0002, "b"), (0.0003, "c"), (0.0004, "d"))):
            raw.append({
                "id": str(index), "computed_geometry": {"coordinates": [lon, 0.0]}, "computed_compass_angle": 270,
                "camera_type": "perspective", "sequence": sequence, "captured_at": index, "width": 100, "height": 100,
            })
        selected = d3.select_frames(raw, target)
        self.assertEqual(4, len(selected))
        self.assertEqual(4, len({item["sequence_id"] for item in selected}))


if __name__ == "__main__":
    unittest.main()
