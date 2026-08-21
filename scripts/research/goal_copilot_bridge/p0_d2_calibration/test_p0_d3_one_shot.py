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

    def test_adjudication_does_not_replace_unavailable_parent(self) -> None:
        names = ["Visible", "Missing 1", "Missing 2", "Missing 3", "Missing 4", "Missing 5"]
        roster = {
            "policy_id": d3.POLICY_ID, "report_sha256": "roster",
            "parents": [{"place_name": name} for name in names],
        }
        acquisition = {
            "roster_sha256": "roster", "report_sha256": "acquisition", "replacement_performed": False,
            "materialized_parent_count": 1,
            "parents": [{"place_name": "Visible", "place_id": "p1", "building_id": "b1", "status": "MATERIALIZED", "frames": [{"id": "f1"}]}] + [
                {"place_name": name, "place_id": f"p{index}", "building_id": f"b{index}", "status": "NO_ELIGIBLE_FRAME_NO_REPLACEMENT", "frames": []}
                for index, name in enumerate(names[1:], start=2)
            ],
        }
        decisions = {"parents": [
            {"place_name": "Visible", "resolution": "UNIQUE", "evidence_note": "one", "valid_targets": [{"frame_id": "f1", "region_normalized_xyxy": [0.1, 0.1, 0.2, 0.2]}]},
            *({"place_name": name, "resolution": "NOT_OBSERVED", "evidence_note": "none", "valid_targets": []} for name in names[1:]),
        ]}
        prior = {"report_sha256": "prior", "claim_ceiling": "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY", "episodes": []}
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as value:
            review = d3.adjudicate(roster, acquisition, decisions, [prior], Path(value) / "out")
        self.assertEqual(sorted(names[1:]), review["unavailable_parent_names"])
        self.assertFalse(review["second_batch_authorized"])


if __name__ == "__main__":
    unittest.main()
