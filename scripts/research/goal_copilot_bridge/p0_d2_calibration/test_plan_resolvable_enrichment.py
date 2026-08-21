from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p0_d2_calibration import plan_resolvable_enrichment as planner


def place(place_id: str, building_id: str, name: str) -> dict:
    return {"place_id": place_id, "place_name": name, "status": "CANDIDATE_ONLY", "building_ids": [building_id]}


def anchor(building_id: str, anchor_id: str = "node/1") -> dict:
    return {"status": "CANDIDATE_ONLY", "entrance": "yes", "overture_building_id": building_id, "osm_entrance_id": anchor_id}


class RosterTest(unittest.TestCase):
    def test_is_deterministic_diverse_and_excludes_prior_buildings(self) -> None:
        source = {
            "report_sha256": "source",
            "place_building_crosswalk_candidates": [],
            "osm_entrance_building_crosswalk_candidates": [],
        }
        features = []
        categories = ["hotel", "cafe", "clothing_store", "museum", "pharmacy"] * 3
        basic_by_primary = {
            "hotel": "hotel", "cafe": "restaurant", "clothing_store": "fashion_and_apparel_store",
            "museum": "museum", "pharmacy": "medical_supply_store",
        }
        for index, category in enumerate(categories):
            building_id, place_id = f"b{index}", f"p{index}"
            source["place_building_crosswalk_candidates"].append(place(place_id, building_id, f"Venue {index}"))
            source["osm_entrance_building_crosswalk_candidates"].append(anchor(building_id, f"node/{index}"))
            features.append({"id": place_id, "properties": {
                "confidence": 0.99, "basic_category": basic_by_primary[category], "categories": {"primary": category},
            }})
        places = {"features": features}
        excluded = [{"acquisition": {"target_building_ids": ["b0"]}, "images": [{"id": "old-frame"}]}]
        first = planner.plan_roster(
            source, places, excluded, requested_parent_count=10, maximum_family_count=3,
            excluded_target_names=["Venue 1"],
        )
        second = planner.plan_roster(
            source, places, excluded, requested_parent_count=10, maximum_family_count=3,
            excluded_target_names=["Venue 1"],
        )
        self.assertEqual(first, second)
        self.assertNotIn("b0", {item["building_id"] for item in first["parents"]})
        self.assertNotIn("Venue 1", {item["place_name"] for item in first["parents"]})
        self.assertEqual(["old-frame"], first["excluded_frame_ids"])
        self.assertLessEqual(max(first["family_counts"].values()), 3)
        self.assertTrue(all(item["anchor_count"] <= 32 for item in first["acquisition_shards"]))
        self.assertEqual(10, sum(item["parent_count"] for item in first["acquisition_shards"]))
        self.assertEqual("DEVELOPMENT_RESOLVABLE_ENRICHMENT_NOT_ADJUDICATION", first["data_role"])

    def test_rejects_multi_place_building_and_insufficient_roster(self) -> None:
        source = {
            "report_sha256": "source",
            "place_building_crosswalk_candidates": [place("p1", "b1", "One"), place("p2", "b1", "Two")],
            "osm_entrance_building_crosswalk_candidates": [anchor("b1")],
        }
        places = {"features": [
            {"id": "p1", "properties": {"confidence": 0.99, "basic_category": "hotel", "categories": {"primary": "hotel"}}},
            {"id": "p2", "properties": {"confidence": 0.99, "basic_category": "hotel", "categories": {"primary": "hotel"}}},
        ]}
        with self.assertRaisesRegex(planner.RosterError, "insufficient"):
            planner.plan_roster(source, places, [], requested_parent_count=10)


if __name__ == "__main__":
    unittest.main()
