from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("prepare_revel_development_manifest.py")
SPEC = importlib.util.spec_from_file_location("prepare_revel_development_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

VALIDATOR_PATH = Path(__file__).with_name("validate_revel_development_manifest.py")
VALIDATOR_SPEC = importlib.util.spec_from_file_location("validate_revel_development_manifest", VALIDATOR_PATH)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


class PrepareRevelDevelopmentManifestTest(unittest.TestCase):
    def test_region_boundaries_are_center_inclusive(self) -> None:
        self.assertEqual(MODULE.region_for_center_x(0.2), "LEFT")
        self.assertEqual(MODULE.region_for_center_x(1 / 3), "CENTER")
        self.assertEqual(MODULE.region_for_center_x(2 / 3), "CENTER")
        self.assertEqual(MODULE.region_for_center_x(0.8), "RIGHT")

    def test_label_parser_clamps_only_machine_epsilon(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("1 0.25 0.5000000000000002 0.3 1.0000000000000002\n", encoding="utf-8")
            self.assertEqual(MODULE.parse_label(path), [(1, 0.25, 0.5000000000000002, 0.3, 1.0)])

    def test_event_segmentation_preserves_parent_across_region_change(self) -> None:
        rows = []
        for index, region in enumerate(("LEFT", "LEFT", "CENTER", "CENTER", "RIGHT")):
            rows.append({
                "target_id": "track-000",
                "source_frame_index": index,
                "bag_image_timestamp_ns": index * 40_000_000,
                "unique_roi_available": True,
                "truth_available": True,
                "truth_state": "approaching",
                "region": region,
            })
        events = MODULE.segment_natural_events(rows)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["primary_event_eligible"])
        self.assertEqual(events[0]["anchor_region"], "LEFT")
        self.assertEqual(events[0]["region_frame_counts"], {"LEFT": 2, "CENTER": 2, "RIGHT": 1})
        self.assertEqual({row["event_id"] for row in rows}, {"track-000:event-0000"})

    def test_state_change_and_gap_split_events(self) -> None:
        rows = [
            {
                "target_id": "track-001",
                "source_frame_index": index,
                "bag_image_timestamp_ns": timestamp,
                "unique_roi_available": True,
                "truth_available": True,
                "truth_state": state,
                "region": "CENTER",
            }
            for index, timestamp, state in (
                (0, 0, "approaching"),
                (1, 40_000_000, "approaching"),
                (2, 80_000_000, "receding"),
                (4, 240_000_000, "receding"),
            )
        ]
        events = MODULE.segment_natural_events(rows)
        self.assertEqual([event["eligible_frame_count"] for event in events], [2, 1, 1])
        self.assertFalse(any(event["primary_event_eligible"] for event in events))

    def test_producer_field_firewall_excludes_truth(self) -> None:
        producer_fields = {
            "source_frame_id",
            "target_id",
            "track_epoch",
            "roi_xywh_normalized",
            "region",
        }
        self.assertFalse(producer_fields.intersection(VALIDATOR.FORBIDDEN_PRODUCER_FIELDS))
        self.assertIn("truth_state", VALIDATOR.FORBIDDEN_PRODUCER_FIELDS)


if __name__ == "__main__":
    unittest.main()
