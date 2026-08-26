from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_manifest.py")
SPEC = importlib.util.spec_from_file_location("ulr_build_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


Location = MODULE.Location
assign_location_splits = MODULE.assign_location_splits
build_manifest = MODULE.build_manifest
capture_group_from_name = MODULE.capture_group_from_name


class ManifestContractTest(unittest.TestCase):
    def test_capture_groups_hold_video_frames_and_social_events_together(self):
        self.assertEqual(("field:IMG_1177", "field_capture"), capture_group_from_name("IMG_1177_frame_000123.jpg"))
        self.assertEqual(("social:2024:2", "social_media"), capture_group_from_name("Social media S-1 2024 Event 2-2.jpg"))

    def test_location_split_is_deterministic_and_identity_disjoint(self):
        locations = [Location(f"N-1-{index}", "node", 30 + index / 1000, 104.0) for index in range(1, 21)]
        first = assign_location_splits(
            locations,
            salt="frozen",
            ratios={"train": 0.7, "development": 0.15, "test": 0.15},
        )
        second = assign_location_splits(
            list(reversed(locations)),
            salt="frozen",
            ratios={"train": 0.7, "development": 0.15, "test": 0.15},
        )
        self.assertEqual(first, second)
        self.assertEqual(set(locations), {item for item in locations})
        self.assertEqual(20, len(first))
        self.assertEqual({"train", "development", "test"}, set(first.values()))

    def test_manifest_keeps_eval_capture_groups_in_one_role(self):
        catalog = {
            f"N-1-{index}": Location(f"N-1-{index}", "node", 30 + index / 1000, 104.0)
            for index in range(1, 10)
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for location_id in catalog:
                folder = root / location_id
                folder.mkdir()
                for capture in (1001, 1002, 1003):
                    for frame in (1, 2):
                        (folder / f"IMG_{capture}_frame_{frame:06d}.jpg").touch()
            manifest, audit = build_manifest(
                images_root=root,
                catalog=catalog,
                split_salt="split",
                group_salt="group",
            )
        self.assertEqual("ADMITTED", audit["status"])
        roles: dict[tuple[str, str], set[str]] = {}
        for row in manifest["images"]:
            roles.setdefault((row["location_id"], row["capture_group"]), set()).add(row["role"])
        self.assertTrue(all(len(values) == 1 for values in roles.values()))
        split_by_location: dict[str, set[str]] = {}
        for row in manifest["images"]:
            split_by_location.setdefault(row["location_id"], set()).add(row["split"])
        self.assertTrue(all(len(values) == 1 for values in split_by_location.values()))


if __name__ == "__main__":
    unittest.main()
