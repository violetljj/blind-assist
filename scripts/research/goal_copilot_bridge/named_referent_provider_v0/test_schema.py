from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.research.goal_copilot_bridge.named_referent_provider_v0.schema import GoalReferencePack, SchemaError


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.image = Path(self.temporary.name) / "reference.png"
        Image.new("RGB", (8, 8), "black").save(self.image)

    def tearDown(self):
        self.temporary.cleanup()

    def test_pack_supports_references_logo_and_optional_bearing(self):
        pack = GoalReferencePack.from_mapping({
            "name": "Example POI",
            "aliases": ["Example"],
            "reference_images": [self.image],
            "logo": self.image,
            "map_bearing_degrees": 12.5,
        })
        self.assertEqual(pack.reference_images, (self.image.resolve(),))
        self.assertEqual(pack.logo, self.image.resolve())
        self.assertEqual(pack.map_bearing_degrees, 12.5)

    def test_invalid_bearing_and_missing_image_fail_closed(self):
        with self.assertRaises(SchemaError):
            GoalReferencePack.from_mapping({"name": "x", "aliases": [], "reference_images": [], "map_bearing_degrees": 360})
        with self.assertRaises(SchemaError):
            GoalReferencePack.from_mapping({"name": "x", "aliases": [], "reference_images": [self.image.with_name("missing.jpg")]})


if __name__ == "__main__":
    unittest.main()
