import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_video_marker_radial_approach_contract as subject


PATH = Path(__file__).resolve().parents[1] / "configs" / "public_video_marker_radial_approach_contract_r725.json"


class MarkerRadialApproachContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = subject.validate_contract(__import__("json").loads(PATH.read_text(encoding="utf-8")))

    def test_frozen_contract_validates(self):
        self.assertEqual("public-video-marker-radial-approach-r725", self.value["contract_id"])

    def test_bottom_progress_cannot_drift(self):
        value = copy.deepcopy(self.value)
        value["radial_approach_gate"]["minimum_bottom_y_progress"] = 0.01
        with self.assertRaisesRegex(ValueError, "gate"):
            subject.validate_contract(value)

    def test_dino_cannot_directly_open(self):
        value = copy.deepcopy(self.value)
        value["feature_contract"]["dinov2_direct_event_open_allowed"] = True
        with self.assertRaisesRegex(ValueError, "DINO"):
            subject.validate_contract(value)

    def test_android_authorization_cannot_be_enabled(self):
        value = copy.deepcopy(self.value)
        value["authorizations"]["android_runtime_change"] = True
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            subject.validate_contract(value)


if __name__ == "__main__":
    unittest.main()
