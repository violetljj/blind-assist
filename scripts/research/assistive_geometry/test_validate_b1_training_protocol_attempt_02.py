import copy
import json
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.validate_b1_training_protocol_attempt_02 import validate


OVERLAY = Path("docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TRAINING_PROTOCOL_2026-08-09_ATTEMPT_02.json")
BASE = Path("docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TRAINING_PROTOCOL_2026-08-09.json")


class ValidateB1TrainingProtocolAttempt02Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
        cls.base = json.loads(BASE.read_text(encoding="utf-8"))

    def test_frozen_overlay_is_valid(self) -> None:
        self.assertEqual([], validate(self.overlay, self.base))

    def test_single_portrait_shape_is_rejected(self) -> None:
        value = copy.deepcopy(self.overlay)
        del value["corrections"]["full_fov_tensor_shapes"]["landscape"]
        self.assertIn("dual-orientation tensor shape drift", validate(value, self.base))

    def test_mixed_orientation_batch_is_rejected(self) -> None:
        value = copy.deepcopy(self.overlay)
        value["corrections"]["mixed_orientation_batch"] = "ALLOWED"
        self.assertIn("mixed orientation batch must be forbidden", validate(value, self.base))

    def test_development_roles_cannot_drift(self) -> None:
        value = copy.deepcopy(self.overlay)
        value["corrections"]["development_roles"]["DEVELOPMENT_CALIBRATION"][0] = copy.deepcopy(
            value["corrections"]["development_roles"]["DEVELOPMENT_SELECTION"][0]
        )
        errors = validate(value, self.base)
        self.assertIn("orientation-balanced calibration identity drift", errors)
        self.assertIn("Development overlay roles overlap", errors)

    def test_formal_training_remains_closed(self) -> None:
        value = copy.deepcopy(self.overlay)
        value["authority"]["formal_student_training"] = True
        self.assertIn("formal training must remain closed", validate(value, self.base))


if __name__ == "__main__":
    unittest.main()
