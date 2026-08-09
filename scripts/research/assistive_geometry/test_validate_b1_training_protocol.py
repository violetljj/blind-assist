import copy
import json
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.validate_b1_training_protocol import validate


PROTOCOL = Path("docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TRAINING_PROTOCOL_2026-08-09.json")


class ValidateB1TrainingProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    def test_frozen_protocol_is_valid(self) -> None:
        self.assertEqual([], validate(self.protocol))

    def test_unknown_cannot_be_negative(self) -> None:
        value = copy.deepcopy(self.protocol)
        value["target_schema"]["unknown_is_negative"] = True
        self.assertIn("UNKNOWN cannot become negative", validate(value))

    def test_roles_cannot_overlap(self) -> None:
        value = copy.deepcopy(self.protocol)
        value["data_roles"]["DEVELOPMENT_SELECTION"][0] = copy.deepcopy(value["data_roles"]["TRAIN"][0])
        self.assertIn("data role identity overlap", validate(value))

    def test_formal_training_waits_for_implementation_lock(self) -> None:
        value = copy.deepcopy(self.protocol)
        value["authority"]["formal_student_training"] = True
        self.assertIn("formal training must wait for implementation lock", validate(value))

    def test_ablation_must_remain_additive(self) -> None:
        value = copy.deepcopy(self.protocol)
        value["ablation_arms"][3]["active_losses"].remove("occupancy_bce")
        self.assertIn("arm is not additive: A3_PLUS_FALSE_CLEAR", validate(value))

    def test_confidence_calibration_cannot_use_selection(self) -> None:
        value = copy.deepcopy(self.protocol)
        value["confidence"]["threshold_fit_role"] = "DEVELOPMENT_SELECTION"
        self.assertIn("threshold role drift", validate(value))


if __name__ == "__main__":
    unittest.main()
