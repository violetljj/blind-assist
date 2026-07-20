import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_video_multiexpert_risk_profile_contract as contract


CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "public_video_multiexpert_risk_profile_contract_r723.json"
)


class MultiExpertRiskProfileContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value, cls.metadata = contract.load_contract(CONTRACT_PATH)

    def test_frozen_contract_and_sidecar_validate(self) -> None:
        self.assertEqual(
            "public-video-multiexpert-risk-profile-r723",
            self.value["contract_id"],
        )
        self.assertEqual(64, len(self.metadata["sha256"]))

    def test_absence_cannot_be_promoted_to_clear(self) -> None:
        value = copy.deepcopy(self.value)
        value["risk_profile_channels"]["general_static_dinov2"]["absence_is_clear"] = True
        with self.assertRaisesRegex(ValueError, "absence"):
            contract.validate_contract(value)

    def test_close_rule_cannot_be_weakened(self) -> None:
        value = copy.deepcopy(self.value)
        value["fusion"]["close_rule"] = "any channel may close"
        with self.assertRaisesRegex(ValueError, "fusion"):
            contract.validate_contract(value)

    def test_segmentation_cannot_become_primary(self) -> None:
        value = copy.deepcopy(self.value)
        value["supervision_roles"]["pixel_segmentation"] = "primary"
        with self.assertRaisesRegex(ValueError, "roles"):
            contract.validate_contract(value)

    def test_android_authorization_cannot_be_enabled(self) -> None:
        value = copy.deepcopy(self.value)
        value["authorizations"]["android_runtime_change"] = True
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            contract.validate_contract(value)


if __name__ == "__main__":
    unittest.main()
