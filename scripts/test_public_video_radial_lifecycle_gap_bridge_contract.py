import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_video_radial_lifecycle_gap_bridge_contract as subject


PATH = Path(__file__).resolve().parents[1] / "configs" / "public_video_radial_lifecycle_gap_bridge_contract_r730.json"


class RadialLifecycleGapBridgeContractTest(unittest.TestCase):
    def test_frozen_contract_loads(self) -> None:
        value, meta = subject.load_contract(PATH)
        self.assertEqual(9, value["lifecycle"]["clear_absent_samples"])
        self.assertEqual(64, len(meta["sha256"]))

    def test_threshold_drift_is_rejected(self) -> None:
        value = subject.lifecycle.verify_json_sidecar(PATH)
        changed = copy.deepcopy(value)
        changed["lifecycle"]["clear_absent_samples"] = 10
        with self.assertRaisesRegex(ValueError, "differs"):
            subject.validate_contract(changed)


if __name__ == "__main__":
    unittest.main()
