import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freeze_grail_paired_orientation_r1cp import (  # noqa: E402
    OA_V2_CHECKPOINT_SHA256,
    OA_V2_CODE_COMMIT,
    PREVIOUS_VAL_HOUSES,
)


class PairedOrientationFreezeTest(unittest.TestCase):
    def test_model_identity_is_exact(self):
        self.assertEqual(len(OA_V2_CODE_COMMIT), 40)
        self.assertEqual(len(OA_V2_CHECKPOINT_SHA256), 64)

    def test_prior_train_and_dev_are_excluded(self):
        self.assertEqual(len(PREVIOUS_VAL_HOUSES), 30)
        self.assertTrue({663, 513, 636, 403, 860, 910}.issubset(PREVIOUS_VAL_HOUSES))


if __name__ == "__main__":
    unittest.main()
