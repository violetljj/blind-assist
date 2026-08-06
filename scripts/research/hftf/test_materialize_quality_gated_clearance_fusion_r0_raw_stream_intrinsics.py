import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import materialize_quality_gated_clearance_fusion_r0_raw_stream as subject


class IntrinsicsTest(unittest.TestCase):
    def test_parse_pincam(self):
        self.assertEqual([214.498, 214.498, 125.055, 94.3456], subject.parse_pincam(b"256 192 214.498 214.498 125.055 94.3456\n"))

    def test_parse_pincam_rejects_drift(self):
        with self.assertRaisesRegex(ValueError, "schema drift"):
            subject.parse_pincam(b"214.498 214.498 125.055 94.3456\n")


if __name__ == "__main__":
    unittest.main()
