import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depthart_admission_r1 as r1


class DepthArtAdmissionR1Test(unittest.TestCase):
    def test_preprocess_audit_is_exact_and_no_translation(self):
        result = r1.preprocess_audit(Path("artifacts.local/evidence/hftf/depthart-admission-r1-test-preprocess.json"))
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["cases"][0]["resized"], [640, 480])
        self.assertEqual(result["cases"][0]["translation"], [0.0, 0.0])


if __name__ == "__main__": unittest.main()
