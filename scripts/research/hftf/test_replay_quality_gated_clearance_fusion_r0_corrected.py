import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_quality_gated_clearance_fusion_r0_corrected import mae
class ReplayTest(unittest.TestCase):
    def test_mae(self): self.assertEqual(mae([1.0,3.0]),2.0)
    def test_empty(self): self.assertIsNone(mae([]))
if __name__=='__main__': unittest.main()
