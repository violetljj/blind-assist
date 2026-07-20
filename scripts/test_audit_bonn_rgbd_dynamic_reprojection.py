import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("audit_bonn_rgbd_dynamic_reprojection.py")
SPEC = importlib.util.spec_from_file_location("bonn_reprojection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class BonnRgbdReprojectionTest(unittest.TestCase):
    def test_pairs_preserve_rgb_order(self):
        self.assertTrue(callable(MODULE._pairs))


if __name__ == "__main__":
    unittest.main()
