import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("audit_revel_rgb_vicon_reprojection.py")
SPEC = importlib.util.spec_from_file_location("revel_reprojection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RevelRgbViconReprojectionTest(unittest.TestCase):
    def test_camera_projection_identity(self):
        import torch
        pixels = MODULE._project_camera_points(
            torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64),
            torch.tensor([[100.0, 0.0, 12.0], [0.0, 200.0, 34.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
        )
        self.assertTrue(torch.allclose(pixels, torch.tensor([[12.0, 34.0]], dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
