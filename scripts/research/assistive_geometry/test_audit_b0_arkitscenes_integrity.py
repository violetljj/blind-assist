import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.research.assistive_geometry.audit_b0_arkitscenes_integrity import (
    decode_image,
    maximum_bracketing_gap,
    parse_intrinsics,
    parse_trajectory,
)


class AuditB0ArkitScenesIntegrityTest(unittest.TestCase):
    def test_decode_expected_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth_path = root / "depth.png"
            confidence_path = root / "confidence.png"
            Image.fromarray(np.full((3, 4), 123, dtype=np.uint16)).save(depth_path)
            Image.fromarray(np.full((3, 4), 2, dtype=np.uint8)).save(confidence_path)
            self.assertEqual((4, 3), decode_image(depth_path, "lowres_depth")[0])
            self.assertEqual(2, decode_image(confidence_path, "confidence")[4])

    def test_intrinsics_and_trajectory_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intrinsics = root / "sample.pincam"
            intrinsics.write_text("256 192 200 201 128 96\n", encoding="utf-8")
            self.assertEqual((256, 192), parse_intrinsics(intrinsics)[:2])
            trajectory = root / "sample.traj"
            trajectory.write_text(
                "1.0 0 0 0 0 0 0\n1.1 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            self.assertEqual(2, len(parse_trajectory(trajectory)))

    def test_maximum_bracketing_gap_rejects_out_of_domain_frame(self) -> None:
        self.assertAlmostEqual(0.1, maximum_bracketing_gap([1.05], [1.0, 1.1]))
        with self.assertRaisesRegex(ValueError, "outside"):
            maximum_bracketing_gap([0.9], [1.0, 1.1])


if __name__ == "__main__":
    unittest.main()
