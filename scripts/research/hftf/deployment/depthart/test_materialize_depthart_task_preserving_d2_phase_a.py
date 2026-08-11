import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2_phase_a import (
    continuous_window,
    member_map,
)


class D2PhaseAMaterializerTest(unittest.TestCase):
    def test_continuous_window_resets_on_gap(self) -> None:
        stems = ["v_1.0", "v_1.1", "v_2.0", "v_2.1", "v_2.2"]
        self.assertEqual(["v_2.0", "v_2.1", "v_2.2"], continuous_window(stems, 3, 0.5))

    def test_continuous_window_fails_when_short(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer than 3"):
            continuous_window(["v_1.0", "v_1.1"], 3, 0.5)

    def test_intrinsics_member_map_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "intrinsics.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("lowres_wide_intrinsics/v_1.0.pincam", "1 1 1 1 0 0")
                bundle.writestr("lowres_wide_intrinsics/v_1.1.pincam", "1 1 1 1 0 0")
            self.assertEqual(
                {"v_1.0", "v_1.1"},
                set(member_map(archive)),
            )


if __name__ == "__main__":
    unittest.main()
