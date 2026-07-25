from __future__ import annotations

import unittest

from audit_thor_source_native_id_time_transform_authority_recovery_r1 import (
    calibration_files,
    raw_qtm_files,
    recovery_files,
)


class ThorSourceNativeAuthorityRecoveryR1Test(unittest.TestCase):
    def test_raw_qtm_classifier_is_exact(self) -> None:
        self.assertEqual(
            raw_qtm_files(["run.qtm", "settings.qtmproj", "run.mat", "qtm_notes.txt"]),
            ["run.qtm", "settings.qtmproj"],
        )

    def test_calibration_classifier(self) -> None:
        self.assertEqual(
            calibration_files(
                ["calibration.yaml", "lidar_extrinsic.json", "trajectory.tsv"]
            ),
            ["calibration.yaml", "lidar_extrinsic.json"],
        )

    def test_recovery_classifier_does_not_treat_filtered_as_mask(self) -> None:
        self.assertEqual(
            recovery_files(
                ["run_filtered.mat", "id_switch_mask.tsv", "recovery_provenance.json"]
            ),
            ["id_switch_mask.tsv", "recovery_provenance.json"],
        )


if __name__ == "__main__":
    unittest.main()
