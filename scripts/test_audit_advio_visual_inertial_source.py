import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit_advio_visual_inertial_source as subject


class AdvioVisualInertialAuditTest(unittest.TestCase):
    def test_load_csv_rejects_wrong_column_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            np.savetxt(path, np.asarray([[0.0, 1.0, 2.0]]), delimiter=",")
            with self.assertRaisesRegex(ValueError, "unexpected CSV shape"):
                subject.load_csv(path, 4)

    def test_modality_summary_reports_rate_and_monotonicity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.csv"
            values = np.asarray([[0.0, 1.0], [0.01, 2.0], [0.02, 3.0]])
            np.savetxt(path, values, delimiter=",")
            summary = subject.modality_summary(path, values)
            self.assertAlmostEqual(100.0, summary["median_sample_hz"])
            self.assertTrue(summary["timestamps_strictly_increasing"])


if __name__ == "__main__":
    unittest.main()
