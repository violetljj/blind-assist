from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dtr_c1_global_obb_cohort_admission import sha256_file
from dtr_m1_confident_direct_velocity import materialize


class ConfidentDirectVelocityTest(unittest.TestCase):
    def test_consistent_motion_is_admitted_and_velocity_flip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "m1.npz"
            np.savez_compressed(
                source,
                frames=np.asarray([0, 5], dtype=np.int32),
                frame_time_s=np.asarray([0.0, 0.35], dtype=np.float64),
                frame_ego_x_m=np.asarray([0.0, 0.0], dtype=np.float64),
                frame_ego_y_m=np.asarray([0.0, 0.0], dtype=np.float64),
                frame_ego_yaw_rad=np.asarray([0.0, 0.0], dtype=np.float64),
                offsets=np.asarray([0, 2, 4], dtype=np.int64),
                forward_m=np.asarray([0.0, 3.0, 0.35, 3.35], dtype=np.float32),
                left_m=np.zeros(4, dtype=np.float32),
                velocity_forward_mps=np.asarray([1.0, 1.0, 1.0, -1.0], dtype=np.float32),
                velocity_left_mps=np.zeros(4, dtype=np.float32),
                component_id=np.asarray([4, 8, 4, 8], dtype=np.int32),
                source_point_count=np.asarray([3, 3, 3, 3], dtype=np.int32),
                flow_support=np.ones(4, dtype=np.float32),
            )
            manifest_path = root / "m1.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "oracle": True,
                        "truth_blind": True,
                        "sequence": "synthetic",
                        "frames": {"first": 0, "last": 5, "count": 2},
                        "frozen_downstream": {},
                        "ledger_sha256": sha256_file(source),
                    }
                ),
                encoding="utf-8",
            )
            output = root / "filtered.npz"
            output_manifest = root / "filtered.json"
            result = materialize(
                source_path=source,
                source_manifest_path=manifest_path,
                output_path=output,
                manifest_path=output_manifest,
            )
            with np.load(output, allow_pickle=False) as values:
                self.assertEqual(values["offsets"].tolist(), [0, 0, 1])
                self.assertEqual(values["component_id"].tolist(), [4])
            self.assertEqual(result["diagnostics"]["admitted_cells"], 1)


if __name__ == "__main__":
    unittest.main()
