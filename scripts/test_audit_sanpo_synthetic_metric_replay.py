from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_sanpo_synthetic_metric_replay import audit


class SanpoSyntheticMetricReplayAuditTest(unittest.TestCase):
    def make_root(self) -> Path:
        root = Path(tempfile.mkdtemp())
        depth = root / "source_depth" / "train" / "sample.float16.gz"
        depth.parent.mkdir(parents=True)
        values = np.array([2, 3, 1, 2, 3, 4, 5, 6], dtype=np.float16)
        depth.write_bytes(gzip.compress(values.tobytes()))
        (root / "source_metadata").mkdir()
        (root / "source_metadata" / "camera_poses.csv").write_text(
            "tracking_state,pos_x,pos_y,pos_z,q_x,q_y,q_z,q_w\nREADY,0,0,0,0,0,0,1\n",
            encoding="utf-8",
        )
        (root / "manifest.replay.jsonl").write_text(json.dumps({
            "id": "sample", "width": 3, "height": 2, "source_depth_path": "source_depth/train/sample.float16.gz", "source_frame_index": 0
        }) + "\n", encoding="utf-8")
        return root

    def test_accepts_structurally_valid_metric_depth_but_refuses_pose_warp_admission(self) -> None:
        report = audit(self.make_root())
        self.assertTrue(report["ok"])
        self.assertTrue(report["metric_depth_source_integrity"])
        self.assertFalse(report["camera_pose_source"]["ustrf_pose_warp_admitted"])
        self.assertFalse(report["camera_pose_source"]["has_explicit_frame_or_timestamp_binding"])

    def test_rejects_depth_with_wrong_payload_size(self) -> None:
        root = self.make_root()
        path = root / "source_depth" / "train" / "sample.float16.gz"
        path.write_bytes(gzip.compress(np.array([2, 3, 1, 2], dtype=np.float16).tobytes()))
        report = audit(root)
        self.assertFalse(report["ok"])
        self.assertFalse(report["metric_depth_source_integrity"])


if __name__ == "__main__":
    unittest.main()
