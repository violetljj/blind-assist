import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2_phase_c_sources import (
    archive_member_map,
    asset_lookup,
    role_rows,
)


class D2PhaseCSourceMaterializerTest(unittest.TestCase):
    def test_role_rows_are_exact_and_ordered(self) -> None:
        roles = [
            {
                "role": "D2_TRAIN" if index < 4 else "D2_DEVELOPMENT_SEALED",
                "role_order": index % 4 + 1,
                "phase_a_order": index + 1,
                "pool_order": index + 2,
                "visit_id": f"v{index}",
                "video_id": f"s{index}",
                "selected_frame_stems": [f"s{index}_{frame}" for frame in range(300)],
            }
            for index in range(8)
        ]
        self.assertEqual(8, len(role_rows({"role_assignments": roles})))
        roles[0]["selected_frame_stems"] = roles[0]["selected_frame_stems"][:299]
        with self.assertRaisesRegex(ValueError, "selected stem drift"):
            role_rows({"role_assignments": roles})

    def test_archive_map_verifies_crc_and_rejects_duplicate_stem(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("a/frame.png", b"one")
                bundle.writestr("b/other.png", b"two")
            mapping, count = archive_member_map(archive, (".png",))
            self.assertEqual(2, count)
            self.assertEqual({"frame", "other"}, set(mapping))
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("a/frame.png", b"one")
                bundle.writestr("b/frame.png", b"two")
            with self.assertRaisesRegex(ValueError, "duplicate source stem"):
                archive_member_map(archive, (".png",))

    def test_asset_lookup_requires_exact_32(self) -> None:
        videos = {f"s{index}" for index in range(8)}
        rgb = {"assets": [{"video_id": video, "asset": "lowres_wide.zip", "http_status": 200, "content_length_bytes": 1} for video in videos]}
        support = {
            "assets": [
                {"video_id": video, "asset": asset, "http_status": 200, "content_length_bytes": 1}
                for video in videos
                for asset in ("lowres_wide_intrinsics.zip", "lowres_wide.traj", "lowres_depth.zip", "confidence.zip")
            ]
        }
        self.assertEqual(32, len(asset_lookup(rgb, support, videos)))
        support["assets"].pop()
        with self.assertRaisesRegex(ValueError, "count drift"):
            asset_lookup(rgb, support, videos)


if __name__ == "__main__":
    unittest.main()
