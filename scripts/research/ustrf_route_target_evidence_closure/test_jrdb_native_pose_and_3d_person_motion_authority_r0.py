from __future__ import annotations

import unittest

from audit_jrdb_native_pose_and_3d_person_motion_authority_r0 import frame_stem, parse_central


class JrdbNativePoseAuditTest(unittest.TestCase):
    def test_frame_stem(self) -> None:
        self.assertEqual(frame_stem("/root/sequence/000119.pcd"), "000119")

    def test_empty_central_directory(self) -> None:
        self.assertEqual(parse_central(b""), [])


if __name__ == "__main__":
    unittest.main()
