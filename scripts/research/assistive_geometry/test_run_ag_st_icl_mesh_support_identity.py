#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_ag_st_icl_mesh_support_identity import (
    load_horizontal_mesh_samples,
    parse_global_pose_text,
    visible_sample_mask,
)


class IclMeshSupportIdentityTest(unittest.TestCase):
    def test_pose_parser_and_frustum(self) -> None:
        poses = parse_global_pose_text("1 0 0 0\n0 1 0 1.5\n0 0 1 0\n")
        self.assertEqual(1, len(poses))
        points = np.asarray([[0.0, 1.0, 1.0], [10.0, 1.0, 1.0], [0.0, 1.0, -1.0]])
        mask = visible_sample_mask(points, poses[0])
        np.testing.assert_array_equal(mask, np.asarray([True, False, False]))

    def test_obj_parser_keeps_horizontal_triangle_and_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.obj"
            path.write_text(
                "o room_floor\n"
                "v 0 0 0\n"
                "v 0 0 1\n"
                "v 1 0 0\n"
                "f 1 2 3\n",
                encoding="utf-8",
            )
            result = load_horizontal_mesh_samples(path, area_per_sample_m2=0.1)
        self.assertEqual(3, result["vertex_count"])
        self.assertEqual(1, result["horizontal_triangle_count"])
        self.assertTrue(np.all(result["object_names"] == "room_floor"))
        np.testing.assert_allclose(result["heights_world"], 0.0)


if __name__ == "__main__":
    unittest.main()
