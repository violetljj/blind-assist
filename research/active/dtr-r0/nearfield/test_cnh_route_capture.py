"""Transport/math regressions; these do not claim UE rendering acceptance."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cnh_route_capture import basis, camera_record, decode_id_color, id_color, write_npy
from cnh_route_geometry import trace


class CaptureContractTest(unittest.TestCase):
    def test_mesh_audit_retains_axial_units_and_interpolates_normals(self):
        tri = np.array([[[2., -1, -1], [2, 1, -1], [2, 0, 1]]])
        normals = np.array([[[-1., 0, 0], [-1, 0, 0], [-1, 0, 0]]])
        rays = np.array([[1., 0, 0], [1, .1, .1], [1, 4, 0]])
        for origin in (np.zeros(3), np.array([0, .06, 0])):
            depth, ids, observed_normal, index = trace(origin, rays, tri, np.array([257]), normals)
            self.assertTrue(np.allclose(depth[:2], [2, 2]))
            self.assertTrue(np.isinf(depth[2]))
            self.assertEqual(ids.tolist(), [257, 257, 0])
            self.assertTrue(np.allclose(observed_normal[:2], [[-1, 0, 0], [-1, 0, 0]]))
            self.assertEqual(index.tolist(), [0, 0, -1])

    def test_all_uint16_ids_roundtrip_through_half_precision(self):
        known = set(range(1, 65535))
        for identifier in known:
            half = np.asarray(id_color(identifier), dtype=np.float16).astype(np.float32)
            self.assertEqual(decode_id_color(half, known), identifier)

    def test_corruption_and_unmapped_ids_are_unknown(self):
        self.assertEqual(decode_id_color(id_color(257), {1}), 65535)
        rgb = id_color(512)
        rgb[2] = 1.
        self.assertEqual(decode_id_color(rgb, {512}), 65535)
        self.assertEqual(decode_id_color([float('nan'), 0, 0], {1}), 65535)
        self.assertEqual(decode_id_color([0, 0, 0], {1}), 0)
        for value in (0, 65535, -1, 2.5):
            with self.assertRaises(ValueError):
                id_color(value)

    def test_rotation_and_right_baseline(self):
        pose = dict(x=3, y=4, z=1.6, pitch=-10, yaw=36, roll=4)
        forward, right, up = np.asarray(basis(pose))
        self.assertTrue(np.allclose(np.array([forward, right, up]) @ np.array([forward, right, up]).T, np.eye(3)))
        rig = dict(width=640, height=360, hfov_deg=100, baseline_m=.06)
        matrix = np.array(camera_record(pose, rig)['T_world_camera'])
        self.assertTrue(np.allclose(matrix[:3, 0], right))
        self.assertTrue(np.allclose(matrix[:3, 1], -up))
        self.assertTrue(np.allclose(matrix[:3, 2], forward))
        self.assertTrue(np.allclose(matrix[:3, 3], [3, 4, 1.6]))
        self.assertAlmostEqual(np.linalg.norm(.06*matrix[:3, 0]), .06)

    def test_transport_preserves_invalid_and_uint16(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_npy(root/'depth.npy', [1., float('nan'), 2., 3.], (2, 2))
            depth = np.load(root/'depth.npy', allow_pickle=False)
            self.assertEqual(depth.dtype, np.dtype('<f4'))
            self.assertTrue(math.isnan(depth[0, 1]))
            write_npy(root/'id.npy', [1, 255, 256, 65534], (2, 2), '<u2')
            self.assertEqual(np.load(root/'id.npy').tolist(), [[1, 255], [256, 65534]])
            with self.assertRaises(FileExistsError):
                write_npy(root/'id.npy', [1], (1,), '<u2')
            with self.assertRaises(ValueError):
                write_npy(root/'bad.npy', [1], (2,), '<u2')


if __name__ == '__main__':
    unittest.main()
