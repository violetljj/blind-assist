"""Independent analytic fixtures; no historical cohort scoring."""
import math
import unittest

from surface_contact_oracle import oracle, oracle_width, camera_rotation
import numpy as np


def cube(center, half):
    return dict(center_m=center, extent_m=half)


def frame(*objects, yaw=0., pitch=0., roll=0.):
    return dict(camera=dict(x=0., y=0., z=1.7, yaw=yaw, pitch=pitch, roll=roll), native_bounds=list(objects))


class ContactOracleTests(unittest.TestCase):
    def test_axial_distance_not_front_offset_or_range(self):
        f = frame(cube([2., .2, 1.7], [.1, .1, .05]))
        self.assertAlmostEqual(oracle(f, 'HEAD', .36), 1.9)
        self.assertAlmostEqual(oracle_width(f, 'HEAD'), .2)

    def test_rotated_cube_not_enclosing_camera_aabb(self):
        f = frame(cube([2., 1., 1.7], [.5, .5, .05]), yaw=45.)
        # In world xy, y=x touches the cube only at (1.5,1.5).
        self.assertAlmostEqual(oracle(f, 'HEAD', 0.), 3/math.sqrt(2), places=8)
        self.assertAlmostEqual(oracle(f, 'HEAD', .2), 3/math.sqrt(2)-.1, places=8)
        self.assertAlmostEqual(oracle_width(f, 'HEAD'), 0.)

    def test_two_solids_keep_empty_gap(self):
        f = frame(cube([2., -.3, 1.7], [.1, .1, .05]), cube([2., .3, 1.7], [.1, .1, .05]))
        self.assertIsNone(oracle(f, 'HEAD', .36))
        self.assertAlmostEqual(oracle_width(f, 'HEAD'), .4)
        self.assertAlmostEqual(oracle(f, 'HEAD', .4), 1.9)

    def test_width_horizon_and_contact_distance_differ(self):
        f = frame(cube([3.5, 0., 1.7], [.1, .1, .05]))
        self.assertAlmostEqual(oracle(f, 'HEAD', .36), 3.4)
        self.assertIsNone(oracle_width(f, 'HEAD'))

    def test_empty_behind_and_far_are_scope_absence(self):
        for f in (frame(), frame(cube([-1., 0., 1.7], [.1, .1, .1])), frame(cube([5., 0., 1.7], [.1, .1, .1]))):
            self.assertIsNone(oracle(f, 'HEAD', .36))
            self.assertIsNone(oracle_width(f, 'HEAD'))

    def test_height_and_frustum_are_real_constraints(self):
        # HEAD y is allowed by 1m width, but outside horizontal common FoV.
        f = frame(cube([.6, .45, 1.7], [.01, .01, .01]))
        self.assertIsNone(oracle(f, 'HEAD', 1.))
        self.assertIsNone(oracle_width(f, 'HEAD'))
        # BODY height is inside template but outside vertical FoV this close.
        g = frame(cube([.6, 0., .8], [.01, .01, .01]))
        self.assertIsNone(oracle(g, 'BODY', .56))
        self.assertIsNone(oracle_width(g, 'BODY'))

    def test_maximum_critical_full_width(self):
        f = frame(cube([2., .7, 1.7], [.1, .1, .05]))
        self.assertIsNone(oracle_width(f, 'HEAD'))

    def test_extra_receipt_solids_and_body_band(self):
        extra = (dict(center_m=[2., 0., 1.], size_m=[.2, .4, .2]),)
        self.assertAlmostEqual(oracle(frame(), 'BODY', .56, extra), 1.9)
        self.assertAlmostEqual(oracle_width(frame(), 'BODY', extra), 0.)
        self.assertIsNone(oracle(frame(), 'HEAD', .36, extra))

    def test_camera_relative_under_pitch_and_roll(self):
        # A small cube is centered on the camera forward ray after rotation.
        f = frame(yaw=12., pitch=10., roll=7.)
        rotation = camera_rotation(f['camera'])
        center = rotation @ np.array([2., 0., 0.])+[0., 0., 1.7]
        f['native_bounds'] = [cube(center.tolist(), [.01, .01, .01])]
        self.assertAlmostEqual(oracle_width(f, 'HEAD'), 0.)
        self.assertTrue(1.97 < oracle(f, 'HEAD', .36) < 2.)

    def test_window_clips_intersecting_solid_at_half_metre(self):
        f = frame(cube([.5, 0., 1.7], [.1, .01, .01]))
        self.assertAlmostEqual(oracle(f, 'HEAD', .36), .5)

    def test_validation_fails_explicitly(self):
        with self.assertRaises(ValueError):oracle(frame(), 'HEAD', 1.01)
        with self.assertRaises(ValueError):oracle(frame(), 'FOOT', .3)
        with self.assertRaises(ValueError):oracle(frame(cube([1.,0.,1.7], [0.,.1,.1])), 'HEAD', .3)


if __name__ == '__main__':unittest.main()
