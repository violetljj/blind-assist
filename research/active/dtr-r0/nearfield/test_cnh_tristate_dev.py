"""Independent focused checks for the instantaneous three-state label contract.

Expected geometry uses physical box volume and closed contact, not the older
triangle-only, [0.3, 3.0]m surface labels. No datasets or model are loaded.
"""
import unittest

import numpy as np

import cnh_tristate_dev as D


def box(lo, hi):
    return dict(lo=list(lo), hi=list(hi))


def travel(yaw=0., position=(0., 0., 0.)):
    radians = np.deg2rad(yaw)
    c, s = np.cos(radians), np.sin(radians)
    pose = np.eye(4)
    pose[:3, :3] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
    pose[:3, 3] = position
    return pose


class InstantaneousLabelChecks(unittest.TestCase):
    def assert_contact(self, obstacle, expected, pose=None):
        result = D.labels([obstacle], travel() if pose is None else pose)
        self.assertEqual(bool(result['contact']), expected)
        return result

    def test_box_contains_entire_corridor(self):
        obstacle = box((-10., -10., -1.), (10., 10., 10.))
        rows = D.contact_boxes([obstacle], travel())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['box_index'], 0)
        self.assertEqual(set(rows[0]['queries']), {0, 1})
        self.assertTrue(D.labels([obstacle], travel())['contact'])

    def test_rotated_aabb_overlap_is_not_physical_contact(self):
        # The transformed AABB overlaps x<=.30 and z<=2.5 separately, but
        # no physical point satisfies both. It is outside the far right corner.
        obstacle = box((1.51, -.1, 2.014), (1.70, .2, 2.20))
        pose = travel(30.)
        corners = np.array([[x, y, z] for x in (1.51, 1.70)
                            for y in (-.1, .2) for z in (2.014, 2.20)])
        local = (corners - pose[:3, 3]) @ pose[:3, :3]
        self.assertLess(local[:, 0].min(), .30)
        self.assertLess(local[:, 2].min(), 2.5)
        self.assertGreater(local[:, 0].max(), -.30)
        self.assertGreater(local[:, 2].max(), 0.)
        self.assert_contact(obstacle, False, pose)

    def test_lateral_boundary_depth_uses_finite_window_intersection(self):
        # This long world box reaches deeper lateral coordinates only after
        # the 2.5m far boundary. Its actual corridor invasion is 2.602mm.
        obstacle = box((.727, -.1, 1.), (1., .2, 4.))
        pose = travel(10.)
        rows = D.contact_boxes([obstacle], pose)
        expected_depth = .30 - (.727 - np.sin(np.deg2rad(10.))*2.5) / np.cos(np.deg2rad(10.))
        self.assertAlmostEqual(rows[0]['depth_m'], expected_depth, places=12)
        result = self.assert_contact(obstacle, True, pose)
        self.assertTrue(result['boundary'])

    def test_forward_window_and_closed_endpoints(self):
        cases = [
            (box((-.1, -.1, -.2), (.1, .1, -.001)), False),
            (box((-.1, -.1, -.2), (.1, .1, 0.)), True),
            (box((-.1, -.1, .01), (.1, .1, .10)), True),
            (box((-.1, -.1, 2.5), (.1, .1, 2.6)), True),
            (box((-.1, -.1, 2.501), (.1, .1, 2.6)), False),
        ]
        for obstacle, expected in cases:
            with self.subTest(obstacle=obstacle):
                self.assert_contact(obstacle, expected)

    def test_lateral_contact_and_graze_are_distinct(self):
        self.assert_contact(box((.30, -.1, 1.), (.35, .1, 1.1)), True)
        result = self.assert_contact(box((.31, -.1, 1.), (.38, .1, 1.1)), False)
        self.assertTrue(result['graze'])
        result = self.assert_contact(box((.29, -.1, 1.), (.38, .1, 1.1)), True)
        self.assertFalse(result['graze'])
        result = self.assert_contact(box((.401, -.1, 1.), (.5, .1, 1.1)), False)
        self.assertFalse(result['graze'])

    def test_other_height_and_translated_pose(self):
        obstacle = box((-.1, .50, 1.), (.1, .70, 1.1))
        rows = D.contact_boxes([obstacle], travel())
        self.assertEqual(set(rows[0]['queries']), {1})
        self.assert_contact(box((-.1, 1.01, 1.), (.1, 1.1, 1.1)), False)
        translated = box((2.1, -.1, 4.), (2.2, .1, 4.1))
        self.assert_contact(translated, True, travel(position=(2., 0., 3.)))

    def test_empty_scene(self):
        result = D.labels([], travel())
        self.assertFalse(result['contact'])
        self.assertFalse(result['graze'])
        self.assertFalse(result['boundary'])


class CausalSmoothingChecks(unittest.TestCase):
    def test_five_weight_window_and_no_future_access(self):
        raw = np.stack((np.arange(13.), np.arange(13.) ** 2), axis=-1)
        expected = np.empty_like(raw)
        weights = np.array([1., 2., 4., 8., 16.])
        for frame in range(13):
            start = max(0, frame - 4)
            w = weights[-(frame - start + 1):]
            expected[frame] = (raw[start:frame + 1] * w[:, None]).sum(0) / w.sum()
        np.testing.assert_allclose(D.smooth(raw), expected, atol=1e-12, rtol=0.)
        changed = raw.copy()
        changed[8:] = 10000.
        np.testing.assert_allclose(D.smooth(changed)[:8], expected[:8], atol=1e-12, rtol=0.)


if __name__ == '__main__':
    unittest.main()
