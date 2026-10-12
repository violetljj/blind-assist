"""Focused CPU contracts for finite CNH support and candidate/evaluator bounds."""
import ast
import inspect
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import cnh_bearing_support_cross_dev as C


def ry(angle):
    angle = np.radians(angle); c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def corners(low, high):
    return np.asarray([np.where(bits, high, low) for bits in
                       itertools.product((0,1), repeat=3)], float)


class FrozenSupportContracts(unittest.TestCase):
    def test_y_x_radial_layout_and_finite_bin_extent(self):
        # Swapping angular Y/X would reverse these signs.
        p = C.bin_samples(np.array([[0, 7, 4]]))[0]
        self.assertEqual(p.shape, (27, 3))
        self.assertTrue((p[:, 0] > 0).all())
        self.assertTrue((p[:, 1] < 0).all())
        self.assertTrue((p[:, 2] > 0).all())
        np.testing.assert_allclose(np.linalg.norm(p, axis=-1).min(), 4*C.WIDTH, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(p, axis=-1).max(), 5*C.WIDTH, atol=1e-12)
        # The midpoint is not silently substituted for a finite angular bin.
        self.assertGreater(np.ptp(p[:, 0]/p[:, 2]), 0.)
        np.testing.assert_allclose(p[13, 0]/p[13, 2], .875*C.EDGE, atol=1e-12)
        np.testing.assert_allclose(p[13, 1]/p[13, 2], -.875*C.EDGE, atol=1e-12)

    def test_branch_extrinsic_and_horizontal_head_transform(self):
        for angle in (-15., 0., 15.):
            actual = C.extrinsic(angle)
            np.testing.assert_allclose(actual[:3,:3], C.rx(10.)@ry(angle)@C.rx(-10.), atol=1e-12)
            np.testing.assert_array_equal(actual[:3,3], 0.)
            np.testing.assert_allclose(actual[:3,:3].T@actual[:3,:3], np.eye(3), atol=1e-12)
            # Mount pitch cancels to make the intended horizontal yaw branch.
            np.testing.assert_allclose(C.rx(-10.)@actual[:3,:3], ry(angle)@C.rx(-10.), atol=1e-12)
        q = np.eye(4); q[:3,:3] = ry(20.)@C.rx(-10.)
        _, transform = C.cell_masks(q)
        np.testing.assert_allclose(transform[:3,:3], ry(-20.), atol=1e-12)

    def test_finite_cells_keep_boundary_overlap(self):
        q = np.eye(4); q[:3,:3] = C.rx(-10.)
        masks, _ = C.cell_masks(q)
        angle = np.degrees(np.arctan2(C.CENTRES[:,0], C.CENTRES[:,2]))
        # Some CENTER centroids have finite volume extending to RIGHT/LEFT.
        self.assertTrue(np.any((np.abs(angle) < 10.) & masks[:,1] & masks[:,2]))
        self.assertTrue(np.any((np.abs(angle) < 10.) & masks[:,1] & masks[:,0]))
        k = np.tan(np.radians(10.))
        tangent = corners(np.array([k*1.1,-.1,1.]), np.array([k*1.1+.02,.1,1.1]))
        with patch.object(C, 'CORNERS', tangent[None]), patch.object(C, 'INDEX', np.array([[0,0,0]])):
            right, _ = C.cell_masks(q)
        np.testing.assert_array_equal(right, [[False,False,True]])

    def test_partial_behind_cell_is_ambiguous(self):
        q = np.eye(4); q[:3,:3] = C.rx(-10.)
        crossing = corners(np.array([-.1,-.1,-.1]), np.array([.1,.1,.1]))
        behind = corners(np.array([-.1,-.1,-2.]), np.array([.1,.1,-1.]))
        with patch.object(C, 'CORNERS', np.stack([crossing, behind])), \
                patch.object(C, 'INDEX', np.zeros((2,3), int)):
            masks, _ = C.cell_masks(q)
        np.testing.assert_array_equal(masks, [[True,True,True],[False,False,False]])

    def test_all_corners_must_share_one_exposure_certificate(self):
        cube = corners(np.array([-.39,-.01,.99]), np.array([.39,.01,1.01]))
        # Two translated exposures cover different corners; their corner union
        # covers the cell, but no exposure certifies the complete finite cell.
        translations = (-.1, .1)
        inside = []
        for shift in translations:
            local = cube-np.array([shift,0.,0.])
            inside.append((np.abs(local[:,:2]/local[:,2:]) <= C.EDGE).all(-1))
        self.assertTrue(np.any(inside, axis=0).all())
        self.assertFalse(np.any(np.all(inside, axis=1)))
        noisy = np.repeat(np.eye(4)[None], 2, axis=0)
        noisy[0,0,3] = -.1; noisy[1,0,3] = .1
        query = np.eye(4); query[0,3] = .1
        z = np.zeros((3,2,8,8,16), np.float16)
        with patch.object(C, 'INDEX', np.array([[0,0,0]])), \
                patch.object(C, 'CORNERS', cube[None]), patch.object(C, 'SHAPE', (1,1,1)), \
                patch.object(C, 'cell_masks', return_value=(np.array([[False,True,False]]),np.eye(4))):
            rec = C.support(z, noisy, query, 1, (0,))
        self.assertFalse(rec['coverage'][1])
        self.assertEqual(rec['coverage_fraction'][1], 0.)
        self.assertEqual(rec['mask'], 0)

    def test_empty_supported_cell_and_unknown_cannot_veto(self):
        cube = corners(np.array([-.1,-.1,.9]), np.array([.1,.1,1.1]))
        with patch.object(C, 'INDEX', np.array([[0,0,0]])), \
                patch.object(C, 'CORNERS', cube[None]), patch.object(C, 'SHAPE', (1,1,1)), \
                patch.object(C, 'cell_masks', return_value=(np.array([[False,True,False]]),np.eye(4))):
            rec = C.support(np.zeros((3,1,8,8,16),np.float16), np.eye(4)[None], np.eye(4), 0, (0,))
        self.assertTrue(rec['coverage'][1])
        self.assertEqual(C.choose(1,rec), (-1,1,2))
        unknown = dict(mask=0,state=0,coverage=np.zeros(3,bool))
        self.assertEqual(C.choose(1,unknown), (1,1,2))
        unknown['coverage'][1] = True
        self.assertEqual(C.choose(1,unknown), (-1,1,2))

    def test_positive_unique_switch_and_ambiguous_fallback(self):
        # A positive unique cluster can switch despite incomplete coverage;
        # this does not claim that unseen alternatives are absent.
        rec = dict(mask=4,state=1,coverage=np.zeros(3,bool))
        self.assertEqual(C.choose(1,rec), (1,2,4))
        rec.update(mask=6,state=2)
        self.assertEqual(C.choose(1,rec), (1,1,6))
        rec.update(mask=5,state=3)
        self.assertEqual(C.choose(1,rec), (1,1,5))

    def test_actual_inline_set_evaluator_requires_subset(self):
        # Exercise the runner's exact inline expressions without loading truth
        # or invoking its full evaluator/file-writing pipeline.
        tree = ast.parse(inspect.getsource(C.evaluate))
        selected = {}
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name = node.targets[0].id
                if name in ('clean','unique'): selected[name] = node
        self.assertEqual(set(selected), {'clean','unique'})
        module = ast.fix_missing_locations(ast.Module(body=[selected['clean'],selected['unique']],type_ignores=[]))
        values = dict(mask=np.array([3,2,1,0,7,4]),legal_bits=np.array([2,2,3,3,7,2]),
                      legal=np.array([[0,1,0],[0,1,0],[1,1,0],[1,1,0],[1,1,1],[0,1,0]],bool))
        exec(compile(module,'inline_set_evaluator_fixture','exec'),{},values)
        np.testing.assert_array_equal(values['clean'], [False,True,True,False,True,False])
        np.testing.assert_array_equal(values['unique'], [False,True,False,False,False,False])


if __name__ == '__main__':
    unittest.main()
