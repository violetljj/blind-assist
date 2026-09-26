"""Geometry parity and soft-union fallback, without fitting any outcomes."""
import unittest
import numpy as np
from scipy.special import ndtr, ndtri
import cnh_soft_prior_readout as s
import cnh_candidate_quality_readout as q


class SoftReadoutTests(unittest.TestCase):
    def test_candidate_union_unchanged_and_confidence_monotone(self):
        oid = np.full((4, 8, 8, 256), 100, int)
        oid[:, 4:] = 101
        oid[:, 2, 3, :8] = 1
        objects = [dict(id=1, category='LOW'), dict(id=100, category='BACKGROUND'), dict(id=101, category='BACKGROUND')]
        pool = [np.array([[0, 0], [0, 1]])]
        old, _ = q.candidate_masks(oid, objects, 96, 0, pool)
        masks, fields, samples = s.candidate_fields(oid, objects, 96, 0, pool)
        fields = fields.reshape(len(s.CONDITIONS), len(s.AUCS), 4, 8, 8)
        for ai, arm in enumerate(s.CONDITIONS):
            np.testing.assert_array_equal(masks[arm], old[arm])
            np.testing.assert_array_equal(fields[ai, 0] > 0, masks[arm][..., 0])
            self.assertTrue((np.diff(fields[ai], axis=0) >= 0).all())
        self.assertTrue(all(row[0] == 1 for row in samples['G1']))
        self.assertTrue(any(row[0] == 0 for row in samples['FA1']))
        np.testing.assert_allclose(ndtr(np.sqrt(2)*ndtri(s.AUCS)/np.sqrt(2)), s.AUCS)

    def test_zero_delta_and_full_confidence_union(self):
        rng = np.random.default_rng(42)
        hist = rng.normal(0, 5, (3, 8, 8, 16))
        ambient, bias = np.ones((3, 8, 8)), np.zeros((8, 8, 16))
        poses = np.repeat(np.eye(4)[None], 3, 0)
        poses[:, 2, 3] = np.arange(3)*.1
        mask = rng.random(hist.shape[:3]) > .7
        masks = {a: np.broadcast_to(mask[..., None], hist.shape).copy() for a in s.CONDITIONS}
        fields = np.broadcast_to(mask, (len(s.CONDITIONS)*len(s.AUCS), *mask.shape)).copy()
        result = s.sequence_scores(hist, ambient, bias, poses, poses, masks, fields)
        old, _ = q.sequence_scores(hist, ambient, bias, poses, poses, {a: masks['G1'] for a in q.SPECS})
        np.testing.assert_array_equal(result['G0'], old['G0'])
        np.testing.assert_array_equal(result['G1'], old['G1'])
        for arm in s.CONDITIONS:
            for auc in s.AUCS:
                np.testing.assert_array_equal(result[s.key(arm, auc, 0)], result['G0'])
                for di, delta in enumerate(s.DELTAS):
                    expected = np.maximum(result['G0'], result[arm]+delta)
                    np.testing.assert_allclose(result[s.key(arm, auc, di)], expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
