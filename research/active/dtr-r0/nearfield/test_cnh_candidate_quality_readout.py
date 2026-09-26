"""Focused mask/covariance/causal-association checks, no experiment outcomes."""
import unittest
import numpy as np
import cnh_candidate_quality_readout as q


class QualityReadoutTests(unittest.TestCase):
    def test_masks_nested_and_query_independent(self):
        oid = np.full((2, 8, 8, 256), 100, int)
        oid[:, 4:] = 101
        oid[:, 2, 3, :8] = 1
        objects = [dict(id=1, category='LOW'), dict(id=100, category='BACKGROUND'), dict(id=101, category='BACKGROUND')]
        masks, stats = q.candidate_masks(oid, objects, 96, 0, [np.array([[0, 0], [0, 1]])])
        self.assertEqual(stats['visible_candidates'], 2)
        for small, big in (('G1', 'FA1'), ('FA1', 'FA3'), ('FA3', 'FA10'), ('G1', 'DILATE05'),
                           ('DILATE05', 'DILATE1'), ('DROP40', 'DROP20'), ('DROP20', 'G1'), ('BGFA1', 'BGFA3')):
            self.assertFalse((masks[small] & ~masks[big]).any(), (small, big))
        self.assertEqual(int(masks['G1'].sum()), 32)
        self.assertEqual(stats['background']['requested'], 20)
        self.assertEqual(stats['background']['generated'], 20)
        self.assertGreater(stats['background']['boundary_placed'], 0)

    def test_half_zone_dilation_is_not_whole_zone_dilation(self):
        pix = np.zeros((128, 128), bool)
        pix[40, 40] = True
        half = q.to_zones(pix, .5)
        whole = q.to_zones(pix, 1.)
        self.assertGreater(int(whole.sum()), int(half.sum()))
        self.assertFalse((half & ~whole).any())

    def test_physical_candidate_requires_approach_and_past_pair(self):
        import torch
        import cnh_track_a_gpu_readout as g
        cells = torch.zeros((4, 8, 8, 16), dtype=torch.bool, device=g.DEV)
        cells[0:2, 3, 3, 5] = True
        poses = torch.eye(4, dtype=g.D64, device=g.DEV).repeat(4, 1, 1)
        empty, _ = q.temporal_mask(cells, poses, g)
        self.assertFalse(empty.any())
        poses[:, 2, 3] = torch.arange(4, device=g.DEV)*.1
        local, _ = q.temporal_mask(cells, poses, g)
        self.assertTrue(local[2].any())
        self.assertFalse(local[:2].any())
        self.assertFalse(local[3].any())
        cells[3].fill_(True)
        future, _ = q.temporal_mask(cells, poses, g)
        self.assertTrue(torch.equal(local, future))

    def test_score_parity_with_previous_implementation(self):
        import cnh_position_prior_readout as old
        rng = np.random.default_rng(33)
        hist, ambient, bias = rng.normal(0, 5, (3, 8, 8, 16)), np.ones((3, 8, 8)), np.zeros((8, 8, 16))
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 2, 3] = np.arange(3)*.1
        mask = rng.random(hist.shape) > .5
        ours, _ = q.sequence_scores(hist, ambient, bias, poses, poses, {a: mask for a in q.SPECS})
        previous = old.sequence_scores(hist, ambient, bias, poses, poses, {a: mask for a in old.ARMS[1:-1]})
        np.testing.assert_array_equal(ours['G0'], previous['G0'])
        np.testing.assert_array_equal(ours['G1'], previous['G1'])
        no_oracle, _ = q.sequence_scores(hist, ambient, bias, poses, poses, {a: np.zeros_like(mask) for a in q.SPECS})
        np.testing.assert_array_equal(ours['G4prime_local'], no_oracle['G4prime_local'])


if __name__ == '__main__':
    unittest.main()
