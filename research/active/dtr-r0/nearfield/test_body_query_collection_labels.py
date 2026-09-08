"""CPU-only checks; no CUDA discovery, model instance or capture execution."""
import unittest

import torch

from body_query_collection_labels import (BASIC, EXTRA, expected_near,
    floor_acceptance, ownership_fields, summarize)
from body_query_labels import labels
from body_query_model import fixed_projection, projection_weights
from body_query_ownership_audit import membership


class CollectionLabelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.grid, cls.valid, _ = fixed_projection()
        cls.weights = projection_weights(cls.grid)

    def test_partition_and_unknown_preserved(self):
        native = torch.full((360, 640), 1.2)
        native[:20, :20] = 0
        native[20, 20] = float('nan')
        camera = dict(x=200., y=-300., z=6.7, pitch=0., yaw=37., roll=0.)
        label = labels(native, camera, 5.)
        cells, known, _ = membership(native)
        self.assertTrue(torch.equal(cells.sum((-2, -1)), label['raw_counts']))
        self.assertTrue(torch.equal(label['counts'], label['raw_counts'].clamp_max(3)))
        self.assertTrue((label['support'][:, ~known] == -1).all())
        self.assertFalse(cells[:, ~known].any())
        self.assertTrue(torch.equal(label['near'].bool(), label['raw_counts'].reshape(2, 6).sum(1) >= 3))

    def test_unknown_footprint_matches_existing_r1_mask(self):
        cells = torch.zeros((12, 360, 640), dtype=torch.bool)
        known = torch.ones((360, 640), dtype=torch.bool)
        c, p = self.valid.nonzero()[0].tolist()
        weights = self.weights.reshape(12, 27, 576)
        k = int(weights[c, p].argmax())
        y, x = (k//32)*20, (k%32)*20
        known[y, x] = False
        # Positive witness remains in target, but does NOT override UNKNOWN.
        cells[c, y+1, x+1] = True
        fields = ownership_fields(cells, known, self.grid, self.valid, self.weights)
        import torch.nn.functional as F
        unknown = F.max_pool2d((~known).float()[None, None], 20).flatten()
        expected = (torch.einsum('cpk,k->cp', weights, unknown) == 0) & self.valid
        self.assertTrue(torch.equal(fields['local_mask'], expected))
        self.assertGreater(float(fields['local_membership_mass'][c, p]), 0)
        self.assertFalse(fields['local_mask'][c, p])
        self.assertGreater(float(fields['local_native_known_fraction_mass'][c, p]),
                           float(fields['local_known_mass'][c, p]))
        self.assertLess(float(fields['local_native_known_fraction_mass'][c, p]),
                        float(fields['local_projection_mass'][c, p]))
        self.assertFalse(fields['ray_known'][~self.valid].any())
        self.assertTrue((fields['ray_mask'][~self.valid] == -1).all())

    def test_all_unknown_has_no_supervision(self):
        fields = ownership_fields(torch.zeros((12, 360, 640), dtype=torch.bool),
            torch.zeros((360, 640), dtype=torch.bool), self.grid, self.valid, self.weights)
        self.assertFalse(fields['local_mask'].any())
        self.assertFalse(fields['ray_known'].any())
        self.assertEqual(float(fields['local_known_mass'].sum()), 0.)
        self.assertTrue((fields['ray_mask'] == -1).all())

    def test_group_coverage_uses_actual_range_and_keeps_failures(self):
        rows = []
        for i, variant in enumerate((*BASIC, *EXTRA)):
            near = expected_near(variant)
            counts = [0]*12
            counts[0], counts[9] = 3*near[0], 3*near[1]
            rows.append(dict(frame_id=i, region_id='r', group_id='g', family='crossbar',
                source_role='DEV_ONLY', declared_range='near', condition=variant,
                near=near, raw_counts=counts, status='PASS', intent_matches=True,
                floor_probe={'accepted': True}))
        result = summarize(rows)
        self.assertTrue(result['complete_group_acceptance'])
        group = result['groups'][0]
        self.assertFalse(group['HEAD_near_positive_group'])
        self.assertTrue(group['HEAD_far_positive_group'])
        self.assertEqual(group['HEAD_expected_positive_frames'], 2)
        rows[2].update(status='FAIL', error='test missing payload')
        rows[-1]['intent_matches'] = False
        result = summarize(rows)
        self.assertFalse(result['complete_group_acceptance'])
        self.assertEqual(result['counts']['region_id']['r']['declared_frames'], 8)
        self.assertEqual(result['counts']['region_id']['r']['failed_frames'], 1)
        self.assertEqual(result['groups'][0]['HEAD_expected_positive_frames'], 2)
        self.assertEqual(result['groups'][0]['HEAD_expected_positive_observed_frames'], 1)
        self.assertEqual(result['groups'][0]['intent_mismatch_frames'], [7])
        self.assertFalse(summarize(rows[:-1])['groups'][0]['complete'])
        with self.assertRaises(ValueError):
            expected_near('UNDECLARED')

    def test_floor_camera_point_tolerance_and_scope(self):
        case = dict(name='a', floor_z_m=12., camera=dict(x=100., y=-50.), floor_check=False)
        probe = dict(case='a', hit=True, point_m=[100., -50., 12.019])
        result = floor_acceptance(case, [probe])
        self.assertTrue(result['accepted'])
        self.assertFalse(result['continuous_floor_patch_enabled'])
        probe['point_m'][2] = 12.021
        self.assertFalse(floor_acceptance(case, [probe])['accepted'])
        self.assertFalse(floor_acceptance(case, [])['accepted'])
        self.assertFalse(floor_acceptance(case, [probe, probe])['accepted'])


if __name__ == '__main__':
    unittest.main()
