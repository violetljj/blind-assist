import unittest
from contact_retina_spec import BODY_BOXES, first_contact, specification


class ContactSourceTest(unittest.TestCase):
    def test_continuous_contact_and_near_miss(self):
        obj = dict(center_m=[3., 0., 1.7], size_m=[.08, 1., .08])
        self.assertAlmostEqual(first_contact([0., 0., .12], 1., [obj], BODY_BOXES[1]), 2.83)
        self.assertIsNone(first_contact([0., 0., .12], 1., [obj], BODY_BOXES[0]))
        obj['center_m'][1] = 1.
        self.assertIsNone(first_contact([0., 0., .12], 1., [obj], BODY_BOXES[1]))

    def test_pair_split_causal_history_and_control_visibility(self):
        spec = specification()
        groups = {}
        for sample in spec['samples']:
            groups.setdefault(sample['group_id'], set()).add(sample['split'])
            indices = sample['frame_indices']
            self.assertEqual(indices, list(range(indices[-1]-7, indices[-1]+1)))
            frames = [spec['cases'][i] for i in indices]
            self.assertEqual(len({f['clip_id'] for f in frames}), 1)
            for row in spec['labels']['targets'][sample['sample_id']]:
                self.assertEqual(row, sorted(row))
            if '_v1_' in sample['sample_id']:
                self.assertEqual(sum(map(sum, spec['labels']['targets'][sample['sample_id']])), 0)
        self.assertTrue(all(len(s) == 1 for s in groups.values()))
        for clip in spec['clips']:
            target = clip['objects'][-1]
            self.assertGreater(target['center_m'][2]-target['size_m'][2]/2, .12)


if __name__ == '__main__':
    unittest.main()
