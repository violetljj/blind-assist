"""Replay identity, split, sampling and retention contracts; no fitting."""
import copy
import unittest
import numpy as np
from adapt_city_native import recipe_config, replay_samples, replay_schedule, thin_pool, retention


def corpus():
    samples = []
    for split, count in (('train', 80), ('val', 16), ('test', 16)):
        for i in range(count):
            samples.append(dict(sample_id=f'{split}_{i}', split=split,
                group_id=f'{split}_group{i//4}', frame_indices=[len(samples)]))
    return dict(samples=samples)


class ReplayRecipeTest(unittest.TestCase):
    def test_saved_order_train_only_and_val16(self):
        data = corpus()
        # Interspersed VAL records cannot affect the saved TRAIN order.
        data['samples'].insert(0, data['samples'].pop(80))
        selected, regression = replay_samples(data)
        self.assertEqual([s['sample_id'] for s in selected], [f'train_{i}' for i in range(64)])
        self.assertEqual(len(regression), 16)
        self.assertTrue(all(s['split'] == 'train' for s in selected))

    def test_reject_shared_group_sample_or_frame(self):
        for key in ('group_id', 'sample_id', 'frame_indices'):
            with self.subTest(key=key):
                data = corpus()
                data['samples'][80][key] = data['samples'][0][key]
                with self.assertRaises(ValueError):
                    replay_samples(data)

    def test_corpus_requires_64_train(self):
        data = corpus()
        data['samples'] = [s for s in data['samples'] if s['split'] != 'train'] + data['samples'][:63]
        with self.assertRaises(ValueError):
            replay_samples(data)

    def test_mixed_schedule_source_counts_and_determinism(self):
        thin = [2, 7, 19]
        schedule = replay_schedule(45, thin)
        np.testing.assert_array_equal(schedule, replay_schedule(45, thin))
        self.assertEqual(schedule.shape, (300, 16))
        self.assertTrue(np.isin(schedule[:, :4], thin).all())
        self.assertTrue(((schedule[:, 4:8] >= 0) & (schedule[:, 4:8] < 45)).all())
        self.assertTrue(((schedule[:, 8:] >= 45) & (schedule[:, 8:] < 109)).all())
        self.assertEqual(int((schedule < 45).sum()), 2400)
        self.assertEqual(int((schedule >= 45).sum()), 2400)
        self.assertTrue(np.isin(schedule[:, 4:8], thin).any())
        self.assertEqual(set(schedule[:, 8:].ravel()), set(range(45, 109)))

    def test_thin_pool_requires_real_known_positive(self):
        mask = np.full((4, 2, 360, 640), -1, dtype=np.int8)
        near = np.array([[1, 0], [1, 0], [-1, 0], [0, 0]], dtype=np.int8)
        mask[0, 0, 2, 2] = 1
        mask[2, 0, 2, 2] = 1
        mask[3, 0, 2, 2] = 1
        np.testing.assert_array_equal(thin_pool(near, mask), [0])
        with self.assertRaises(ValueError):
            thin_pool(near, np.full_like(mask, -1))

    def test_recipe_does_not_mutate_default(self):
        replay = recipe_config('replay_thin')
        original = recipe_config('native_only')
        self.assertEqual(original['sampling'], 'Uniform TRAIN replacement seed17')
        self.assertNotIn('batch_source_counts', original)
        for key in ('seed', 'steps', 'batch_size', 'head_lr', 'backbone_lr', 'weight_decay', 'support_weight'):
            self.assertEqual(original[key], replay[key])

    def test_replay_retention_rejects_localization_forgetting(self):
        heads = {h:dict(near=dict(recall=1., FP=0, TN=8),
            support=dict(positive_frame_mean_iou=.5)) for h in ('BODY', 'HEAD')}
        target = dict(target_id='bollard', methods=dict(g13=dict(BODY=dict(
            reliable_positive_frames=2, alert_FN=0, joint_alert_and_overlap_hits=2))))
        metric = dict(heads=heads, targets=[target])
        domains = {d:{p:copy.deepcopy(metric) for p in ('historical', 'dev_selected')}
            for d in ('dev', 'consumed_route40', 'willow_val16')}
        result = {a:copy.deepcopy(domains) for a in ('initial', 'adapted')}
        lost = dict(historical=0, dev_selected=0)
        self.assertEqual(retention(result, lost, 'replay_thin')['disposition'], 'DEVELOPMENT_CHALLENGER')
        result['adapted']['willow_val16']['dev_selected']['heads']['BODY']['support']['positive_frame_mean_iou'] = .39
        self.assertEqual(retention(result, lost, 'replay_thin')['disposition'], 'RETAIN_ORIGINAL')
        self.assertEqual(retention(result, lost, 'native_only')['disposition'], 'DEVELOPMENT_CHALLENGER')


if __name__ == '__main__':
    unittest.main()
