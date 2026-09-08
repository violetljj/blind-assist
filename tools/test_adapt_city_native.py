"""Narrow checks of the frozen fit's supervision and DEV selection contract."""
import unittest
import numpy as np
import torch
from adapt_city_native import masked_near_bce, select_thresholds, metrics, route
from city_native_targets import ray_status


class AdaptationContract(unittest.TestCase):
    def test_same_instance_collision_envelope_is_not_occlusion(self):
        # Native lamp sample: original component/instance hits 16.5cm too early.
        self.assertEqual(ray_status(True, .7473565104455211, .9126062989234924), 'UNKNOWN')
        self.assertEqual(ray_status(False, .7473565104455211, .9126062989234924), 'OCCLUDED')
        self.assertEqual(ray_status(True, .9, .9126062989234924), 'MATCH')

    def test_unknown_near_has_zero_gradient(self):
        logits = torch.tensor([[.3, -.5], [-.2, .8]], requires_grad=True)
        target = torch.tensor([[1., -1.], [0., -1.]])
        masked_near_bce(logits, target).backward()
        self.assertEqual(logits.grad[:, 1].abs().sum().item(), 0)
        self.assertGreater(logits.grad[:, 0].abs().sum().item(), 0)

    def test_empirical_fpr_and_ties(self):
        labels = np.array([[1, 1]] * 10 + [[0, 0]] * 10)
        probability = np.repeat(np.array([.8] * 9 + [.2] + [.7] + [.1] * 9)[:, None], 2, axis=1)
        choice = select_thresholds(probability, labels)[0]
        self.assertEqual(choice['value'], .2)
        self.assertEqual((choice['TP'], choice['FP']), (10, 1))
        probability[:10] = .8
        choice = select_thresholds(probability, labels)[0]
        self.assertEqual(choice['value'], .8)
        self.assertEqual(choice['FP'], 0)

    def test_missing_classes_not_evaluable(self):
        labels = np.array([[1, -1]] * 8 + [[0, -1]] * 8)
        choices = select_thresholds(np.full((16, 2), .5), labels)
        self.assertEqual(choices[1]['status'], 'NOT_EVALUABLE')
        self.assertIsNone(choices[1]['value'])
        self.assertGreater(choices[0]['value'], 1)

    def test_positive_wins_unknown_pool(self):
        raw = np.zeros((1, 2, 360, 640), np.int8)
        raw[0, 0, 0, 0] = -1
        raw[0, 0, 1, 1] = 1
        raw[0, 1, 0, 0] = -1
        pooled = route.pooled(raw)
        self.assertEqual(pooled[0, 0, 0, 0], 1)
        self.assertEqual(pooled[0, 1, 0, 0], -1)


if __name__ == '__main__':
    unittest.main()
