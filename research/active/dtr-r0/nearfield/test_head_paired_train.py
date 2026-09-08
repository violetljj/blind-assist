"""Focused schedule, supervision-boundary and gradient tests; no model fitting."""
import copy
import unittest
import numpy as np
import torch
from city_data import pixel_support_bce
from head_paired_train import (train_units, validate_pairs, make_schedule,
    consistency_losses, supervised_loss)


def fixture():
    records = []
    for u in range(16):
        for relation in ('CLEAR', 'HEAD_ONLY'):
            for appearance in ('A', 'L', 'M'):
                records.append(dict(sample_index=len(records), group_id=f'g{u}',
                    source_partition='train' if u < 8 else 'eval',
                    source_role='TRAIN_ONLY' if u < 8 else 'EVAL_ONLY',
                    relation=relation, appearance=appearance, original_case_name=f'g{u}-{relation}',
                    camera={'x': u}, native_mask_sha256=('a' if relation == 'CLEAR' else 'b')*64,
                    native_rgb_sha256=appearance*64))
    groups = [f'g{u}' for u in range(16)]
    units = train_units(records, groups)
    near = np.tile(np.array([[0, 0]]*3+[[0, 1]]*3, dtype=np.float32), (8, 1))
    support = np.zeros((48, 2, 18, 32), dtype=np.int8)
    support[near[:, 1] == 1, 1, 0, 0] = 1
    return records, groups, units, near, support


class BoundaryTests(unittest.TestCase):
    def test_exact_schedule_and_train_only_complete_units(self):
        records, groups, units, y, s = fixture()
        validate_pairs(records, units, y, s)
        first, second = make_schedule(units), make_schedule(units)
        for name in first:
            self.assertEqual(first[name].tobytes(), second[name].tobytes())
        self.assertEqual(first['original_indices'].shape, (2000, 20))
        self.assertTrue(np.all(first['original_indices'] < 1198))
        self.assertEqual(first['paired_sample_indices'].shape, (2000, 12))
        self.assertTrue(np.all(first['paired_sample_indices'] < 48))
        self.assertTrue(np.all(first['unit_indices'][:, 0] != first['unit_indices'][:, 1]))
        self.assertEqual(int(sum(np.bincount(first['paired_local_indices'].reshape(-1), minlength=48))), 24000)
        for row in first['paired_sample_indices'].reshape(-1, 6):
            selected = [records[int(i)] for i in row]
            self.assertEqual(len({r['group_id'] for r in selected}), 1)
            self.assertEqual([(r['relation'], r['appearance']) for r in selected],
                [(r, a) for r in ('CLEAR', 'HEAD_ONLY') for a in ('A', 'L', 'M')])

    def test_eval_and_wrong_source_units_rejected(self):
        records, groups, units, y, s = fixture()
        bad = copy.deepcopy(records); bad[0]['source_partition'] = 'eval'
        with self.assertRaisesRegex(ValueError, '48 TRAIN'): train_units(bad, groups)
        with self.assertRaisesRegex(ValueError, 'first eight'): train_units(records, groups[::-1])
        bad = copy.deepcopy(records); bad[0]['source_role'] = 'EVAL_ONLY'
        with self.assertRaisesRegex(ValueError, 'role mismatch'): train_units(bad, groups)

    def test_native_unknown_and_geometry_mismatch_rejected(self):
        records, groups, units, y, s = fixture()
        bad = copy.deepcopy(records); bad[1]['native_mask_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'hashes unequal'): validate_pairs(bad, units, y, s)
        wrong = s.copy(); wrong[1, 0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'UNKNOWN mask'): validate_pairs(records, units, y, wrong)
        bad = copy.deepcopy(records); bad[1]['camera']['x'] += 1
        with self.assertRaisesRegex(ValueError, 'Camera changed'): validate_pairs(bad, units, y, s)
        wrong = y.copy(); wrong[1, 1] = 1
        with self.assertRaisesRegex(ValueError, 'Unexpected CLEAR'): validate_pairs(records, units, wrong, s)


class LossTests(unittest.TestCase):
    def test_class_balance_and_common_known_gradient(self):
        near = torch.zeros(32, 2, requires_grad=True)
        ny = torch.full((32, 2), -1.)
        probability = torch.full((32, 2, 2, 5), .5)
        probability[21, 1] = .6; probability[21, 1, 0, 0] = .9
        support = torch.logit(probability).requires_grad_()
        sy = torch.full((32, 2, 2, 5), -1, dtype=torch.int8)
        sy[20:22, 1] = 0; sy[20:22, 1, 0, 0] = 1
        nl, sl = consistency_losses(near, support, ny, sy)
        self.assertAlmostEqual(float(nl.detach()), 0.)
        # One positive gets the same class weight as nine negative pixels.
        self.assertAlmostEqual(float(sl.detach()), (.4**2+.1**2)/2, places=6)
        (nl+sl).backward()
        self.assertTrue(torch.all(near.grad == 0))
        self.assertTrue(torch.all(support.grad[sy == -1] == 0))
        self.assertGreater(float(support.grad[21, 1].abs().sum()), 0.)

    def test_no_geometry_invariance_and_supervision_prevents_collapse(self):
        # Large CLEAR/HEAD difference is allowed: every A/L/M within relation agrees.
        near = torch.zeros(32, 2, requires_grad=True)
        sy = torch.zeros(32, 2, 2, 3, dtype=torch.int8)
        ny = torch.zeros(32, 2)
        with torch.no_grad():
            for base in (20, 26):
                near[base+3:base+6, 1] = 4
                ny[base+3:base+6, 1] = 1
        maps = torch.zeros(32, 2, 2, 3, requires_grad=True)
        nl, sl = consistency_losses(near, maps, ny, sy)
        self.assertEqual(float((nl+sl).detach()), 0.)
        collapsed = torch.full((32, 2), -4., requires_grad=True)
        loss, _, _ = supervised_loss(collapsed, maps, ny, sy, pixel_support_bce)
        loss.backward()
        self.assertLess(float(collapsed.grad[23, 1]), 0.)
        self.assertGreater(float(collapsed.grad[20, 1]), 0.)

    def test_unknown_supervised_near_zero_gradient_and_legacy_equivalence(self):
        n = torch.randn(32, 2, requires_grad=True)
        m = torch.randn(32, 2, 2, 3, requires_grad=True)
        y = torch.randint(0, 2, (32, 2)).float()
        s = torch.randint(-1, 2, (32, 2, 2, 3), dtype=torch.int8)
        loss, _, _ = supervised_loss(n, m, y, s, pixel_support_bce)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(n, y)+.25*pixel_support_bce(m, s)
        self.assertTrue(torch.equal(loss, expected))
        y[0, 0] = -1
        loss, _, _ = supervised_loss(n, m, y, s, pixel_support_bce)
        loss.backward()
        self.assertEqual(float(n.grad[0, 0]), 0.)
        self.assertTrue(torch.all(m.grad[s == -1] == 0))

    def test_cross_relation_labels_rejected(self):
        n = torch.zeros(32, 2); m = torch.zeros(32, 2, 2, 3)
        y = torch.zeros(32, 2); s = torch.zeros_like(m)
        y[21, 1] = 1
        with self.assertRaisesRegex(ValueError, 'label relation'): consistency_losses(n, m, y, s)
        y.zero_(); s[21, 1, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'label relation'): consistency_losses(n, m, y, s)


if __name__ == '__main__':
    unittest.main()
