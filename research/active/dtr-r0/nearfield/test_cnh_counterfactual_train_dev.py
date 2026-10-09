"""Focused matched-schedule and zero-channel cache restoration checks."""
import unittest

import numpy as np
import torch

from cnh_counterfactual_train_dev import KEPT_CHANNELS, restore_features, slot_schedule
from cnh_boundary_token_model import BoundaryTokenReadout


class MatchedTrainTests(unittest.TestCase):
    def test_matched_schedule_and_base_extra_sampling(self):
        order, base = slot_schedule(2026100955, 0, 12, 6)
        np.testing.assert_array_equal(np.sort(order), np.arange(18))
        np.testing.assert_array_equal(base[order < 12], order[order < 12])
        extra = base[order >= 12]
        assert len(np.unique(extra)) == 6
        assert np.all((0 <= base) & (base < 12))
        again_order, again_base = slot_schedule(2026100955, 0, 12, 6)
        np.testing.assert_array_equal(order, again_order)
        np.testing.assert_array_equal(base, again_base)
        later_order, _ = slot_schedule(2026100955, 1, 12, 6)
        assert not np.array_equal(order, later_order)
        # Ordinary and cf are offset bindings of the exact same new slots.
        ordinary, cf = order.copy(), order.copy()
        ordinary[order >= 12] += 0
        cf[order >= 12] += 6
        assert np.array_equal(ordinary[order < 12], cf[order < 12])
        np.testing.assert_array_equal(cf[order >= 12]-ordinary[order >= 12], np.full(6, 6))

    def test_nine_channel_restore(self):
        compact = torch.arange(2*16*9*8*8, dtype=torch.float32).reshape(1, 2, 16, 9, 8, 8).half()
        full = restore_features(compact)
        assert full.shape == (1, 2, 16, 15, 8, 8)
        assert full.dtype == torch.float32
        assert torch.count_nonzero(full[:, :, :, 6:12]) == 0
        torch.testing.assert_close(full[:, :, :, list(KEPT_CHANNELS)], compact.float(), rtol=0, atol=0)

    def test_production_steps_and_source_binding(self):
        old_rows, new_rows, batch, epochs = 39936, 19968, 256, 32
        order, base = slot_schedule(2026100955, 0, old_rows, new_rows)
        assert len(order) == 59904
        assert epochs*((len(order)+batch-1)//batch) == 7488
        assert np.all(base < old_rows)
        for arm_offset in (old_rows, old_rows+new_rows):
            bound = order.copy()
            new = bound >= old_rows
            bound[new] = arm_offset+bound[new]-old_rows
            np.testing.assert_array_equal(bound[~new], order[~new])
            np.testing.assert_array_equal(np.sort(bound[new]), np.arange(arm_offset, arm_offset+new_rows))

    def test_restore_preserves_fixed_model_function(self):
        torch.manual_seed(1)
        full = torch.randn(1, 2, 16, 15, 8, 8).half()
        compact = full[:, :, :, list(KEPT_CHANNELS)]
        full[:, :, :, 6:12] = 0
        restored = restore_features(compact)
        torch.testing.assert_close(restored, full.float(), rtol=0, atol=0)
        model = BoundaryTokenReadout().eval()
        with torch.no_grad():
            torch.testing.assert_close(model(restored, [8]), model(full, [8]), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
