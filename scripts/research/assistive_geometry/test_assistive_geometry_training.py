import unittest

import numpy as np
import torch

from scripts.research.assistive_geometry.assistive_geometry_training import (
    FrozenA0Scheduler,
    a0_lr_multiplier,
    build_epoch_effective_batches,
    parent_balanced_epoch_order,
)


def frames() -> list[dict]:
    rows = []
    for parent in range(16):
        for index in range(300):
            rows.append({
                "video_id": f"v{parent:02d}",
                "orientation_family": "portrait" if (parent * 300 + index) < 2724 else "landscape",
            })
    return rows


class AssistiveGeometryTrainingTest(unittest.TestCase):
    def test_parent_balanced_order_is_deterministic_and_complete(self) -> None:
        rows = frames()
        first = parent_balanced_epoch_order(rows, 17, 0)
        second = parent_balanced_epoch_order(rows, 17, 0)
        self.assertEqual(first, second)
        self.assertEqual(list(range(len(rows))), sorted(first))
        for start in range(0, len(rows), 16):
            parents = {rows[index]["video_id"] for index in first[start : start + 16]}
            self.assertEqual(16, len(parents))

    def test_twenty_epoch_orientation_carry_closes_exactly(self) -> None:
        rows = frames()
        carry = {"portrait": [], "landscape": []}
        steps = 0
        orientation_samples = {"portrait": 0, "landscape": 0}
        for epoch in range(20):
            batches, carry = build_epoch_effective_batches(rows, 17, epoch, carry)
            steps += len(batches)
            for orientation, batch in batches:
                self.assertEqual(16, len(batch))
                self.assertTrue(all(rows[index]["orientation_family"] == orientation for index in batch))
                orientation_samples[orientation] += len(batch)
        self.assertEqual(6000, steps)
        self.assertEqual({"portrait": [], "landscape": []}, carry)
        self.assertEqual(2724 * 20, orientation_samples["portrait"])
        self.assertEqual(2076 * 20, orientation_samples["landscape"])

    def test_schedule_boundaries_and_resume(self) -> None:
        self.assertAlmostEqual(1 / 300, a0_lr_multiplier(1))
        self.assertEqual(1.0, a0_lr_multiplier(300))
        self.assertAlmostEqual(0.05, a0_lr_multiplier(6000))
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=2e-5)
        scheduler = FrozenA0Scheduler(optimizer)
        scheduler.prepare_next_step()
        scheduler.mark_completed()
        state = scheduler.state_dict()
        resumed = FrozenA0Scheduler(optimizer)
        resumed.load_state_dict(state)
        self.assertEqual(1, resumed.completed_steps)


if __name__ == "__main__":
    unittest.main()
