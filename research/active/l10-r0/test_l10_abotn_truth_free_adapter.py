from __future__ import annotations

import unittest

from l10_abotn_truth_free_adapter import (
    FORBIDDEN_POLICY_FIELDS,
    assert_truth_free,
    make_action_receipt,
    policy_field_names,
    strip_policy_truth,
)


class AbotnTruthFreeAdapterTest(unittest.TestCase):
    def test_truth_fields_are_not_projected_to_policy(self) -> None:
        raw = {
            "poi_name": "target",
            "images": {"front": b"frame"},
            "position": [1, 2, 3],
            "rotation": [[1, 0], [0, 1]],
            "heading": 0.2,
            "step_count": 4,
            "history_images": [],
            "history_poses": [],
            "target_position": [0.1, 4.0],
            "distance_to_goal": 4.1,
            "goal_world": [9, 9, 9],
            "occ_map": "private map",
            "extra": {"endpoint": [9, 9]},
        }
        clean = strip_policy_truth(raw)
        assert_truth_free(clean)
        self.assertFalse(policy_field_names(clean) & FORBIDDEN_POLICY_FIELDS)
        self.assertEqual(clean.poi_name, "target")
        self.assertEqual(clean.position, [1, 2, 3])

    def test_action_receipt_binds_arm_action_pose_and_images(self) -> None:
        receipt = make_action_receipt(
            episode_id="opaque-7",
            step_count=2,
            arm="TRIGGERED_ACTION",
            action_label="SIDESTEP_LEFT",
            requested_waypoint=(-0.3, 0.0),
            before_pose=[0, 0, 0],
            after_pose=[-0.3, 0, 0],
            before_image=b"before",
            after_image=b"after",
        )
        repeated = make_action_receipt(
            episode_id="opaque-7",
            step_count=2,
            arm="TRIGGERED_ACTION",
            action_label="SIDESTEP_LEFT",
            requested_waypoint=(-0.3, 0.0),
            before_pose=[0, 0, 0],
            after_pose=[-0.3, 0, 0],
            before_image=b"before",
            after_image=b"after",
        )
        self.assertEqual(receipt.receipt_digest, repeated.receipt_digest)
        self.assertNotEqual(receipt.before_image_digest, receipt.after_image_digest)


if __name__ == "__main__":
    unittest.main()
