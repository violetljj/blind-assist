#!/usr/bin/env python3

from __future__ import annotations

import unittest

from evaluate_grail_r1cg_g0 import evaluate, pose_transport_mode


def pair(pair_id: str, modes: list[str], yaw: float, object_type: str) -> dict:
    return {
        "pair_id": pair_id,
        "valid_slot_modes": modes,
        "relative_yaw_label_degrees": yaw,
        "object_type": object_type,
    }


class PoseTransportTest(unittest.TestCase):
    def test_fixed_geometric_boundary(self) -> None:
        self.assertEqual(pose_transport_mode(0.0), "PRESERVE")
        self.assertEqual(pose_transport_mode(89.0), "PRESERVE")
        self.assertEqual(pose_transport_mode(91.0), "FLIP")
        self.assertEqual(pose_transport_mode(-179.0), "FLIP")

    def test_evaluator_separates_flip_preserve_and_ambiguous(self) -> None:
        collection = {
            "houses": 2,
            "pairs": [
                pair("p0", ["PRESERVE"], 0.0, "Drawer"),
                pair("p1", ["FLIP"], 180.0, "Drawer"),
                pair("p2", ["PRESERVE"], 20.0, "Doorway"),
                pair("p3", ["FLIP"], -160.0, "Doorway"),
                pair("p4", ["PRESERVE", "FLIP"], 90.0, "Drawer"),
            ],
        }
        result = evaluate(collection)
        self.assertEqual(result["cohort"]["discriminative_pairs"], 4)
        self.assertEqual(result["cohort"]["ambiguous_pairs"], 1)
        self.assertEqual(result["arms"]["preserve_prior"]["flip_only"]["correct"], 0)
        self.assertEqual(result["arms"]["g0_pose_transport"]["discriminative"]["correct"], 4)
        self.assertTrue(result["decision"]["passed"])


if __name__ == "__main__":
    unittest.main()
