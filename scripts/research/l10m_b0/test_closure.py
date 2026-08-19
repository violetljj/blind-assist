from __future__ import annotations

import unittest

from .b0d_arrival_orthogonality import B0C_FROZEN_VERDICT, B0D_CONFIRMED_VERDICT
from .closure import PROTOCOL_ID, VERDICT, build_closure_manifest


class L10MB0ClosureTest(unittest.TestCase):
    def test_b0_closes_without_b0e_and_preserves_b0c_b0d_verdicts(self) -> None:
        result = build_closure_manifest()
        self.assertEqual(result["protocol_id"], PROTOCOL_ID)
        self.assertEqual(result["verdict"], VERDICT)
        self.assertFalse(result["b0_e_required"])
        self.assertEqual(
            result["frozen_verdicts"],
            {"b0c": B0C_FROZEN_VERDICT, "b0d": B0D_CONFIRMED_VERDICT},
        )

    def test_all_frozen_semantic_probes_hold(self) -> None:
        result = build_closure_manifest()
        self.assertTrue(all(result["probe_receipts"]["b0d_invariants"].values()))
        unknown = result["probe_receipts"]["unknown_does_not_fabricate_state"]
        self.assertIsNone(unknown["stuck_detection_step"])
        self.assertFalse(unknown["success"])


if __name__ == "__main__":
    unittest.main()
