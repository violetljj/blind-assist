from __future__ import annotations

import unittest

from grail_relational_r0 import canonical_signature, select_with_relational_oracle


def signature(kind: str, side: str) -> dict:
    return {
        "semantic_type": kind, "support": "FLOOR_OR_STRUCTURE", "room_types": ("LivingRoom",),
        "height_band": "LOW", "part_horizontal": "SINGLE", "part_vertical": "SINGLE",
        "nearby": (("Sofa", side, "NEAR", "LEVEL"),),
    }


class RelationalOracleTest(unittest.TestCase):
    def test_unique_relation_overrides_appearance(self) -> None:
        target = signature("Chair", "RIGHT")
        candidates = [signature("Chair", "RIGHT"), signature("Chair", "LEFT")]
        selected, confidence, reason = select_with_relational_oracle(target, candidates, [0.1, 0.9], ["a", "b"])
        self.assertEqual((selected, confidence, reason), (0, 1.0, "UNIQUE_RELATION_MATCH"))

    def test_collision_uses_frozen_appearance_and_stable_id(self) -> None:
        target = signature("Chair", "RIGHT")
        candidates = [target, dict(target)]
        selected, confidence, reason = select_with_relational_oracle(target, candidates, [0.4, 0.8], ["a", "b"])
        self.assertEqual((selected, confidence, reason), (1, 0.8, "RELATION_COLLISION_APPEARANCE_TIEBREAK"))
        reverse = select_with_relational_oracle(target, list(reversed(candidates)), [0.8, 0.4], ["b", "a"])
        self.assertEqual("b", ["b", "a"][reverse[0]])

    def test_no_exact_relation_match_abstains(self) -> None:
        selected, confidence, reason = select_with_relational_oracle(
            signature("Chair", "RIGHT"), [signature("Chair", "LEFT")], [0.99], ["wrong"]
        )
        self.assertEqual((selected, confidence, reason), (None, 0.0, "NO_EXACT_RELATION_MATCH"))

    def test_signature_is_hashable_and_ordered(self) -> None:
        self.assertEqual(canonical_signature(signature("Chair", "RIGHT"))[0], "Chair")


if __name__ == "__main__":
    unittest.main()
