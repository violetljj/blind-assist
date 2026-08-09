from __future__ import annotations

import unittest

from scripts.research.assistive_geometry_qsf.audit_h1_parent_support import (
    REQUIRED_SUPPORT_KEYS,
    add_counts,
    choose_eval_parents,
)


class H1ParentSupportAuditTest(unittest.TestCase):
    def test_eval_selection_is_manifest_order_and_nonzero_support_only(self) -> None:
        order = ["p0", "p1", "p2", "p3", "p4", "p5"]
        support = {
            parent: {key: 1 for key in REQUIRED_SUPPORT_KEYS}
            for parent in order
        }
        support["p1"]["right_censor_count"] = 0
        selected, eligible = choose_eval_parents(order, support, 4)
        self.assertEqual(["p0", "p2", "p3", "p4"], selected)
        self.assertEqual(["p0", "p2", "p3", "p4", "p5"], eligible)

    def test_add_counts_preserves_every_support_denominator(self) -> None:
        first = {key: index + 1 for index, key in enumerate(REQUIRED_SUPPORT_KEYS)}
        second = {key: 10 * (index + 1) for index, key in enumerate(REQUIRED_SUPPORT_KEYS)}
        self.assertEqual(
            {key: first[key] + second[key] for key in REQUIRED_SUPPORT_KEYS},
            add_counts([first, second]),
        )


if __name__ == "__main__":
    unittest.main()
