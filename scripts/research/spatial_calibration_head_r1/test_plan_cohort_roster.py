#!/usr/bin/env python3

import json
import unittest

from plan_cohort_roster import build_roster, load_rows
from validate_protocol import DEFAULT_PROTOCOL


class CohortRosterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        source = DEFAULT_PROTOCOL.parents[3] / "artifacts.local/downloads/ARKitScenes-7283761/raw/raw_train_val_splits.csv"
        cls.rows = load_rows(source)

    def test_official_metadata_yields_disjoint_24_parent_roster(self) -> None:
        result = build_roster(self.rows, self.protocol)
        self.assertEqual(result["selected_parent_count"], 24)
        self.assertEqual(result["role_visit_overlap"], [])
        self.assertEqual(result["source_inventory"]["cross_official_fold_visits_excluded"], ["381879"])
        self.assertEqual([len(result["roles"][role]) for role in ("train", "validation", "sealed")], [16, 4, 4])
        self.assertEqual(sorted(row["cv_fold"] for row in result["roles"]["train"]), [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)

    def test_selection_is_deterministic(self) -> None:
        self.assertEqual(build_roster(self.rows, self.protocol), build_roster(list(reversed(self.rows)), self.protocol))


if __name__ == "__main__":
    unittest.main()
