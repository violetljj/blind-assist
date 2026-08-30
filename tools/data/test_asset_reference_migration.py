from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_PATH = REPO_ROOT / "data" / "asset-reference-migration-20260831.json"


class AssetReferenceMigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.migration = json.loads(MIGRATION_PATH.read_text(encoding="utf-8"))
        self.entries = self.migration["entries"]

    def test_all_baseline_unresolved_references_have_one_disposition(self) -> None:
        self.assertEqual("blindassist-asset-reference-migration-v1", self.migration["schema"])
        self.assertEqual(106, len(self.entries))
        keys = {
            (item["source_file"], item["baseline_line"], item["old_locator"])
            for item in self.entries
        }
        self.assertEqual(106, len(keys))
        resolution_counts = Counter(item["old_resolution"] for item in self.entries)
        self.assertEqual(
            {"missing_within_root": 80, "unknown_root": 26},
            dict(resolution_counts),
        )
        classification_counts = Counter(item["classification"] for item in self.entries)
        self.assertEqual(
            self.migration["classification_counts"],
            dict(classification_counts),
        )

    def test_backlog_is_concrete_and_aliases_are_relative(self) -> None:
        backlog = [item for item in self.entries if item["actionability"] == "backlog"]
        self.assertEqual(8, len(backlog))
        for item in backlog:
            self.assertTrue(item.get("reason"), item)
            self.assertTrue(item.get("next_action"), item)
        for item in self.entries:
            canonical = item.get("canonical_locator")
            if canonical is None:
                continue
            self.assertFalse(Path(canonical).is_absolute(), item)
            self.assertNotIn("//", canonical, item)
            self.assertNotIn("artifacts.local", canonical.casefold(), item)

    def test_active_metrics_keep_missing_dependencies_visible(self) -> None:
        active_missing = [
            item
            for item in self.entries
            if item["classification"] == "active_missing_dependency_property_gated"
        ]
        self.assertEqual(3, len(active_missing))
        self.assertEqual(
            0,
            self.migration["active_metrics"]["stale_hardcoded_active_callers_after_migration"],
        )
        self.assertEqual(
            len(active_missing),
            self.migration["active_metrics"][
                "unavailable_active_dependencies_requiring_explicit_property_or_rebuild"
            ],
        )
        post = self.migration["post_migration_report"]
        self.assertEqual(11, post["unresolved_repository_references"])
        self.assertEqual(
            post["unresolved_repository_references"],
            sum(post["unresolved_breakdown"].values()),
        )


if __name__ == "__main__":
    unittest.main()
