"""Contract tests for the evidence-bound failure-case album renderer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.research.failure_case_atlas.batch_album import (
    CATEGORY_CONFIG_PATH,
    CATEGORY_SPECS,
    _category_specs,
    _classify_frame,
    _event_phase,
    _load_category_config,
    _missing_panel,
)


class BatchAlbumContractTest(unittest.TestCase):
    def test_requested_categories_have_stable_slugs(self) -> None:
        self.assertTrue(CATEGORY_CONFIG_PATH.is_file())
        labels = [label for _, label in CATEGORY_SPECS]
        self.assertEqual(
            labels[:4],
            ["漏检", "误检", "晚提醒", "提醒无法清除"],
        )
        self.assertEqual(CATEGORY_SPECS[-1][0], "other")

    def test_classification_keeps_truth_and_explicit_mechanism_evidence(self) -> None:
        prediction = np.zeros((2, 2), dtype=bool)
        residual_truth = np.zeros((2, 2), dtype=bool)
        residual_truth[0, 0] = True
        categories, evidence = _classify_frame(
            frame={"view_row_id": "v-1"},
            manifest={"scene_bucket": "parallel_boundary_pedestrian"},
            components=[
                {
                    "false_activation": True,
                    "mechanism_tags": [
                        "TEMPORAL_FLICKER",
                        "SMALL_FRAGMENT_NOISE",
                    ],
                    # NOT_EVALUABLE must not be promoted to shadow/texture.
                    "texture_or_shadow_confusion_status": "NOT_EVALUABLE_NO_APPEARANCE_LABEL",
                }
            ],
            event={
                "late_alert": True,
                "clearance_status": "unable_to_clear",
            },
            prediction=prediction,
            residual_truth=residual_truth,
        )
        self.assertEqual(
            categories,
            [
                "miss",
                "false_positive",
                "late_alert",
                "uncleared_alert",
                "small_fragment",
                "temporal_flicker",
                "parallel_curb",
                "pedestrian_crossing",
            ],
        )
        self.assertNotIn("shadow_texture", categories)
        self.assertIn("miss", evidence)
        self.assertIn("temporal_flicker", evidence)

    def test_custom_category_config_adds_category_without_classifier_edit(self) -> None:
        config = {
            "schema_version": "test.category_rules.v1",
            "default_category": "other",
            "categories": [
                {
                    "slug": "custom_marker",
                    "label": "自定义标记",
                    "rules": [
                        {
                            "type": "event_token",
                            "tokens": ["custom_marker"],
                            "evidence": "custom event marker",
                        }
                    ],
                },
                {"slug": "other", "label": "其他", "rules": []},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom_rules.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded = _load_category_config(path)
        categories, evidence = _classify_frame(
            frame={"view_row_id": "v-custom"},
            manifest={},
            components=[],
            event={"reason": "custom_marker"},
            prediction=np.zeros((2, 2), dtype=bool),
            residual_truth=np.zeros((2, 2), dtype=bool),
            category_rules=loaded["categories"],
            category_specs=_category_specs(loaded),
        )
        self.assertEqual(categories, ["custom_marker"])
        self.assertEqual(evidence["custom_marker"], "custom event marker")

    def test_event_phase_is_explicitly_not_evaluable_without_ledger(self) -> None:
        value, source = _event_phase(None, "MIDDLE")
        self.assertEqual(value, "NOT_EVALUABLE_EVENT_LEDGER_ABSENT")
        self.assertEqual(source, "derived_sequence_position=MIDDLE")

    def test_missing_panel_is_visual_and_readable(self) -> None:
        panel = _missing_panel((160, 100), "risk heatmap\nNOT_AVAILABLE")
        self.assertEqual(panel.size, (160, 100))
        self.assertEqual(panel.mode, "RGB")
        self.assertNotEqual(panel.getpixel((80, 50)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
