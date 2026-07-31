from __future__ import annotations

import unittest

from .fixed_clip_units import (
    MIXED_OBSERVATION,
    NO_EVIDENCE,
    NOT_EVALUABLE,
    PRESENT,
    STABLE_NO_EVIDENCE,
    STABLE_PRESENT,
    FixedClipUnitError,
    compare_observation_reviews,
    derive_unit_state,
)


def manifest_rows() -> list[dict[str, object]]:
    rows = []
    for unit_index in range(2):
        unit_id = f"unit-{unit_index}"
        for slot in range(4):
            rows.append(
                {
                    "unit_id": unit_id,
                    "clip_id": unit_id,
                    "source_id": f"source-{unit_index}",
                    "session_id": f"session-{unit_index}",
                    "slot_ordinal": slot,
                    "source_frame_index": unit_index * 10 + slot,
                    "claim_critical": True,
                }
            )
    return rows


def review_rows(labels: list[list[str]]) -> list[dict[str, object]]:
    rows = []
    for unit_index, unit_labels in enumerate(labels):
        for slot, label in enumerate(unit_labels):
            rows.append(
                {
                    "unit_id": f"unit-{unit_index}",
                    "slot_ordinal": slot,
                    "label": label,
                }
            )
    return rows


class FixedClipUnitTests(unittest.TestCase):
    def test_frozen_mapping_is_conservative_and_threshold_free(self) -> None:
        self.assertEqual(STABLE_PRESENT, derive_unit_state([PRESENT, PRESENT]))
        self.assertEqual(STABLE_NO_EVIDENCE, derive_unit_state([NO_EVIDENCE, NO_EVIDENCE]))
        self.assertEqual(NOT_EVALUABLE, derive_unit_state([PRESENT, NOT_EVALUABLE]))
        self.assertEqual(MIXED_OBSERVATION, derive_unit_state([PRESENT, NO_EVIDENCE]))

    def test_empty_or_unknown_labels_fail_closed(self) -> None:
        with self.assertRaises(FixedClipUnitError):
            derive_unit_state([])
        with self.assertRaises(FixedClipUnitError):
            derive_unit_state(["UNKNOWN"])

    def test_fixed_unit_summary_does_not_move_when_labels_change(self) -> None:
        manifest = manifest_rows()
        primary = review_rows(
            [
                [PRESENT] * 4,
                [NO_EVIDENCE] * 4,
            ]
        )
        isolated = review_rows(
            [
                [PRESENT, PRESENT, NO_EVIDENCE, PRESENT],
                [NO_EVIDENCE] * 4,
            ]
        )
        result = compare_observation_reviews(
            {
                "protocol_id": "CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR",
                "evidence_instance": "CENTRAL_OBSTRUCTION_D0_A_SUCCESSOR_FIXED_CLIP_CALIBRATION_R0",
                "observations": manifest,
            },
            {
                "review_id": "primary",
                "observations": primary,
            },
            {
                "review_id": "isolated",
                "observations": isolated,
            },
            gates={
                "minimum_fixed_units": 2,
                "minimum_source_count": 2,
                "overall_observation_label_agreement": 0.8,
                "claim_critical_observation_label_agreement": 0.8,
                "unresolved_fraction": 0.1,
                "union_not_evaluable_fraction": 0.4,
                "fixed_unit_state_match_rate": 0.5,
            },
        )
        self.assertFalse(result["natural_event_grouping_used"])
        self.assertEqual(1.0, result["metrics"]["fixed_unit_boundary_reproducibility"])
        self.assertEqual(STABLE_PRESENT, result["primary_fixed_units"][0]["unit_state"])
        self.assertEqual(0.875, result["metrics"]["overall_observation_label_agreement"])

    def test_duplicate_review_slot_rejected(self) -> None:
        manifest = manifest_rows()
        rows = review_rows([[PRESENT] * 4, [NO_EVIDENCE] * 4])
        rows.append(rows[-1].copy())
        with self.assertRaises(FixedClipUnitError):
            compare_observation_reviews(
                {"observations": manifest},
                {"observations": rows},
                {"observations": review_rows([[PRESENT] * 4, [NO_EVIDENCE] * 4])},
                gates={},
            )


if __name__ == "__main__":
    unittest.main()
