from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dtr_m0_r7_error_attribution import (
    CAUSE_ATTRIBUTION,
    CAUSE_EXTRAPOLATION,
    CAUSE_INHERITED,
    CAUSE_NONCRITICAL,
    CAUSE_STATIC,
    CAUSE_UNKNOWN,
    PROVENANCE_EXTENDED,
    PROVENANCE_INHERITED,
    PROVENANCE_MERGED,
    PROVENANCE_NEW,
    classify_primary_cause,
    classify_provenance,
    render_timeline_svg,
)
from jrdb_native_ceiling import AlertSegment


class DTRM0R7ErrorAttributionTest(unittest.TestCase):
    def test_provenance_separates_inherited_new_extended_and_merged(self) -> None:
        segment = AlertSegment(10, 20)
        self.assertEqual(
            PROVENANCE_INHERITED,
            classify_provenance(segment, [AlertSegment(12, 18)], []),
        )
        self.assertEqual(
            PROVENANCE_NEW,
            classify_provenance(segment, [], [10]),
        )
        self.assertEqual(
            PROVENANCE_EXTENDED,
            classify_provenance(segment, [AlertSegment(12, 18)], [10]),
        )
        self.assertEqual(
            PROVENANCE_MERGED,
            classify_provenance(
                segment,
                [AlertSegment(10, 12), AlertSegment(18, 20)],
                [15],
            ),
        )

    def test_primary_cause_uses_existing_motion_threshold(self) -> None:
        self.assertEqual(
            CAUSE_INHERITED,
            classify_primary_cause(PROVENANCE_INHERITED, None, 0.0, None, None),
        )
        self.assertEqual(
            CAUSE_UNKNOWN,
            classify_primary_cause(PROVENANCE_NEW, None, 0.5, None, 0.1),
        )
        self.assertEqual(
            CAUSE_UNKNOWN,
            classify_primary_cause(PROVENANCE_NEW, False, None, None, None),
        )
        self.assertEqual(
            CAUSE_STATIC,
            classify_primary_cause(PROVENANCE_NEW, False, 0.249, None, 2.0),
        )
        self.assertEqual(
            CAUSE_NONCRITICAL,
            classify_primary_cause(PROVENANCE_NEW, False, 0.25, None, 0.20),
        )
        self.assertEqual(
            CAUSE_ATTRIBUTION,
            classify_primary_cause(PROVENANCE_NEW, False, 0.25, None, 0.30),
        )
        self.assertEqual(
            CAUSE_EXTRAPOLATION,
            classify_primary_cause(PROVENANCE_NEW, False, 0.25, 1.0, 2.0),
        )

    def test_timeline_is_a_self_contained_svg(self) -> None:
        rows = [
            {
                "first_frame": 120,
                "last_frame": 125,
                "diagnostic_frame": 121,
                "provenance": PROVENANCE_NEW,
                "primary_cause": CAUSE_STATIC,
            }
        ]
        values = [0] * 143
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.svg"
            render_timeline_svg(
                path,
                global_risky_cells=values,
                truth_counts=values,
                baseline_counts=values,
                flow_counts=values,
                rows=rows,
            )
            payload = path.read_text(encoding="utf-8")
        self.assertIn("<svg", payload)
        self.assertIn(CAUSE_STATIC, payload)
        self.assertIn("R7-P false-segment attribution", payload)


if __name__ == "__main__":
    unittest.main()
