from __future__ import annotations

import unittest

from scripts.research.egomotion_compensated_looming.rcle_low_reference_false_trigger_r1.attribution import (
    classify_trigger,
)


def old(trigger: bool = True) -> dict:
    return {"trigger": trigger}


def baseline(*, evaluable: bool = True, trigger: bool = True, raw: float = 0.02) -> dict:
    return {
        "evaluable": evaluable,
        "trigger": trigger,
        "raw_expansion_median_per_s": raw,
    }


def geometry(value: float = 0.0) -> dict:
    return {"median_signed_radial_expansion_per_s": value}


class AttributionClassificationTest(unittest.TestCase):
    def test_nontrigger_is_not_attributed(self) -> None:
        self.assertEqual(
            classify_trigger(old(False), baseline(), geometry()),
            "OLD_NOT_TRIGGERED",
        )

    def test_geometry_consistent_precedes_mechanism_labels(self) -> None:
        self.assertEqual(
            classify_trigger(old(), baseline(trigger=False), geometry(0.01)),
            "GEOMETRY_AT_OR_ABOVE_THRESHOLD",
        )

    def test_support_manager_enabled_evaluability(self) -> None:
        self.assertEqual(
            classify_trigger(old(), baseline(evaluable=False), geometry()),
            "SUPPORT_MANAGER_ENABLED_EVALUABILITY",
        )

    def test_support_manager_induced_trigger(self) -> None:
        self.assertEqual(
            classify_trigger(old(), baseline(trigger=False), geometry()),
            "SUPPORT_MANAGER_INDUCED_TRIGGER",
        )

    def test_rotation_compensation_crossing(self) -> None:
        self.assertEqual(
            classify_trigger(old(), baseline(raw=0.01), geometry()),
            "ROTATION_COMPENSATION_THRESHOLD_CROSSING",
        )

    def test_baseline_local_flow_crossing(self) -> None:
        self.assertEqual(
            classify_trigger(old(), baseline(raw=0.02), geometry()),
            "BASELINE_LOCAL_FLOW_THRESHOLD_CROSSING",
        )


if __name__ == "__main__":
    unittest.main()
