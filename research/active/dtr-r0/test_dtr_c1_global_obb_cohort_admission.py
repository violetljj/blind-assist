from __future__ import annotations

import unittest

from dtr_c1_global_obb_cohort_admission import (
    CLEAR,
    CONTACT,
    PROXIMITY,
    STATUS_ADMITTED_MINIMUM,
    STATUS_ADMITTED_PREFERRED,
    STATUS_INSUFFICIENT,
    UNKNOWN,
    NativeBox,
    bounded_contact_events,
    global_truth_timeline,
    select_roster,
)


class DTRC1GlobalOBBCohortAdmissionTest(unittest.TestCase):
    def test_truth_contract_keeps_circle_only_secondary(self) -> None:
        frames = list(range(6))
        timestamps = {frame: float(frame) for frame in frames}
        circle_only = NativeBox("p:1", 1.10, 0.0, 1.0, 0.2, 1.57079632679)
        timeline = global_truth_timeline(
            frames=frames,
            timestamps=timestamps,
            boxes_by_frame={frame: [circle_only] for frame in frames},
            horizon_s=2.0,
            route_radius_m=0.65,
        )
        self.assertEqual(timeline[0]["label"], PROXIMITY)
        self.assertEqual(timeline[-1]["label"], PROXIMITY)
        self.assertEqual(timeline[0]["responsible_components"], ["p:1"])

        contact = NativeBox("p:2", 0.70, 0.0, 0.4, 0.4, 0.0)
        contact_timeline = global_truth_timeline(
            frames=frames,
            timestamps=timestamps,
            boxes_by_frame={frame: [contact] for frame in frames},
            horizon_s=2.0,
            route_radius_m=0.65,
        )
        self.assertEqual(contact_timeline[0]["label"], CONTACT)

    def test_clear_and_unknown_require_full_future(self) -> None:
        frames = list(range(5))
        timestamps = {frame: float(frame) for frame in frames}
        timeline = global_truth_timeline(
            frames=frames,
            timestamps=timestamps,
            boxes_by_frame={frame: [] for frame in frames},
            horizon_s=2.0,
        )
        self.assertEqual([row["label"] for row in timeline], [CLEAR, CLEAR, CLEAR, UNKNOWN, UNKNOWN])

    def test_bounded_event_requires_known_non_contact_on_both_sides(self) -> None:
        def row(frame: int, label: str, components: list[str] | None = None) -> dict:
            return {
                "frame": frame,
                "time_s": frame / 10.0,
                "label": label,
                "responsible_components": components or [],
                "first_hit_delta_s": 1.0 if label == CONTACT else None,
            }

        bounded = [
            row(0, CLEAR),
            row(1, CONTACT, ["p:1"]),
            row(2, CONTACT, ["p:1"]),
            row(3, PROXIMITY),
        ]
        events = bounded_contact_events(bounded)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["unique_responsible_component"])

        right_censored = bounded[:-1] + [row(3, UNKNOWN)]
        self.assertEqual(bounded_contact_events(right_censored), [])
        self.assertEqual(bounded_contact_events(bounded[1:]), [])

    def test_roster_uses_lexicographic_shortest_prefix(self) -> None:
        def summary(name: str, events: int, unique: int, non_contact: float) -> dict:
            return {
                "sequence": name,
                "frames": 100,
                "timeline_duration_s": 10.0,
                "bounded_contact_events": events,
                "unique_responsible_events": unique,
                "known_non_contact_s": non_contact,
                "truth_duration_s": {
                    CONTACT: 10.0,
                    PROXIMITY: 0.0,
                    CLEAR: non_contact,
                    UNKNOWN: 0.0,
                },
            }

        rows = [
            summary("b", 8, 4, 50.0),
            summary("a", 6, 3, 50.0),
            summary("c", 7, 4, 30.0),
        ]
        status, selected = select_roster(rows)
        self.assertEqual(status, STATUS_ADMITTED_PREFERRED)
        self.assertEqual([row["sequence"] for row in selected], ["a", "b", "c"])

        status, selected = select_roster(rows[:2])
        self.assertEqual(status, STATUS_ADMITTED_MINIMUM)
        self.assertEqual([row["sequence"] for row in selected], ["a", "b"])

        status, selected = select_roster([summary("a", 1, 1, 100.0)])
        self.assertEqual(status, STATUS_INSUFFICIENT)
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
