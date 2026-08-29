from __future__ import annotations

import unittest

from dtr_x0_motion_source_attribution import (
    BAD_FLOW,
    FRAGMENTATION,
    REAL_MOVER_NONCRITICAL,
    ROUTE_GEOMETRY_MISS,
    STATIC_PSEUDO_MOTION,
    WRONG_COMPONENT_BINDING,
    c31_incremental_ranges,
    choose_primary_cause,
    choose_route,
)


class DTRX0MotionSourceAttributionTest(unittest.TestCase):
    def test_incremental_ranges_require_no_pdc_overlap(self) -> None:
        pdc = [{"first_frame": 10, "last_frame": 20}]
        c31 = [
            {"first_frame": 8, "last_frame": 12},
            {"first_frame": 25, "last_frame": 30},
        ]
        self.assertEqual([c31[1]], c31_incremental_ranges(pdc, c31))

    def test_structural_cause_precedes_cell_plurality(self) -> None:
        counts = {BAD_FLOW: 5, REAL_MOVER_NONCRITICAL: 2}
        self.assertEqual(
            WRONG_COMPONENT_BINDING,
            choose_primary_cause(counts, wrong_binding=True),
        )
        self.assertEqual(
            FRAGMENTATION,
            choose_primary_cause(counts, fragmentation=True),
        )
        self.assertEqual(BAD_FLOW, choose_primary_cause(counts))

    def test_route_selection_is_source_first_then_geometry_then_authority(self) -> None:
        self.assertEqual(
            "STRONGER_SCENE_FLOW_SOURCE",
            choose_route({BAD_FLOW: 1}, {REAL_MOVER_NONCRITICAL: 9})[0],
        )
        self.assertEqual(
            "STRONGER_SCENE_FLOW_SOURCE",
            choose_route({}, {STATIC_PSEUDO_MOTION: 5, REAL_MOVER_NONCRITICAL: 4})[0],
        )
        self.assertEqual(
            "CONTINUOUS_COLLISION_GEOMETRY",
            choose_route({ROUTE_GEOMETRY_MISS: 1}, {REAL_MOVER_NONCRITICAL: 4})[0],
        )
        self.assertEqual(
            "LEARNED_MOTION_AUTHORITY",
            choose_route({FRAGMENTATION: 2}, {REAL_MOVER_NONCRITICAL: 4})[0],
        )


if __name__ == "__main__":
    unittest.main()
