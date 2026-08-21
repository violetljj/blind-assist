from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.responsive_replay_runner import (
    _action_for_event,
    _build_edges,
    _choose_starts,
    _expand_viewport_graph,
    _pose_arrival,
    _signed_angle,
)


def _node(frame_id: str, x: float, heading: float, distance: float, error: float, timestamp: int):
    # At this latitude, a longitude delta of about 1.43e-5 is one metre.
    return {
        "frame_id": frame_id,
        "coordinates": [3.0 + x * 1.43e-5, 51.0],
        "heading_deg": heading,
        "target_distance_m": distance,
        "target_bearing_error_deg": error,
        "captured_at_ms": timestamp,
    }


class ResponsiveReplayTest(unittest.TestCase):
    def test_signed_angle_wraps(self) -> None:
        self.assertEqual(-20.0, _signed_angle(350.0, 10.0))
        self.assertEqual(20.0, _signed_angle(10.0, 350.0))

    def test_pose_arrival_is_fail_closed(self) -> None:
        self.assertTrue(_pose_arrival(_node("near", 0, 0, 5.9, 29.9, 0)))
        self.assertFalse(_pose_arrival(_node("far", 0, 0, 8.1, 0.0, 0)))
        self.assertFalse(_pose_arrival(_node("turned", 0, 0, 5.0, 31.0, 0)))

    def test_graph_edges_are_geometry_conditioned(self) -> None:
        nodes = [
            _node("base", 0.0, 90.0, 20.0, 0.0, 0),
            _node("left", 0.1, 70.0, 20.0, -20.0, 1),
            _node("right", -0.1, 110.0, 20.0, 20.0, 2),
            _node("forward", 1.8, 90.0, 18.2, 0.0, 3),
            _node("hold", 0.2, 91.0, 19.9, 1.0, 4),
        ]
        edges = _build_edges(nodes)["base"]
        self.assertEqual("left", edges["TURN_LEFT"])
        self.assertEqual("right", edges["TURN_RIGHT"])
        self.assertEqual("forward", edges["FORWARD"])
        self.assertEqual("hold", edges["RESCAN_HOLD"])

    def test_action_comes_only_from_current_event(self) -> None:
        common = {"to_state": "ADVANCE_AND_REOBSERVE"}
        self.assertEqual("TURN_LEFT", _action_for_event(common | {"candidate": {"center_x": 0.2}}))
        self.assertEqual("TURN_RIGHT", _action_for_event(common | {"candidate": {"center_x": 0.8}}))
        self.assertEqual("FORWARD", _action_for_event(common | {"candidate": {"center_x": 0.5}}))
        self.assertEqual("RESCAN_HOLD", _action_for_event({"to_state": "ARRIVAL_CONFIRM"}))
        self.assertIsNone(_action_for_event({"to_state": "COMPLETE"}))

    def test_viewport_turns_do_not_change_real_capture(self) -> None:
        base = _node("base", 0.0, 90.0, 20.0, 24.0, 0)
        expanded = _expand_viewport_graph([base], {"base": {}})
        center = next(item for item in expanded if item["viewport_yaw_index"] == 0)
        right = next(item for item in expanded if item["viewport_yaw_index"] == 1)
        self.assertEqual(right["frame_id"], center["actions"]["TURN_RIGHT"])
        self.assertEqual(center["coordinates"], right["coordinates"])
        self.assertEqual(12.0, right["viewport_yaw_offset_deg"])
        self.assertEqual(12.0, right["target_bearing_error_deg"])

    def test_start_selection_requires_reachable_arrival(self) -> None:
        nodes = [_node(f"n{i}", float(i), 0.0, 30.0 - i * 3.5, 0.0, i) for i in range(8)]
        nodes.append(_node("near2", 8.0, 0.0, 4.5, 0.0, 8))
        edges = {str(node["frame_id"]): {} for node in nodes}
        for first, second in zip(nodes, nodes[1:]):
            edges[str(first["frame_id"])]["FORWARD"] = str(second["frame_id"])
        starts = _choose_starts(nodes, edges)
        self.assertGreaterEqual(len(starts), 5)
        self.assertTrue(all(start.startswith("n") for start in starts))


if __name__ == "__main__":
    unittest.main()
