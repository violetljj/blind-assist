from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_future_servo_cohort import servo_phases


def episode(depth=1.2):
    return {
        "start_frame_id": 10,
        "start_depth_m": 4.0,
        "target_bbox_xyxy": [10, 10, 20, 30],
        "demonstrated_action": "TURN_LEFT",
        "future_min_depth_m": depth,
        "closest_frame_id": 30,
        "closest_target_bbox_xyxy": [5, 5, 40, 50],
    }


def test_servo_phases_materialize_far_and_bounded_near_stop():
    phases = servo_phases([episode()])
    assert [row["phase"] for row in phases] == ["FAR_GUIDANCE", "NEAR_STOP"]
    assert phases[1]["targets"][0]["desired_action"] == "STOP"


def test_servo_phases_omit_stop_when_route_never_reaches_stop_depth():
    assert [row["phase"] for row in servo_phases([episode(1.8)])] == ["FAR_GUIDANCE"]
