from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.future_door_approach_dev import approach_summary, match_target


def target(box, depth, pixels=1000):
    return {"bbox_xyxy": box, "depth_median_m": depth, "pixel_count": pixels, "image_width": 640, "image_height": 640}


def test_match_target_prefers_spatially_continuous_component():
    previous = target([100, 100, 200, 300], 4.0)
    good = target([105, 100, 210, 310], 3.8, 1100)
    distractor = target([400, 100, 500, 300], 3.8, 1100)
    assert match_target(previous, [distractor, good], 640, 640) is good


def test_approach_requires_depth_and_area_evidence():
    track = [target([250, 100, 390, 300], 4.0, 1000)]
    track.extend(target([245, 90, 395, 320], 4.0 - index * 0.3, 1000 + index * 80) for index in range(1, 9))
    summary = approach_summary(track)
    assert summary["approached"] is True
    assert summary["demonstrated_action"] == "ADVANCE"
    assert summary["closest_target_bbox_xyxy"] == track[-1]["bbox_xyxy"]


def test_lateral_target_maps_to_turn():
    track = [target([20, 100, 120, 300], 4.0, 1000)]
    track.extend(target([25, 90, 135, 320], 4.0 - index * 0.3, 1000 + index * 80) for index in range(1, 9))
    assert approach_summary(track)["demonstrated_action"] == "TURN_LEFT"
