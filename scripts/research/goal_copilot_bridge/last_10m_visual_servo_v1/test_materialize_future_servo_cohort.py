import io
import zipfile

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_future_servo_cohort import route_plan, servo_phases


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


def test_route_plan_is_derived_from_pose_without_target_truth(tmp_path):
    path = tmp_path / "imu.zip"
    positions = np.zeros((401, 3), dtype=np.float64)
    positions[300, :2] = [3.0, 1.0]
    orientations = np.zeros((401, 3), dtype=np.float64)
    with zipfile.ZipFile(path, "w") as archive:
        for name, values in (("pos_global.npy", positions), ("ori_global.npy", orientations)):
            stream = io.BytesIO()
            np.save(stream, values)
            archive.writestr(f"E/Data_easy/P000/imu/{name}", stream.getvalue())
    with zipfile.ZipFile(path) as archive:
        plan = route_plan(archive, "E", "P000", 0)
    assert plan["bearing_fraction"] > 0.5
    assert plan["derived_without_semantic_or_target_truth"] is True
