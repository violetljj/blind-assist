import json
import zipfile

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.tartanground_future_door_approach import depth_member, frame_id, semantic_id


def test_tartanground_member_mapping():
    member = "seg_lcam_front/000123_lcam_front_seg.png"
    assert frame_id(member) == 123
    assert depth_member(member) == "depth_lcam_front/000123_lcam_front_depth.png"


def test_tartanground_semantic_id_is_exact(tmp_path):
    path = tmp_path / "labels.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("seg_label_map.json", json.dumps({"name_map": {"hangardoor": 70}}))
    assert semantic_id(path, "hangardoor") == 70
