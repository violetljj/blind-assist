from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_remote_s5 import modality_member


def test_modality_member_maps_segmentation_to_aligned_payload() -> None:
    source = "Office/Data_easy/P001/seg_lcam_front/000001_lcam_front_seg.png"
    assert modality_member(source, "image").endswith("/image_lcam_front/000001_lcam_front.png")
    assert modality_member(source, "depth").endswith("/depth_lcam_front/000001_lcam_front_depth.png")
