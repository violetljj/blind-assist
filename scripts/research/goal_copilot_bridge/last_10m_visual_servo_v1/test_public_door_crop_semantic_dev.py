from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.public_door_crop_semantic_dev import crop_box


def test_crop_box_clamps_to_image() -> None:
    image = Image.new("RGB", (10, 8))
    assert crop_box(image, [-2, 1, 12, 7]).size == (10, 6)
