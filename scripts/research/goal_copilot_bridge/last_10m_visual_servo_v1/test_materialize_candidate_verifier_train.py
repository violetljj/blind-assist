from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_candidate_verifier_train import expanded_crop


def test_expanded_crop_adds_context_and_clamps() -> None:
    image = Image.new("RGB", (100, 80))
    assert expanded_crop(image, [10, 10, 50, 50]).size == (52, 52)
    assert expanded_crop(image, [0, 0, 20, 20]).size == (23, 23)
