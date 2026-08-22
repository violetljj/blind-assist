from pathlib import Path

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.train_functional_door_detector import split_name


def test_split_is_deterministic_and_path_sensitive() -> None:
    path = "images/example.jpg"
    assert split_name(path) == split_name(path)
    assert split_name(path) in {"train", "val"}
    observed = {split_name(f"images/{index:04d}.jpg") for index in range(50)}
    assert observed == {"train", "val"}


def test_training_script_does_not_reference_s2_data() -> None:
    source = Path(__file__).with_name("train_functional_door_detector.py").read_text(encoding="utf-8")
    assert "tartanair-s2" not in source.lower()
    assert "evaluation.json" not in source.lower()
