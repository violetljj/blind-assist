import pytest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.train_public_door_semantic import validate_receipt


def test_validate_receipt_rejects_formal_environment() -> None:
    receipt = {"source_environments": ["House"], "excluded_formal_environments": ["House", "RetroOffice", "CountryHouse", "AmericanDiner"], "private_truth_access": False, "case_count": 1000, "val_count": 200}
    with pytest.raises(ValueError, match="formal environment"):
        validate_receipt(receipt)


def test_validate_receipt_accepts_public_development_sources() -> None:
    receipt = {"source_environments": ["Hospital"], "excluded_formal_environments": ["House", "RetroOffice", "CountryHouse", "AmericanDiner"], "private_truth_access": False, "case_count": 1000, "val_count": 200}
    validate_receipt(receipt)


def test_semantic_training_disables_incompatible_mosaic() -> None:
    source = __import__(
        "pathlib"
    ).Path(__file__).with_name("train_public_door_semantic.py").read_text(encoding="utf-8")
    assert "mosaic=0.0" in source
