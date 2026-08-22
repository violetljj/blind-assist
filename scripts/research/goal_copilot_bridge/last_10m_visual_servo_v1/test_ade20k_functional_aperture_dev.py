import numpy as np
import pytest

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.ade20k_functional_aperture_dev import region_depth_percentile, semantic_evidence


def test_region_depth_percentile_ignores_invalid_values() -> None:
    depth = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)
    assert region_depth_percentile(depth, [0, 0, 2, 2], 2, 2) == pytest.approx(1.2)


def test_semantic_evidence_requires_door_pixels_to_outvote_furniture() -> None:
    names = {10: "cabinet", 14: "door"}
    accepted = semantic_evidence(np.array([[14, 14], [14, 10]]), [0, 0, 2, 2], 2, 2, names)
    rejected = semantic_evidence(np.array([[14, 10], [10, 10]]), [0, 0, 2, 2], 2, 2, names)
    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
