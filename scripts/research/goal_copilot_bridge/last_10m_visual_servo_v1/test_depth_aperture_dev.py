import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.depth_aperture_dev import select_aperture


def test_depth_aperture_rejects_flat_panel_and_accepts_straddling_opening() -> None:
    candidate = {"bbox_xyxy": [0, 0, 4, 4], "proposal_score": 0.5, "provider_rank": 1}
    dino = [{"bbox_xyxy": [0, 0, 4, 4], "score": 0.9}]
    flat = np.full((4, 4), 1.0, dtype=np.float32)
    opening = np.array([[1.0] * 4] * 3 + [[4.0] * 4], dtype=np.float32)
    assert select_aperture([candidate], dino, flat, 4, 4) is None
    assert select_aperture([candidate], dino, opening, 4, 4) is None  # median remains near
    opening[1:, :] = 4.0
    assert select_aperture([candidate], dino, opening, 4, 4) is not None
