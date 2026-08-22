from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.functional_door_supervised_dev import verify_candidate


CANDIDATE = {"bbox_xyxy": [0, 0, 100, 100], "proposal_score": 0.8}


def test_room_door_match_accepts_candidate() -> None:
    selected, evidence = verify_candidate(CANDIDATE, [{"bbox_xyxy": [5, 5, 95, 95], "class_id": 0, "confidence": 0.7}])
    assert selected is not None
    assert evidence["reason"] == "ROOM_DOOR_WINS"


def test_stronger_furniture_door_rejects_candidate() -> None:
    detections = [
        {"bbox_xyxy": [5, 5, 95, 95], "class_id": 0, "confidence": 0.6},
        {"bbox_xyxy": [5, 5, 95, 95], "class_id": 2, "confidence": 0.8},
    ]
    selected, evidence = verify_candidate(CANDIDATE, detections)
    assert selected is None
    assert evidence["reason"] == "FURNITURE_DOOR_WINS"


def test_unmatched_room_detection_rejects_candidate() -> None:
    selected, evidence = verify_candidate(CANDIDATE, [{"bbox_xyxy": [200, 200, 300, 300], "class_id": 0, "confidence": 0.9}])
    assert selected is None
    assert evidence["reason"] == "NO_ROOM_DOOR_MATCH"
