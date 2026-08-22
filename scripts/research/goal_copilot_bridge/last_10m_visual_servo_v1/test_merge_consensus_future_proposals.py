from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.merge_consensus_future_proposals import consensus_merge


def row(box, rank=1):
    return {"bbox_xyxy": box, "source_score": 0.5, "source_rank": rank}


def test_cross_provider_consensus_ranks_supported_region_first():
    merged = consensus_merge([
        ("a", [row([0, 0, 10, 10]), row([50, 0, 80, 40], 2)]),
        ("b", [row([52, 0, 82, 40])]),
        ("c", [row([51, 1, 81, 41])]),
    ])
    assert merged[0]["provider_support"] == 3
    assert merged[0]["supporting_providers"] == ["a", "b", "c"]
