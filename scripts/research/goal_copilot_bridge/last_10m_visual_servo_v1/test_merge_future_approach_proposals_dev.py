from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.merge_future_approach_proposals_dev import round_robin_merge


def row(box, rank):
    return {"bbox_xyxy": box, "source_rank": rank, "source_score": 1 / rank}


def test_round_robin_preserves_provider_diversity_and_deduplicates():
    merged = round_robin_merge([
        ("a", [row([0, 0, 10, 10], 1), row([20, 0, 30, 10], 2)]),
        ("b", [row([0, 0, 10, 10], 1), row([40, 0, 50, 10], 2)]),
    ])
    assert [item["source_provider"] for item in merged] == ["a", "a", "b"]
    assert [item["provider_rank"] for item in merged] == [1, 2, 3]
