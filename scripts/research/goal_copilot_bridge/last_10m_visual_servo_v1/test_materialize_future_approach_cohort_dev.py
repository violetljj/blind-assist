from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_future_approach_cohort_dev import cluster_events


def episode(trajectory, frame):
    return {"trajectory": trajectory, "start_frame_id": frame}


def test_cluster_events_separates_trajectory_and_horizon():
    clusters = cluster_events([episode("P000", 10), episode("P000", 20), episode("P000", 60), episode("P001", 15)])
    assert [[row["start_frame_id"] for row in cluster] for cluster in clusters] == [[10, 20], [60], [15]]
