from dataclasses import dataclass

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.select_tartanair_functional_environment import rank_environments


@dataclass
class File:
    path: str
    size: int


def test_rank_requires_both_modalities_and_excludes_consumed():
    files = [
        File("A/Data_easy/seg_lcam_front.zip", 3),
        File("A/Data_easy/depth_lcam_front.zip", 4),
        File("B/Data_easy/seg_lcam_front.zip", 1),
        File("C/Data_easy/seg_lcam_front.zip", 10),
        File("C/Data_easy/depth_lcam_front.zip", 10),
    ]
    assert rank_environments(files, {"A", "B", "C"}, {"C"}) == [
        {"environment": "A", "required_archive_bytes": 7, "modalities": {"Data_easy/seg_lcam_front.zip": 3, "Data_easy/depth_lcam_front.zip": 4}}
    ]
