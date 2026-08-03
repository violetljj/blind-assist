import unittest

from render_motion_occupancy_a0_demo import select_middle_frames


def row(sequence, frame, timestamp, band):
    return {
        "sequence_id": sequence,
        "frame_path": frame,
        "timestamp": timestamp,
        "band": band,
        "horizon_m": 1.5,
        "label_occupied": False,
        "scores": {"motion_probability_field": 0.1},
    }


class RenderMotionOccupancyA0DemoTest(unittest.TestCase):
    def test_selects_middle_complete_frame(self) -> None:
        details = []
        for timestamp in (1.0, 2.0, 3.0):
            for band in ("left", "center", "right"):
                details.append(row("s0", f"f{timestamp}", timestamp, band))
        selected = select_middle_frames(details)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["frame_path"], "f2.0")


if __name__ == "__main__":
    unittest.main()
