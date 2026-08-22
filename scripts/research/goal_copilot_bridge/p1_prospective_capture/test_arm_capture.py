from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import content_sha256
from scripts.research.goal_copilot_bridge.p1_prospective_capture.arm_capture import arm_capture
from scripts.research.goal_copilot_bridge.p1_prospective_capture.materialize_capture import ProspectiveCaptureError


def c0_receipt(count: int = 5) -> dict:
    base = datetime(2026, 8, 22, 7, 0, tzinfo=timezone.utc)
    value = {
        "schema_version": "blindassist_p1_pa3_c0_public_goal_cohort_v1",
        "private_truth_access": False,
        "pa3_inference_authorized": False,
        "episodes": [{"episode_id": f"capture-{index:02d}", "goal_provenance": {"goal_recorded_at_utc": (base + timedelta(seconds=index)).isoformat()}} for index in range(count)],
    }
    value["receipt_body_sha256"] = content_sha256(value)
    return value


class ArmCaptureTest(unittest.TestCase):
    def test_arms_complete_roster_before_capture(self) -> None:
        plan = arm_capture(c0_receipt(), "2026-08-22T07:01:00+00:00")
        self.assertEqual(5, plan["episode_count"])
        self.assertEqual("capture-00.mp4", plan["episodes"][0]["media_relative_path"])
        self.assertEqual("NOT_STARTED", plan["capture_state_at_arming"])
        self.assertEqual(content_sha256({key: value for key, value in plan.items() if key != "capture_plan_body_sha256"}), plan["capture_plan_body_sha256"])

    def test_rejects_underpowered_roster(self) -> None:
        with self.assertRaisesRegex(ProspectiveCaptureError, "at least 5"):
            arm_capture(c0_receipt(4), "2026-08-22T07:01:00+00:00")

    def test_rejects_arming_before_goal(self) -> None:
        with self.assertRaisesRegex(ProspectiveCaptureError, "goal must precede"):
            arm_capture(c0_receipt(), "2026-08-22T06:59:00+00:00")

    def test_rejects_unsafe_episode_path_identity(self) -> None:
        c0 = c0_receipt()
        c0.pop("receipt_body_sha256")
        c0["episodes"][0]["episode_id"] = "../escape"
        c0["receipt_body_sha256"] = content_sha256(c0)
        with self.assertRaisesRegex(ProspectiveCaptureError, "unsafe episode_id"):
            arm_capture(c0, "2026-08-22T07:01:00+00:00")


if __name__ == "__main__":
    unittest.main()

