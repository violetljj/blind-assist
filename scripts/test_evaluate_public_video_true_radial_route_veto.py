import unittest

import evaluate_public_video_true_radial_route_veto as subject


class TrueRadialRouteVetoChecksTest(unittest.TestCase):
    def review(self):
        return {
            "obstacle_remains_safely_lateral_to_ego_route": True,
            "route_relation_should_veto_event_entry": True,
            "hard_cut_or_montage_present": False,
        }

    def test_accepts_safe_lateral_radial_event_when_route_delta_is_nonpositive(self):
        checks = subject.veto_checks(
            frozen_event={"radial_approach_passed": True}, negative=self.review(), route_delta=-0.01
        )
        self.assertTrue(all(checks.values()))

    def test_rejects_route_intrusion_even_when_visual_review_expected_veto(self):
        checks = subject.veto_checks(
            frozen_event={"radial_approach_passed": True}, negative=self.review(), route_delta=0.01
        )
        self.assertFalse(checks["frozen_route_relation_vetoes_entry"])

    def test_rejects_nonradial_or_unsafe_review(self):
        review = self.review()
        review["obstacle_remains_safely_lateral_to_ego_route"] = False
        checks = subject.veto_checks(
            frozen_event={"radial_approach_passed": False}, negative=review, route_delta=-0.01
        )
        self.assertFalse(checks["frozen_radial_entry_present"])
        self.assertFalse(checks["visual_review_confirms_safe_lateral"])


if __name__ == "__main__":
    unittest.main()
