import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_video_multiexpert_risk_profile_prototype as prototype


class MultiExpertRiskProfilePrototypeTest(unittest.TestCase):
    def test_chromatic_channel_can_cover_dino_open_failure(self):
        dino = {"evaluation": {"open_projection": -0.1, "close_projection": 0.2, "close_ordered": True}, "prospective_pair_gate": {"passed": False}}
        chromatic = {"acceptance": {"passed": True}, "lifecycle": {"terminal_state": "clear", "open_event": None}, "diagnostics": {"risk_present_window": {"active_fraction": 0.8}, "stable_clear_window": {"active_fraction": 0.0}}}
        result = prototype.route_profile(dino, chromatic)
        self.assertEqual(["chromatic_construction_marker"], result["selected_positive_channels"])
        self.assertTrue(result["event_present"])
        self.assertTrue(result["event_closed"])

    def test_open_channel_without_close_stays_present(self):
        dino = {"evaluation": {"open_projection": 0.1, "close_projection": -0.2, "close_ordered": False}, "prospective_pair_gate": {"passed": True}}
        chromatic = {"acceptance": {"passed": False}, "lifecycle": {"terminal_state": "clear", "open_event": None}}
        result = prototype.route_profile(dino, chromatic)
        self.assertTrue(result["event_present"])
        self.assertFalse(result["event_closed"])
        self.assertEqual("present", result["terminal_state"])

    def test_absence_never_claims_clear(self):
        result = prototype.route_profile({"evaluation": {}, "prospective_pair_gate": {"passed": False}}, {"acceptance": {"passed": False}, "lifecycle": {}})
        self.assertFalse(result["event_present"])
        self.assertEqual("uncertain", result["terminal_state"])


if __name__ == "__main__":
    unittest.main()
