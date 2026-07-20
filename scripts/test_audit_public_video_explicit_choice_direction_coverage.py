import unittest

import audit_public_video_explicit_choice_direction_coverage as subject


class ExplicitChoiceDirectionCoverageTest(unittest.TestCase):
    def test_summarize_keeps_direction_and_class_separate(self) -> None:
        events = [
            {"explicit_choice": "LEFT", "reference_intervention_required": False},
            {"explicit_choice": "STRAIGHT", "reference_intervention_required": True},
            {"explicit_choice": "RIGHT", "reference_intervention_required": False},
        ]
        result = subject.summarize(events, ["LEFT", "STRAIGHT", "RIGHT"])
        self.assertEqual(0, result["LEFT"]["intervention_event_count"])
        self.assertEqual(1, result["STRAIGHT"]["intervention_event_count"])
        self.assertEqual(1, result["RIGHT"]["context_event_count"])


if __name__ == "__main__":
    unittest.main()
