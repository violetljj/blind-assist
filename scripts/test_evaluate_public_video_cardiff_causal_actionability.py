import unittest

import evaluate_public_video_cardiff_causal_actionability as subject


class CardiffCausalActionabilityTest(unittest.TestCase):
    def test_selects_retained_real_coherent_candidate(self) -> None:
        review = {"reviews": [{
            "candidate_id": "kept",
            "decision": "retain_as_provisional_positive_event_role_candidate",
            "real_target_object": True,
            "same_object_track": True,
        }]}
        self.assertEqual("kept", subject.select_retained_review(review, "kept")["candidate_id"])

    def test_rejects_excluded_candidate(self) -> None:
        review = {"reviews": [{
            "candidate_id": "excluded",
            "decision": "exclude_from_route_role_training_and_evaluation",
            "real_target_object": True,
            "same_object_track": False,
        }]}
        with self.assertRaisesRegex(ValueError, "not retained"):
            subject.select_retained_review(review, "excluded")

    def test_rejects_detector_confusion(self) -> None:
        review = {"reviews": [{
            "candidate_id": "confused",
            "decision": "retain_as_provisional_positive_event_role_candidate",
            "real_target_object": False,
            "same_object_track": True,
        }]}
        with self.assertRaisesRegex(ValueError, "real coherent"):
            subject.select_retained_review(review, "confused")


if __name__ == "__main__":
    unittest.main()
