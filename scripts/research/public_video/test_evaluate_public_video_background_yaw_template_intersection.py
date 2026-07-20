import unittest

import evaluate_public_video_background_yaw_template_intersection as subject


class BackgroundYawTemplateIntersectionTest(unittest.TestCase):
    def test_module_exposes_candidate_scorer(self) -> None:
        self.assertTrue(callable(subject.score_candidate))


if __name__ == "__main__":
    unittest.main()
