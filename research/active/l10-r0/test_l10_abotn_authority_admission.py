from __future__ import annotations

import unittest

from l10_abotn_authority_admission import REQUIRED_IDENTIFIERS, REQUIRED_TRUE, evaluate, template_for
from test_l10_abotn_cohort_freeze import episode
from l10_abotn_cohort_freeze import freeze_cohort


class AuthorityAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for scene in ("a", "b", "c"):
            rows.extend([episode(scene, 0, 0, f"{scene}-a"), episode(scene, 1, 1, f"{scene}-b")])
        self.cohort = freeze_cohort(rows, "manifest")
        self.sha = "a" * 64

    def test_unknown_template_fails_closed(self) -> None:
        result = evaluate(self.cohort, template_for(self.cohort, self.sha), self.sha)
        self.assertFalse(result["admitted"])
        self.assertTrue(all(scene["disposition"] == "NOT_EVALUABLE" for scene in result["scenes"]))

    def test_complete_exact_roster_can_be_admitted_but_user_completion_is_not_claimed(self) -> None:
        addendum = template_for(self.cohort, self.sha)
        for scene in addendum["scenes"]:
            for index, target in enumerate(scene["frozen_targets"]):
                for field in REQUIRED_IDENTIFIERS:
                    target[field] = f"{field}-{index}"
                target["frame_sha256"] = "b" * 64
                for field in REQUIRED_TRUE:
                    target[field] = "TRUE"
            scene["target_absent_control"]["target_present_truth"] = "FALSE"
            scene["target_absent_control"]["adjudication_source"] = "pixel-blind-review"
            scene["scene_disposition"] = "ADMITTED"
        result = evaluate(self.cohort, addendum, self.sha)
        self.assertTrue(result["admitted"])
        self.assertNotIn("user completion", result["verdict"].lower())

    def test_cannot_replace_frozen_target_after_pixels(self) -> None:
        addendum = template_for(self.cohort, self.sha)
        addendum["scenes"][0]["frozen_targets"][0]["annotation_path"] = "replacement.json"
        result = evaluate(self.cohort, addendum, self.sha)
        self.assertFalse(result["admitted"])
        self.assertIn("frozen_target_roster_or_order_mismatch", result["scenes"][0]["issues"])


if __name__ == "__main__":
    unittest.main()
