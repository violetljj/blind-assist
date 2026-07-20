import copy
import unittest

import run_public_silver_dual_evidence_lifecycle_fusion as fusion


class DualEvidenceLifecycleFusionTest(unittest.TestCase):
    def test_strong_increase_opens_from_trusted_clear(self) -> None:
        result = fusion.decide_transition(
            previous_state="clear", normalized_signed_change=0.6,
            semantic_exit=False, trusted_reference=True,
        )
        self.assertEqual(result["predicted_transition"], "open_event")

    def test_strong_decrease_closes_from_trusted_risk(self) -> None:
        result = fusion.decide_transition(
            previous_state="risk", normalized_signed_change=-0.6,
            semantic_exit=False, trusted_reference=True,
        )
        self.assertEqual(result["predicted_transition"], "close_event")
        self.assertEqual(result["reason"], "strong_relative_decrease")

    def test_weak_decrease_needs_semantic_exit(self) -> None:
        without_exit = fusion.decide_transition(
            previous_state="risk", normalized_signed_change=-0.04,
            semantic_exit=False, trusted_reference=True,
        )
        with_exit = fusion.decide_transition(
            previous_state="risk", normalized_signed_change=-0.04,
            semantic_exit=True, trusted_reference=True,
        )
        self.assertEqual(without_exit["predicted_transition"], "uncertain")
        self.assertEqual(with_exit["predicted_transition"], "close_event")

    def test_semantic_absence_alone_cannot_close(self) -> None:
        result = fusion.decide_transition(
            previous_state="risk", normalized_signed_change=0.0,
            semantic_exit=True, trusted_reference=True,
        )
        self.assertEqual(result["predicted_transition"], "uncertain")

    def test_conflicting_rise_and_exit_stays_uncertain(self) -> None:
        result = fusion.decide_transition(
            previous_state="risk", normalized_signed_change=0.2,
            semantic_exit=True, trusted_reference=True,
        )
        self.assertEqual(result["predicted_transition"], "uncertain")

    def test_missing_trusted_reference_stays_uncertain(self) -> None:
        result = fusion.decide_transition(
            previous_state="risk", normalized_signed_change=-0.8,
            semantic_exit=False, trusted_reference=False,
        )
        self.assertEqual(result["predicted_transition"], "uncertain")

    def test_strong_decrease_can_close_with_semantic_support(self) -> None:
        result = fusion.decide_transition(
            previous_state="risk", normalized_signed_change=-0.8,
            semantic_exit=True, trusted_reference=True,
        )
        self.assertEqual(result["predicted_transition"], "close_event")
        self.assertEqual(result["reason"], "strong_relative_decrease_with_semantic_support")

    def test_invalid_margin_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "margin"):
            fusion.decide_transition(
                previous_state="risk", normalized_signed_change=-0.8,
                semantic_exit=False, trusted_reference=True, strong_margin=0.0,
            )


if __name__ == "__main__":
    unittest.main()
