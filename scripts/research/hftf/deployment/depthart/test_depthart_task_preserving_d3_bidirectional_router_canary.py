import unittest

from scripts.research.hftf.deployment.depthart.depthart_task_preserving_d3_bidirectional_router_canary import (
    RouterPolicy,
    compose_route,
    run_canary,
)


class D3BidirectionalRouterCanaryTest(unittest.TestCase):
    def test_neutral_certificates_are_identity_composition(self) -> None:
        result = compose_route(
            baseline_states=("CLEAR", "CLEAR", "OCCUPIED"),
            clear_certificates=(0.5, 0.5, 0.5),
            occupied_certificates=(0.5, 0.5, 0.5),
            hard_evidence=(True, True, True),
        )
        self.assertEqual(result["final_states"], result["baseline_states"])

    def test_strong_conflicting_certificates_fail_closed(self) -> None:
        result = compose_route(
            baseline_states=("CLEAR", "CLEAR", "CLEAR"),
            clear_certificates=(0.95, 0.95, 0.95),
            occupied_certificates=(0.95, 0.95, 0.95),
            hard_evidence=(True, True, True),
        )
        self.assertEqual(result["final_states"], ["UNKNOWN_GROUND"] * 3)

    def test_no_hard_evidence_cannot_override_known_baseline(self) -> None:
        result = compose_route(
            baseline_states=("CLEAR", "CLEAR", "CLEAR"),
            clear_certificates=(0.0, 0.0, 0.0),
            occupied_certificates=(1.0, 1.0, 1.0),
            hard_evidence=(False, False, False),
        )
        self.assertEqual(result["final_states"], ["CLEAR"] * 3)

    def test_invalid_probability_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            compose_route(
                baseline_states=("CLEAR", "CLEAR", "CLEAR"),
                clear_certificates=(0.5, float("nan"), 0.5),
                occupied_certificates=(0.5, 0.5, 0.5),
                hard_evidence=(True, True, True),
            )

    def test_invalid_shape_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            compose_route(
                baseline_states=("CLEAR", "CLEAR", "CLEAR"),
                clear_certificates=(0.5, 0.5),
                occupied_certificates=(0.5, 0.5, 0.5),
                hard_evidence=(True, True, True),
            )

    def test_policy_cannot_disable_hard_evidence(self) -> None:
        with self.assertRaises(ValueError):
            RouterPolicy(hard_evidence_required_for_override=False).validate()

    def test_frozen_canary_passes(self) -> None:
        result = run_canary()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
