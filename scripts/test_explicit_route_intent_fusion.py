import unittest

import explicit_route_intent_fusion as subject


class ExplicitRouteIntentFusionTest(unittest.TestCase):
    def sample(self, timestamp: int, score: float | None, valid: bool = True) -> subject.RouteRiskSample:
        return subject.RouteRiskSample(timestamp, valid, score)

    def test_missing_route_never_opens_or_clears(self) -> None:
        rows = [self.sample(0, 1.0, False), self.sample(1000, 1.0, False)]
        self.assertEqual([], subject.decode_route_risk_lifecycle(rows))

    def test_two_active_samples_open_and_two_clear_samples_close(self) -> None:
        rows = [self.sample(0, 1.0), self.sample(1000, 1.0),
                self.sample(2000, 0.0), self.sample(3000, 0.0)]
        self.assertEqual([
            {"state": "intervention_needed", "timestamp_ms": 1000},
            {"state": "route_clear", "timestamp_ms": 3000},
        ], subject.decode_route_risk_lifecycle(rows))

    def test_timestamp_gap_resets_open_run(self) -> None:
        rows = [self.sample(0, 1.0), self.sample(2000, 1.0)]
        self.assertEqual([], subject.decode_route_risk_lifecycle(rows))

    def test_unknown_route_does_not_fake_clear(self) -> None:
        rows = [self.sample(0, 1.0), self.sample(1000, 1.0),
                self.sample(2000, None, False), self.sample(3000, None, False)]
        transitions = subject.decode_route_risk_lifecycle(rows)
        self.assertEqual([{"state": "intervention_needed", "timestamp_ms": 1000}], transitions)


if __name__ == "__main__":
    unittest.main()
