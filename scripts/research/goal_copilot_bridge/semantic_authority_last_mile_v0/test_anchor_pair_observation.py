import unittest

from .anchor_pair_observation import AnchorBoundaryHypothesis, _pair_score
from .two_view_observation import BoundaryGeometry, ImageLine


def _boundary(x: float, coverage: float) -> AnchorBoundaryHypothesis:
    line = ImageLine((1.0, 0.0, -x), coverage * 144.0, 1)
    return AnchorBoundaryHypothesis(line, line, 2.0, x, x, coverage, 0.6, 0.8)


def _geometry() -> BoundaryGeometry:
    return BoundaryGeometry(0.0, 0.95, 2.0, 0.8, (-0.4, 0.0, 2.0), (0.4, 0.0, 2.0), 1.0, 1.0, 1.0)


class AnchorPairObservationTest(unittest.TestCase):
    def test_balanced_coverage_beats_one_strong_one_weak_boundary(self) -> None:
        balanced, _, _ = _pair_score(_boundary(70.0, 0.45), _boundary(170.0, 0.45), _geometry(), 120.0, 256)
        unbalanced, _, _ = _pair_score(_boundary(70.0, 0.90), _boundary(170.0, 0.20), _geometry(), 120.0, 256)
        self.assertGreater(balanced, unbalanced)

    def test_anchor_bracketing_is_explicit_pair_component(self) -> None:
        _, relation, components = _pair_score(
            _boundary(70.0, 0.45), _boundary(170.0, 0.45), _geometry(), 120.0, 256
        )
        self.assertEqual(relation, "BRACKETS_ANCHOR")
        self.assertEqual(components["anchor_bracketing"], 1.0)


if __name__ == "__main__":
    unittest.main()
