import unittest

import numpy as np

import run_public_video_marker_relation_geometry_residual_probe as subject


class MarkerRelationGeometryResidualProbeTest(unittest.TestCase):
    def test_residualizer_removes_geometry_driven_semantic_change(self) -> None:
        geometry = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        semantic = 2.0 * geometry[:, :1]
        x = np.concatenate([semantic, geometry], axis=1)
        pairs = [{"source_id": "a", "positive_index": 1, "negative_index": 0},
                 {"source_id": "b", "positive_index": 2, "negative_index": 1}]
        mapping = subject.fit_geometry_residualizer(x, pairs, np.asarray([True, True]), 1e-9)
        residual = subject.residualize_frames(x, mapping)
        self.assertLess(float(np.ptp(residual[:, 0])), 1e-6)
        np.testing.assert_allclose(0.0, residual[:, -3:])

    def test_source_weights_are_equal(self) -> None:
        sources = np.asarray(["a", "a", "b"])
        weights = subject.source_equal_pair_weights(sources)
        self.assertAlmostEqual(0.5, float(weights[:2].sum()))
        self.assertAlmostEqual(0.5, float(weights[2]))


if __name__ == "__main__":
    unittest.main()
