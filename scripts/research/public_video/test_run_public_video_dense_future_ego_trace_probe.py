import unittest

import numpy as np

import run_public_video_dense_future_ego_trace_probe as subject


class DenseFutureEgoTraceTest(unittest.TestCase):
    def test_zero_flow_preserves_anchor(self):
        flow = np.zeros((100, 200, 2), dtype=np.float32)
        point = subject.map_anchor_with_flow(flow, [0.5, 0.9])
        self.assertAlmostEqual(0.5, point[0])
        self.assertAlmostEqual(0.89, point[1])

    def test_constant_flow_maps_future_anchor_back(self):
        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[..., 0] = -20.0
        flow[..., 1] = -10.0
        point = subject.map_anchor_with_flow(flow, [0.5, 0.9])
        self.assertAlmostEqual(0.4, point[0])
        self.assertAlmostEqual(0.79, point[1])

    def test_rejects_nonfinite_flow(self):
        flow = np.zeros((10, 10, 2), dtype=np.float32)
        flow[0, 0, 0] = np.nan
        self.assertIsNone(subject.map_anchor_with_flow(flow, [0.5, 0.9]))


if __name__ == "__main__":
    unittest.main()
