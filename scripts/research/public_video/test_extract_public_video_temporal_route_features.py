import unittest

import numpy as np

import extract_public_video_temporal_route_features as subject


class TemporalRouteFeatureExtractionTest(unittest.TestCase):
    def test_causal_flow_grid_has_two_channels_per_past_frame(self):
        current = np.zeros((32, 48, 3), dtype=np.uint8)
        past = [current.copy(), current.copy(), current.copy()]
        grid = subject.causal_flow_grid(current, past, 8)
        self.assertEqual((8, 8, 6), grid.shape)
        self.assertTrue(np.isfinite(grid).all())

    def test_composed_grid_uses_channel_first_layout(self):
        token = np.zeros((8, 8, 4), dtype=np.float32)
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        flow = np.zeros((8, 8, 6), dtype=np.float32)
        result = subject.compose_feature_grid(token, image, flow)
        self.assertEqual((15, 8, 8), result.shape)


if __name__ == "__main__":
    unittest.main()
