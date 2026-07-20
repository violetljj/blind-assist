import unittest

import build_public_video_temporal_route_auxiliary_dataset as subject


class TemporalRouteAuxiliaryDatasetTest(unittest.TestCase):
    def test_evenly_spaced_keeps_all_short_inputs(self):
        self.assertEqual([1, 2, 3], subject.evenly_spaced([1, 2, 3], 4))

    def test_evenly_spaced_keeps_endpoints_and_limit(self):
        values = list(range(100))
        selected = subject.evenly_spaced(values, 8)
        self.assertEqual(8, len(selected))
        self.assertEqual(0, selected[0])
        self.assertEqual(99, selected[-1])
        self.assertEqual(sorted(set(selected)), selected)


if __name__ == "__main__":
    unittest.main()
