import unittest

from prepare_external_rgb_video_manifest import sample_frame_indices


class PrepareExternalRgbVideoManifestTest(unittest.TestCase):
    def test_samples_thirty_fps_at_ten_fps(self) -> None:
        self.assertEqual(sample_frame_indices(30.0, 30, 10.0), list(range(0, 30, 3)))


if __name__ == "__main__":
    unittest.main()
