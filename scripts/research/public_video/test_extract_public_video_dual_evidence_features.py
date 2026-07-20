import unittest

import extract_public_video_dual_evidence_features as extractor


class DualEvidenceFeatureExtractorTest(unittest.TestCase):
    def test_sample_timestamps_cover_complete_half_open_duration(self) -> None:
        self.assertEqual(extractor.sample_timestamps(3500, 1000), [0, 1000, 2000, 3000])

    def test_exact_duration_does_not_add_out_of_range_sample(self) -> None:
        self.assertEqual(extractor.sample_timestamps(3000, 1000), [0, 1000, 2000])

    def test_invalid_sampling_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            extractor.sample_timestamps(0, 1000)

    def test_semantic_ids_use_only_frozen_groups(self) -> None:
        names = {0: "person", 1: "sand", 2: "barrier", 3: "road"}
        groups = {"surface_material": ["sand"], "barrier_structure": ["barrier"]}
        self.assertEqual(extractor.semantic_class_ids(names, groups), [1, 2])

    def test_semantic_model_without_frozen_classes_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "none"):
            extractor.semantic_class_ids({0: "person"}, {"surface_material": ["sand"]})


if __name__ == "__main__":
    unittest.main()
