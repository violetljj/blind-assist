import unittest

import numpy as np

import evaluate_public_video_majority_prototype_external_negative as subject


class MajorityPrototypeExternalNegativeTest(unittest.TestCase):
    def test_fit_models_returns_frozen_ensemble_size(self) -> None:
        contract = {
            "target": {"strong_intrusion_fraction_at_least": 2 / 3},
            "prototype_ensemble": {"seeds": [1, 2, 3, 4, 5], "final_seed_offset": 10},
        }
        original_load = subject.pair_probe.load_data
        original_pairs = subject.pair_probe.nearest_time_pairs
        subject.pair_probe.load_data = lambda _contract: (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
            np.asarray([0.0, 1.0, 0.0, 1.0]),
            np.asarray(["a", "a", "b", "b"]),
            np.asarray([0, 1, 0, 1]),
        )
        subject.pair_probe.nearest_time_pairs = lambda *_args: [
            {"positive_index": 1, "negative_index": 0, "source_id": "a"},
            {"positive_index": 3, "negative_index": 2, "source_id": "b"},
        ]
        try:
            models, audits = subject.fit_models(contract, {})
        finally:
            subject.pair_probe.load_data = original_load
            subject.pair_probe.nearest_time_pairs = original_pairs
        self.assertEqual(5, len(models))
        self.assertEqual(5, len(audits))

    def test_accepted_event_vectors_skips_gap_and_checks_count(self) -> None:
        report = {"source_id": "s", "frozen_radial_event": {
            "event_entry_timestamp_ms": 0, "last_active_timestamp_ms": 2000,
            "accepted_sample_count": 2}}
        features = {"sources": [{"source_id": "s"}]}
        original = subject.prospective._build_features
        subject.prospective._build_features = lambda *_args: (
            np.ones((3, 2, 2, 2)),
            [{"detections": [{"features": {"center_x_norm": .5, "bottom_y_norm": .5,
                                              "height_norm": .2}}]},
             {"detections": []},
             {"detections": [{"features": {"center_x_norm": .5, "bottom_y_norm": .5,
                                              "height_norm": .2}}]}],
        )
        original_mask = subject.linear.marker_grid_mask
        subject.linear.marker_grid_mask = lambda detections, *_args: np.asarray(
            [[bool(detections), False], [False, False]])
        try:
            values, timestamps = subject.accepted_event_vectors(
                report, features, {}, None, .5, 8)
        finally:
            subject.prospective._build_features = original
            subject.linear.marker_grid_mask = original_mask
        self.assertEqual((2, 9), values.shape)
        self.assertEqual([0, 2000], timestamps)


if __name__ == "__main__":
    unittest.main()
