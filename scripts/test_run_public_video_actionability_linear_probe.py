import unittest

import numpy as np

import run_public_video_actionability_linear_probe as subject


class ActionabilityLinearProbeTest(unittest.TestCase):
    def test_merge_source_accepts_disjoint_samples(self) -> None:
        target = {"source_id": "s", "local_video_path": "v", "video_sha256": "h", "samples": [{"timestamp_ms": 1}]}
        incoming = {"source_id": "s", "local_video_path": "v", "video_sha256": "h", "samples": [{"timestamp_ms": 2}]}
        subject.merge_source(target, incoming)
        self.assertEqual([1, 2], [row["timestamp_ms"] for row in target["samples"]])

    def test_build_event_rows_skips_empty_marker_event(self) -> None:
        manifest = {"items": [
            {"item_id": "kept", "parent_source_id": "s", "window_ms": [0, 2000], "intervention_required": True},
            {"item_id": "empty", "parent_source_id": "s", "window_ms": [2000, 3000], "intervention_required": False},
        ]}
        sources = {"s": {"samples": [
            {"timestamp_ms": 1000, "detections": [{"class_name": "traffic cone"}]},
            {"timestamp_ms": 2000, "detections": []},
        ]}}
        rows = subject.build_event_rows(manifest, sources)
        self.assertEqual(["kept"], [row["event_id"] for row in rows])

    def test_source_loso_probe_is_finite(self) -> None:
        x = np.asarray([[0.0], [0.1], [0.9], [1.0], [0.2], [0.8]])
        y = np.asarray([0, 0, 1, 1, 0, 1])
        sources = np.asarray(["a", "b", "c", "d", "e", "f"])
        scores, folds = subject.source_loso_probe(x, y, sources, 1.0)
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(6, len(folds))


if __name__ == "__main__":
    unittest.main()
