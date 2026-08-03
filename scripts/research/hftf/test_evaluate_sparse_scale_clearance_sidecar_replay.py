#!/usr/bin/env python3

import unittest

from evaluate_sparse_scale_clearance_sidecar_replay import evaluate


def bands(value: float) -> dict:
    return {
        band: {"clearance_m": value}
        for band in ("left", "center", "right")
    }


class EvaluateSparseScaleClearanceSidecarReplayTest(unittest.TestCase):
    def test_replay_requires_unknown_prefix_and_scores_suffix(self) -> None:
        sidecar = []
        truth_frames = []
        for index in range(12):
            sidecar.append(
                {
                    "sequence_id": "s",
                    "frame_index": index,
                    "depth_latency_ms": 1.0,
                    "geometry_and_scale_latency_ms": 0.1,
                    "scaled_clearance": (
                        {"status": "UNKNOWN_NO_METRIC_SCALE_ANCHOR"}
                        if index < 10
                        else {"status": "VALID", "bands": bands(1.0)}
                    ),
                }
            )
            truth_frames.append(
                {
                    "sequence_id": "s",
                    "timestamp": float(index),
                    "sensor": {"status": "VALID", "bands": bands(1.0)},
                }
            )
        result = evaluate(sidecar, {"frames": truth_frames})
        self.assertEqual(result["task_gates_passed"], 5)
        self.assertEqual(result["premature_valid_before_anchor"], [])


if __name__ == "__main__":
    unittest.main()
