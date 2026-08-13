from __future__ import annotations

import unittest

import numpy as np

from scripts.research.hftf.deployment.depthart import (
    run_depthart_d3r4_selective_router_canary as subject,
)


class SelectiveRouterCanaryTest(unittest.TestCase):
    def test_far_release_is_disabled_but_far_veto_is_enabled(self) -> None:
        dataset = {
            "baseline_state": np.asarray([subject.STATE_OCCUPIED, subject.STATE_OCCUPIED, subject.STATE_CLEAR], dtype=np.int8),
            "hard_evidence": np.asarray([True, True, True]),
            "horizon_index": np.asarray([0, 1, 2], dtype=np.int8),
            "parent_index": np.asarray([0, 0, 0], dtype=np.int16),
            "frame_index": np.asarray([0, 0, 0], dtype=np.int16),
            "band_index": np.asarray([0, 0, 0], dtype=np.int8),
        }
        release = np.asarray([0.95, 0.99, 0.0])
        veto = np.asarray([0.05, 0.01, 0.95])
        states, actions = subject.route_states(dataset, release, veto, 0.9)
        self.assertEqual(subject.STATE_CLEAR, states[0])
        self.assertEqual(subject.STATE_OCCUPIED, states[1])
        self.assertEqual(subject.STATE_OCCUPIED, states[2])
        self.assertEqual(1, actions["release_actions"])
        self.assertEqual(1, actions["veto_actions"])

    def test_unknown_source_frame_never_becomes_training_evidence(self) -> None:
        arrays = {key: [] for key in (
            "features", "truth_state", "baseline_state", "hard_evidence",
            "source_available", "parent_index", "frame_index", "band_index", "horizon_index",
        )}
        subject._append_unknown_frame(arrays, 0, 7)
        self.assertEqual(9, len(arrays["truth_state"]))
        self.assertTrue(all(value == subject.STATE_UNKNOWN for value in arrays["truth_state"]))
        self.assertTrue(all(value is False for value in arrays["source_available"]))
        self.assertTrue(all(value is False for value in arrays["hard_evidence"]))


if __name__ == "__main__":
    unittest.main()
