#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depthart_admission_r0 as admission


class DepthArtAdmissionR0Test(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "gate_tolerances": {
                "ground_recovery_rate_drop": 0.01,
                "clearance_mae_increase_m": 0.025,
                "collision_agreement_drop": 0.02,
                "false_clear_rate_increase": 0.01,
                "truth_status_exact_drop": 0.01,
                "false_block_rate_increase": 0.01,
                "temporal_clearance_delta_mae_increase_m": 0.015,
                "temporal_depth_delta_mae_increase_m": 0.05,
                "frame_scale_drift_increase": 0.02,
            }
        }
        baseline_task = {
            "ground_recovery_success_rate": 1.0,
            "clearance_mae_m": 0.38,
            "collision_agreement": 0.80,
            "false_clear_rate_all_known_decisions": 0.24,
            "temporal_clearance_delta_mae_m": 0.11,
        }
        candidate_task = dict(baseline_task)
        self.projection = {
            "parent_r0_result": {"baseline": {"task": baseline_task}, "candidate": {"task": candidate_task}},
            "baseline_truth_geometry": {"truth_status_exact_agreement": 0.99, "false_block_rate_all_known": 0.01},
            "candidate_truth_geometry": {"truth_status_exact_agreement": 0.99, "false_block_rate_all_known": 0.01},
        }
        self.temporal = {
            "baseline_depth_delta_mae_m": 0.10,
            "candidate_depth_delta_mae_m": 0.10,
            "baseline_scale_drift_median": 0.01,
            "candidate_scale_drift_median": 0.01,
        }

    def test_equal_candidate_passes(self):
        result = admission.decide_gates(self.protocol, self.projection, self.temporal)
        self.assertTrue(result["gate_1_passed"])
        self.assertTrue(result["gate_2_passed"])

    def test_undefined_false_clear_fails_closed(self):
        self.projection["parent_r0_result"]["candidate"]["task"]["false_clear_rate_all_known_decisions"] = math.nan
        result = admission.decide_gates(self.protocol, self.projection, self.temporal)
        self.assertFalse(result["gate_1_task_quality"]["false_clear"])
        self.assertFalse(result["gate_1_passed"])

    def test_temporal_failure_cannot_be_compensated(self):
        self.temporal["candidate_depth_delta_mae_m"] = 0.151
        result = admission.decide_gates(self.protocol, self.projection, self.temporal)
        self.assertTrue(result["gate_1_passed"])
        self.assertFalse(result["gate_2_temporal_quality"]["depth_delta"])
        self.assertFalse(result["gate_2_passed"])


if __name__ == "__main__":
    unittest.main()
