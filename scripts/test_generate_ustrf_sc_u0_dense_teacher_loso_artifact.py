from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate_ustrf_sc_u0_dense_teacher_loso_artifact.py")
SPEC = importlib.util.spec_from_file_location("dense_fit", SCRIPT)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


class DenseTeacherLosoArtifactTest(unittest.TestCase):
    def manifest(self) -> dict:
        return {
            "schema": subject.TRAINING_MANIFEST_SCHEMA,
            "contract_id": "ustrf_sc_u0_teacher_upper_bound_v1",
            "arm_id": subject.ARM_ID,
            "candidate_adapter_id": subject.ADAPTER_ID,
            "fit_policy": subject.FIT_POLICY,
            "held_out_session_id": "held-out",
            "truth_manifest_sha256": "0" * 64,
            "training_session_ids": ["train-a"],
            "training_episode_ids": ["episode-a"],
            "held_out_inputs_used": False,
            "blind_accessed": False,
            "future_inputs_used": False,
        }

    def calibration(self) -> dict:
        return {
            "schema": subject.CALIBRATION_SCHEMA,
            "contract_id": "ustrf_sc_u0_teacher_upper_bound_v1",
            "arm_id": subject.ARM_ID,
            "candidate_adapter_id": subject.ADAPTER_ID,
            "fit_policy": subject.FIT_POLICY,
            "held_out_session_id": "held-out",
            "training_input_manifest_sha256": "1" * 64,
            "training_session_ids": ["train-a"],
            "episodes": [{
                "session_id": "train-a",
                "episode_id": "episode-a",
                "video_path": "video.mp4",
                "video_sha256": "2" * 64,
                "frames": [{"frame_id": "frame-a", "video_pts_ms": 0}],
            }],
            "held_out_inputs_used": False,
            "blind_accessed": False,
            "future_inputs_used": False,
            "human_event_truth_used": False,
            "production_authorized": False,
        }

    def test_exact_fold_inventory_is_accepted_without_labels(self) -> None:
        subject.validate_inputs(self.manifest(), self.calibration(), training_manifest_sha256="1" * 64)

    def test_held_out_or_label_injection_fails_closed(self) -> None:
        held_out = self.calibration()
        held_out["episodes"][0]["session_id"] = "held-out"
        with self.assertRaisesRegex(subject.FitError, "outside the training fold"):
            subject.validate_inputs(self.manifest(), held_out, training_manifest_sha256="1" * 64)
        label = self.calibration()
        label["episodes"][0]["should_alert"] = True
        with self.assertRaisesRegex(subject.FitError, "forbidden"):
            subject.validate_inputs(self.manifest(), label, training_manifest_sha256="1" * 64)

    def test_fold_calibration_is_finite_and_non_degenerate(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        first = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
        second = np.linspace(0.5, 2.0, 16, dtype=np.float32).reshape(4, 4)
        fitted = subject.fit_calibration([first, second])
        self.assertLess(fitted["raw_depth_lower_quantile"], fitted["raw_depth_upper_quantile"])
        self.assertGreater(fitted["gradient_upper_quantile"], 0.0)
        self.assertEqual(2, fitted["training_frame_count"])


if __name__ == "__main__":
    unittest.main()
