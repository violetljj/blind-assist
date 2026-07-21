from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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

    def test_runtime_timing_does_not_change_the_hash_bound_artifact(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "video.mp4"
            model = root / "teacher.onnx"
            video.write_bytes(b"video-fixture")
            model.write_bytes(b"model-fixture")
            manifest_path = root / "training.json"
            manifest_path.write_text(json.dumps(self.manifest(), sort_keys=True), encoding="utf-8")
            manifest_sha = subject.sha256_file(manifest_path)
            calibration = self.calibration()
            calibration["training_input_manifest_sha256"] = manifest_sha
            calibration["episodes"][0]["video_sha256"] = subject.sha256_file(video)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(calibration, sort_keys=True), encoding="utf-8")
            model_sha = subject.sha256_file(model)
            depth = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
            rgb = np.zeros((4, 4, 3), dtype=np.uint8)

            class FakeTeacher:
                def __init__(self, _path: Path) -> None:
                    self.model_sha256 = model_sha

                def infer_rgb(self, _rgb: object):
                    return depth, 999.0

            def execute(suffix: str, decode_duration: float) -> tuple[bytes, bytes]:
                artifact = root / f"artifact-{suffix}.json"
                receipt = root / f"receipt-{suffix}.json"
                args = SimpleNamespace(
                    training_input_manifest=manifest_path,
                    calibration_inputs=calibration_path,
                    inference_root=root,
                    teacher_model=model,
                    expected_teacher_model_sha256=model_sha,
                    artifact_output=artifact,
                    training_receipt_output=receipt,
                )
                with mock.patch.object(subject, "DepthAnythingOnnxTeacher", FakeTeacher), mock.patch.object(
                    subject, "decode_video_frame_rgb", return_value=(rgb, decode_duration)
                ):
                    subject.run(args)
                return artifact.read_bytes(), receipt.read_bytes()

            first = execute("first", 1.0)
            second = execute("second", 10_000.0)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
