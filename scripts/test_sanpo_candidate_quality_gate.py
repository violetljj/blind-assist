from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanpo_candidate_quality_gate as gate
import train_export_sanpo_segmentation as shared


def record(sample: str, session: str, scene: str | None) -> shared.Record:
    return shared.Record(
        sample_id=sample,
        split="dev",
        session_id=session,
        image_path=Path("unused.png"),
        masks={},
        semantic_mask_path=None,
        scene_bucket=scene,
        label_authority="source_ground_truth",
    )


class CandidateQualityGateTest(unittest.TestCase):
    def test_ai_release_review_replaces_human_release_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_sha = "a" * 64
            receipt = root / "release-review.json"
            receipt.write_text(json.dumps({
                "schema": "blindassist_ai_review_consensus_v1",
                "subject_id": "sanpo-release:model-sha",
                "input_sha256": input_sha,
                "reviews": [
                    {
                        "reviewer_id": "gpt-release-1", "reviewer_type": "ai_model",
                        "reviewer_role": "gpt_release_reviewer", "provider": "openai",
                        "model": "gpt-multimodal", "model_version": "2026-07-21",
                        "review_run_id": "gpt-release-run", "workflow_id": "sanpo_release_review_v1",
                        "prompt_sha256": "1" * 64, "input_sha256": input_sha,
                        "isolated_context": True, "other_review_visible_before_submission": False,
                        "confidence": 0.91, "abstained": False, "abstain_reasons": [], "verdict": "accept",
                    },
                    {
                        "reviewer_id": "codex-release-1", "reviewer_type": "ai_model",
                        "reviewer_role": "codex_evidence_reviewer", "provider": "openai",
                        "model": "codex", "model_version": "2026-07-21",
                        "review_run_id": "codex-release-run", "workflow_id": "sanpo_release_review_v1",
                        "prompt_sha256": "2" * 64, "input_sha256": input_sha,
                        "isolated_context": True, "other_review_visible_before_submission": False,
                        "confidence": 0.89, "abstained": False, "abstain_reasons": [], "verdict": "accept",
                    },
                ],
                "consensus": {"method": "model_consensus", "disposition": "accept"},
            }), encoding="utf-8")
            result = gate.evaluate_ai_release_review(
                receipt, subject_id="sanpo-release:model-sha", release_input_sha256=input_sha,
            )
            self.assertTrue(result["passed"])
            self.assertEqual("green", result["status"])

    def test_quality_gate_cli_carries_model_config(self) -> None:
        args = gate.parse_args([
            "--dataset-root", "dataset",
            "--weights", "candidate.weights.h5",
            "--backend-equivalence-report", "equivalence.json",
            "--report", "quality.json",
            "--backbone-alpha", "1.0",
            "--decoder-channels", "128",
            "--input-size", "512",
            "--detail-output-stride", "4",
            "--semantic-output-stride", "16",
        ])
        self.assertEqual(1.0, args.backbone_alpha)
        self.assertEqual(128, args.decoder_channels)
        self.assertEqual(512, args.input_size)
        self.assertEqual(4, args.detail_output_stride)
        self.assertEqual(16, args.semantic_output_stride)

    def test_stratifies_by_session_and_scene_and_macro_averages_sessions(self) -> None:
        records = [record("a", "short", "curb"), record("b", "long", "night"), record("c", "long", "night")]
        targets = [
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            np.array([[0, 0], [3, 3]], dtype=np.uint8),
            np.array([[0, 0], [3, 3]], dtype=np.uint8),
        ]
        predictions = [targets[0].copy(), np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)]
        metrics = gate.stratified_metrics(records, predictions, targets)
        self.assertEqual({"short", "long"}, set(metrics["per_session"]))
        self.assertEqual({"curb", "night"}, set(metrics["per_scene"]))
        expected = np.mean([metrics["per_session"]["short"]["mean_iou"], metrics["per_session"]["long"]["mean_iou"]])
        self.assertAlmostEqual(expected, metrics["macro_session_mean_iou"])
        self.assertNotAlmostEqual(metrics["macro_session_mean_iou"], metrics["global"]["mean_iou"])

    def test_boundary_and_unknown_abstention_are_explicit(self) -> None:
        target = np.array([[1, 1, 3, 3], [0, 2, 3, 0]], dtype=np.uint8)
        prediction = np.array([[1, 0, 3, 0], [0, 2, 3, 3]], dtype=np.uint8)
        metrics = gate.metrics_from_confusion(gate.confusion_matrix([prediction], [target]))
        self.assertAlmostEqual(1.0, metrics["boundary"]["precision"])
        self.assertAlmostEqual(1 / 2, metrics["boundary"]["recall"])
        self.assertAlmostEqual(3 / 8, metrics["unknown_abstention"]["abstain_rate"])
        self.assertAlmostEqual(5 / 8, metrics["unknown_abstention"]["known_coverage"])
        self.assertAlmostEqual(2 / 3, metrics["unknown_abstention"]["unknown_precision"])

    def test_missing_scene_bucket_is_a_hard_quality_failure(self) -> None:
        targets = [np.array([[0, 1], [2, 3]], dtype=np.uint8)] * 2
        records = [record("a", "s1", None), record("b", "s2", "night")]
        metrics = gate.stratified_metrics(records, targets, targets)
        result = gate.evaluate_training_quality(metrics, gate.QualityThresholds())
        self.assertFalse(result["passed"])
        self.assertIn("scene_bucket_complete", result["failed_checks"])

    def test_quantization_gate_catches_class_collapse_despite_high_pixel_agreement(self) -> None:
        target = np.zeros((20, 20), dtype=np.uint8)
        target[0, :] = 1
        reference = target.copy()
        quantized = target.copy()
        quantized[0, :] = 0
        metrics = gate.quantization_fidelity([reference], [quantized], [target])
        self.assertEqual(0.95, metrics["argmax_agreement"])
        self.assertEqual(0.0, metrics["per_class_prediction_iou"]["boundary_step_curb"])
        result = gate.evaluate_fidelity(metrics, gate.FidelityThresholds())
        self.assertFalse(result["passed"])
        self.assertIn("per_class_prediction_iou", result["failed_checks"])

    def test_identical_int8_predictions_pass_fidelity(self) -> None:
        prediction = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        metrics = gate.quantization_fidelity([prediction], [prediction.copy()], [prediction])
        result = gate.evaluate_fidelity(metrics, gate.FidelityThresholds())
        self.assertTrue(result["passed"])
        self.assertEqual(1.0, metrics["argmax_agreement"])

    def test_device_gate_is_independent_and_model_bound(self) -> None:
        missing = gate.evaluate_device_event_report(None, "a" * 64, gate.DeviceEventThresholds())
        self.assertEqual("not_evaluated", missing["status"])
        payload = {
            "schema": gate.DEVICE_REPORT_SCHEMA,
            "model_sha256": "a" * 64,
            "metrics": {
                "event_recall": 0.95,
                "critical_miss_rate": 0.01,
                "false_alerts_per_minute": 0.2,
                "post_event_clearance_rate": 0.95,
                "repeated_alert_rate": 0.05,
                "p95_latency_ms": 60.0,
            },
        }
        self.assertTrue(gate.evaluate_device_event_report(payload, "a" * 64, gate.DeviceEventThresholds())["passed"])
        with self.assertRaisesRegex(ValueError, "not bound"):
            gate.evaluate_device_event_report(payload, "b" * 64, gate.DeviceEventThresholds())

    def test_device_event_failure_does_not_relabel_offline_quality(self) -> None:
        payload = {
            "schema": gate.DEVICE_REPORT_SCHEMA,
            "model_sha256": "a" * 64,
            "metrics": {
                "event_recall": 0.99,
                "critical_miss_rate": 0.0,
                "false_alerts_per_minute": 2.0,
                "post_event_clearance_rate": 0.99,
                "repeated_alert_rate": 0.0,
                "p95_latency_ms": 50.0,
            },
        }
        result = gate.evaluate_device_event_report(payload, "a" * 64, gate.DeviceEventThresholds())
        self.assertEqual("red", result["status"])
        self.assertEqual(["false_alerts_per_minute"], result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
