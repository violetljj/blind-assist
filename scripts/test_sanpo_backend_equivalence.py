from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanpo_backend_equivalence as gate


class SanpoBackendEquivalenceTest(unittest.TestCase):
    def write_authorization(self, root: Path, weights: Path, **overrides: object) -> Path:
        model_path = Path(gate.sanpo_segmentation_model.__file__).resolve()
        report = {
            "schema": gate.REPORT_SCHEMA,
            "status": "green",
            "export_authorized": True,
            "weights_sha256": gate.sha256_file(weights),
            "model_config": gate.model_config(
                gate.DEFAULT_BACKBONE_ALPHA, gate.DEFAULT_DECODER_CHANNELS,
            ),
            "model_config_sha256": gate.model_config_sha256(
                gate.model_config(gate.DEFAULT_BACKBONE_ALPHA, gate.DEFAULT_DECODER_CHANNELS)
            ),
            "model_definition_sha256": gate.sha256_file(model_path),
            "equivalence_tool_sha256": gate.sha256_file(Path(gate.__file__).resolve()),
            "fixed_input": {
                "kind": "deterministic_synthetic_rgb_0_255",
                "seed": gate.FIXED_INPUT_SEED,
                "count": gate.FIXED_INPUT_COUNT,
                "shape": [gate.FIXED_INPUT_COUNT, gate.INPUT_SIZE, gate.INPUT_SIZE, 3],
                "sha256": "1" * 64,
                "dataset_access": "none",
                "blind_holdout_access": "not_accessed",
            },
            "thresholds": {
                "max_abs_lte": gate.MAX_ABS_THRESHOLD,
                "argmax_agreement_gte": gate.ARGMAX_AGREEMENT_THRESHOLD,
            },
            "metrics": {
                "max_abs": gate.MAX_ABS_THRESHOLD / 2,
                "mean_abs": gate.MAX_ABS_THRESHOLD / 10,
                "argmax_agreement": 1.0,
                "passed": True,
            },
            "torch_execution_contract": gate.TORCH_EXECUTION_CONTRACT,
        }
        report.update(overrides)
        path = root / "equivalence.json"
        gate.write_json(path, report)
        path.with_suffix(".json.sha256").write_text(
            f"{gate.sha256_file(path)}  {path.name}\n", encoding="ascii",
        )
        return path

    def test_fixed_inputs_are_deterministic_and_dataset_free(self) -> None:
        first = gate.fixed_inputs()
        second = gate.fixed_inputs()
        self.assertEqual((4, 256, 256, 3), first.shape)
        np.testing.assert_array_equal(first, second)

    def test_fixed_inputs_follow_resolution_and_have_distinct_hashes(self) -> None:
        hashes: set[str] = set()
        for input_size in gate.ALLOWED_INPUT_SIZES:
            values = gate.fixed_inputs(input_size)
            self.assertEqual((gate.FIXED_INPUT_COUNT, input_size, input_size, 3), values.shape)
            hashes.add(hashlib.sha256(values.tobytes()).hexdigest())
        self.assertEqual(len(gate.ALLOWED_INPUT_SIZES), len(hashes))

    def test_compare_records_max_abs_and_argmax_agreement(self) -> None:
        left = np.array([[[[0.0, 1.0, 2.0, 3.0]]]], dtype=np.float32)
        right = left.copy()
        result = gate.compare_logits(left, right)
        self.assertEqual(0.0, result["max_abs"])
        self.assertEqual(1.0, result["argmax_agreement"])
        self.assertTrue(result["passed"])

    def test_consumer_accepts_only_bound_green_preregistered_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "candidate.weights.h5"
            weights.write_bytes(b"weights")
            report = self.write_authorization(root, weights)
            verified = gate.consume_equivalence_authorization(weights, report)
            self.assertEqual("green", verified["status"])
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["metrics"]["max_abs"] = gate.MAX_ABS_THRESHOLD * 2
            gate.write_json(report, payload)
            report.with_suffix(".json.sha256").write_text(
                gate.sha256_file(report) + "\n", encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "does not authorize"):
                gate.consume_equivalence_authorization(weights, report)

    def test_consumer_rejects_threshold_or_weight_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "candidate.weights.h5"
            weights.write_bytes(b"weights")
            report = self.write_authorization(root, weights, thresholds={"max_abs_lte": 1.0})
            with self.assertRaisesRegex(ValueError, "preregistered"):
                gate.consume_equivalence_authorization(weights, report)
            report = self.write_authorization(root, weights)
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "different weights"):
                gate.consume_equivalence_authorization(weights, report)

    def test_consumer_rejects_missing_exact_float32_torch_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "candidate.weights.h5"
            weights.write_bytes(b"weights")
            report = self.write_authorization(root, weights, torch_execution_contract={})
            with self.assertRaisesRegex(ValueError, "exact-float32"):
                gate.consume_equivalence_authorization(weights, report)

    def test_consumer_rejects_requested_model_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "candidate.weights.h5"
            weights.write_bytes(b"weights")
            report = self.write_authorization(root, weights)
            with self.assertRaisesRegex(ValueError, "requested export config"):
                gate.consume_equivalence_authorization(
                    weights,
                    report,
                    backbone_alpha=1.0,
                    decoder_channels=gate.DEFAULT_DECODER_CHANNELS,
                )
            with self.assertRaisesRegex(ValueError, "requested export config"):
                gate.consume_equivalence_authorization(
                    weights,
                    report,
                    backbone_alpha=gate.DEFAULT_BACKBONE_ALPHA,
                    decoder_channels=128,
                )
            with self.assertRaisesRegex(ValueError, "requested export config"):
                gate.consume_equivalence_authorization(
                    weights,
                    report,
                    input_size=512,
                )

    def test_consumer_rejects_model_config_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "candidate.weights.h5"
            weights.write_bytes(b"weights")
            report = self.write_authorization(root, weights, model_config_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "config hash mismatch"):
                gate.consume_equivalence_authorization(weights, report)

    def test_worker_command_carries_model_config(self) -> None:
        command = gate.worker_command(
            Path("python"), "torch", Path("weights"), Path("inputs"), Path("output"), 1.0, 128,
        )
        self.assertEqual("1.0", command[command.index("--backbone-alpha") + 1])
        self.assertEqual("128", command[command.index("--decoder-channels") + 1])
        self.assertEqual("256", command[command.index("--input-size") + 1])

    def test_worker_command_carries_nondefault_input_size(self) -> None:
        command = gate.worker_command(
            Path("python"), "tensorflow", Path("weights"), Path("inputs"), Path("output"),
            0.75, 96, 512,
        )
        self.assertEqual("512", command[command.index("--input-size") + 1])

    def test_consumer_rejects_sidecar_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "candidate.weights.h5"
            weights.write_bytes(b"weights")
            report = self.write_authorization(root, weights)
            report.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sidecar mismatch"):
                gate.consume_equivalence_authorization(weights, report)


if __name__ == "__main__":
    unittest.main()
