from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_stage_c_g0_d1_fresh as target  # noqa: E402

from evaluate_stage_c_g0_d1_fresh import (  # noqa: E402
    NOT_SUPPORTED,
    PREDICTION_SCHEMA,
    SUPPORTED,
    TRUTH_SCHEMA,
    _counts,
    _decision,
    _matrix,
    _metrics,
    _truth,
    _validate_exact_sets,
)


class FreshEvaluationTest(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_truth_preserves_unknown_and_clearance_sign(self) -> None:
        known = np.ones((2, 6, 6), dtype=int).tolist()
        risk = np.zeros((2, 6, 6), dtype=object).tolist()
        clearance = np.full(
            (2, 6, 6), 0.2, dtype=object
        ).tolist()
        known[0][0][0] = 0
        risk[0][0][0] = None
        clearance[0][0][0] = None
        risk[1][0][0] = 1
        clearance[1][0][0] = -0.1
        decoded = _truth(
            {
                "known_target": known,
                "risk_target_nullable": risk,
                "clearance_target_m_nullable": clearance,
            }
        )
        self.assertFalse(decoded[0][0, 0, 0])
        self.assertTrue(decoded[1][1, 0, 0])

    def test_metrics_use_known_truth_only(self) -> None:
        probability = np.asarray([0.9, 0.9, 0.1])
        truth = np.asarray([True, False, False])
        known = np.asarray([True, True, False])
        metrics = _metrics(_counts(probability, truth, known))
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_matrix_rejects_nonfinite_or_probability_drift(self) -> None:
        with self.assertRaises(ValueError):
            _matrix(np.full((2, 6, 6), np.nan))
        with self.assertRaises(ValueError):
            _matrix(np.full((2, 6, 6), 1.1), probability=True)

    @staticmethod
    def runs(
        *,
        clearance_f1: float,
        direct_f1: float,
        mae: float,
    ) -> list[dict]:
        runs = []
        for seed in (17, 29, 43):
            for arm, f1 in (
                ("DIRECT_RISK_CURRENT", direct_f1),
                ("SIGNED_CLEARANCE_CURRENT", clearance_f1),
            ):
                run = {
                    "seed": seed,
                    "arm": arm,
                    "risk_micro": {
                        "f1": f1,
                        "recall": 0.8,
                        "false_positive_rate": 0.1,
                    },
                    "risk_by_height": {
                        height: {"f1": f1}
                        for height in ("body", "head")
                    },
                    "risk_by_source": {
                        source: {"f1": f1}
                        for source in ("a", "b", "c")
                    },
                }
                if arm == "SIGNED_CLEARANCE_CURRENT":
                    run["clearance_source_height_macro_mae_m"] = {
                        "overall": mae,
                        "risk": mae,
                        "safe": mae,
                        "near": mae,
                    }
                    run[
                        "raw_prediction_out_of_target_range_fraction"
                    ] = 0.0
                runs.append(run)
        return runs

    def test_all_frozen_gates_required(self) -> None:
        _, gates, terminal = _decision(
            self.runs(
                clearance_f1=0.7,
                direct_f1=0.6,
                mae=0.05,
            ),
            ["a", "b", "c"],
            0,
        )
        self.assertTrue(all(gates.values()))
        self.assertEqual(terminal, SUPPORTED)

        _, gates, terminal = _decision(
            self.runs(
                clearance_f1=0.62,
                direct_f1=0.60,
                mae=0.05,
            ),
            ["a", "b", "c"],
            0,
        )
        self.assertFalse(gates["median_micro_f1_delta"])
        self.assertEqual(terminal, NOT_SUPPORTED)

    def test_one_bad_seed_fails_max_seed_mae_gate(self) -> None:
        runs = self.runs(
            clearance_f1=0.7,
            direct_f1=0.6,
            mae=0.05,
        )
        clearance_runs = [
            run
            for run in runs
            if run["arm"] == "SIGNED_CLEARANCE_CURRENT"
        ]
        clearance_runs[-1]["clearance_source_height_macro_mae_m"][
            "overall"
        ] = 0.100001
        _, gates, terminal = _decision(
            runs,
            ["a", "b", "c"],
            0,
        )
        self.assertFalse(gates["overall_clearance_mae"])
        self.assertEqual(terminal, NOT_SUPPORTED)

    def test_exact_sets_reject_truth_order_drift(self) -> None:
        sources = ["a", "b", "c"]
        frames = {source: list(range(25)) for source in sources}
        checkpoints = [
            {
                "seed": seed,
                "arm": arm,
                "sha256": f"{index + 1:064x}",
            }
            for index, (seed, arm) in enumerate(
                (seed, arm)
                for seed in (17, 29, 43)
                for arm in (
                    "DIRECT_RISK_CURRENT",
                    "SIGNED_CLEARANCE_CURRENT",
                )
            )
        ]
        labels = {
            "known_target": np.ones((2, 6, 6), dtype=int).tolist(),
            "risk_target_nullable": np.zeros(
                (2, 6, 6), dtype=int
            ).tolist(),
            "clearance_target_m_nullable": np.full(
                (2, 6, 6), 0.1
            ).tolist(),
        }
        truths = []
        for source in sources:
            for frame in frames[source]:
                truths.append(
                    {
                        "schema": TRUTH_SCHEMA,
                        "sample_id": f"{source}-{frame}",
                        "session_id": source,
                        "source_frame_index": frame,
                        "manifest_id": f"manifest-{source}-{frame}",
                        "labels": labels,
                    }
                )
        predictions = []
        for checkpoint in checkpoints:
            direct = checkpoint["arm"] == "DIRECT_RISK_CURRENT"
            for truth in truths:
                raw = np.zeros((2, 6, 6)) if direct else np.full(
                    (2, 6, 6), 0.1
                )
                predictions.append(
                    {
                        "schema": PREDICTION_SCHEMA,
                        "prediction_index": len(predictions),
                        "seed": checkpoint["seed"],
                        "arm": checkpoint["arm"],
                        "checkpoint_sha256": checkpoint["sha256"],
                        "sample_id": truth["sample_id"],
                        "session_id": truth["session_id"],
                        "source_frame_index": truth[
                            "source_frame_index"
                        ],
                        "manifest_id": truth["manifest_id"],
                        "raw_task_output": raw.tolist(),
                        "risk_probability": (
                            np.full((2, 6, 6), 0.5)
                            if direct
                            else np.zeros((2, 6, 6))
                        ).tolist(),
                        "known_probability": np.ones(
                            (2, 6, 6)
                        ).tolist(),
                    }
                )
        contract = {
            "checkpoint_contract": {"checkpoints": checkpoints},
        }
        package = {
            "source_order": sources,
            "source_frame_indices": frames,
        }
        truth_by_sample, prediction_by_key = _validate_exact_sets(
            contract, package, predictions, truths
        )
        self.assertEqual(len(truth_by_sample), 75)
        self.assertEqual(len(prediction_by_key), 450)
        truths[0], truths[1] = truths[1], truths[0]
        with self.assertRaisesRegex(ValueError, "cardinality"):
            _validate_exact_sets(
                contract, package, predictions, truths
            )

    def test_primary_join_receipt_precedes_single_truth_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            package_path = root / "package-validation/validation.json"
            authorization_path = (
                package_path.parent / "prediction_authorization.json"
            )
            truth_path = root / "truth.jsonl"
            completion_path = root / "predictions/completion.json"
            predictions_path = root / "predictions/predictions.jsonl"
            output_root = root / "effect"
            truth_path.parent.mkdir(parents=True, exist_ok=True)
            truth_path.write_bytes(b"sealed truth placeholder\n")
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_path.write_bytes(b"")
            checkpoints = [
                {
                    "seed": seed,
                    "arm": arm,
                    "sha256": f"{index + 1:064x}",
                }
                for index, (seed, arm) in enumerate(
                    (seed, arm)
                    for seed in (17, 29, 43)
                    for arm in (
                        "DIRECT_RISK_CURRENT",
                        "SIGNED_CLEARANCE_CURRENT",
                    )
                )
            ]
            contract = {
                "schema": target.CONTRACT_SCHEMA,
                "status": target.CONTRACT_STATUS,
                "implementations": {
                    "fresh_evaluator": {
                        "path": target.IMPLEMENTATION_PATH,
                        "sha256": target._sha256(
                            Path(target.__file__).resolve()
                        ),
                        "execution_authorized": True,
                    },
                    "fresh_package_validator": {
                        "sha256": "a" * 64
                    },
                    "fresh_predictor": {"sha256": "b" * 64},
                },
                "fresh_source_contract": {
                    "source_order": ["a", "b", "c"]
                },
                "checkpoint_contract": {"checkpoints": checkpoints},
            }
            self._write_json(contract_path, contract)
            contract_sha = target._sha256(contract_path)
            source_frames = {
                source: list(range(25))
                for source in ("a", "b", "c")
            }
            package = {
                "schema": target.PACKAGE_VALIDATION_SCHEMA,
                "terminal": target.PACKAGE_READY,
                "contract_sha256": contract_sha,
                "package_validator_sha256": "a" * 64,
                "source_order": ["a", "b", "c"],
                "source_frame_indices": source_frames,
                "truth_labels_path": str(truth_path.resolve()),
                "truth_labels_sha256": "c" * 64,
                "truth_label_count": 75,
                "unknown_to_safe_violation_count": 0,
                "prediction_inputs_path": str(
                    (root / "prediction_inputs.jsonl").resolve()
                ),
                "prediction_inputs_sha256": "d" * 64,
                "authorization": {
                    "fresh_prediction_authorized": True,
                    "truth_join_authorized_before_predictions_frozen": False,
                },
            }
            self._write_json(package_path, package)
            authorization = {
                "schema": target.PREDICTION_AUTHORIZATION_SCHEMA,
                "terminal": target.PREDICTION_AUTHORIZED,
                "contract_sha256": contract_sha,
                "package_validator_sha256": "a" * 64,
                "prediction_inputs_path": package[
                    "prediction_inputs_path"
                ],
                "prediction_inputs_sha256": "d" * 64,
                "prediction_input_count": 75,
                "source_order": ["a", "b", "c"],
                "source_frame_indices": source_frames,
                "authorization": {
                    "fresh_prediction_authorized": True,
                    "truth_join_authorized_before_predictions_frozen": False,
                    "source_replacement_or_package_rematerialization": False,
                },
            }
            self._write_json(authorization_path, authorization)
            checkpoint_receipts = [
                {
                    "seed": item["seed"],
                    "arm": item["arm"],
                    "checkpoint_sha256": item["sha256"],
                }
                for item in checkpoints
            ]
            completion = {
                "schema": target.COMPLETION_SCHEMA,
                "terminal": target.PREDICTIONS_READY,
                "contract_sha256": contract_sha,
                "prediction_authorization_sha256": target._sha256(
                    authorization_path
                ),
                "prediction_inputs_path": package[
                    "prediction_inputs_path"
                ],
                "prediction_inputs_sha256": "d" * 64,
                "predictor_sha256": "b" * 64,
                "predictions_sha256": target._sha256(predictions_path),
                "checkpoint_receipts": checkpoint_receipts,
                "all_outputs_finite": True,
                "truth_files_opened": False,
                "teacher_files_opened": False,
                "prediction_count": 450,
                "ordered_prediction_key_sha256": hashlib.sha256(
                    b""
                ).hexdigest(),
                "truth_join_authorized": True,
                "second_prediction_run_authorized": False,
            }
            self._write_json(completion_path, completion)
            truth_read_count = 0

            def read_truth_once(path: Path):
                nonlocal truth_read_count
                truth_read_count += 1
                self.assertEqual(path, truth_path)
                self.assertTrue(
                    (output_root / "truth_join_receipt.json").is_file()
                )
                return [], "c" * 64

            with (
                mock.patch.object(target, "_canonical_paths"),
                mock.patch.object(
                    target, "_load_jsonl", return_value=[]
                ),
                mock.patch.object(
                    target,
                    "_load_jsonl_once_with_sha256",
                    side_effect=read_truth_once,
                ),
                mock.patch.object(
                    target,
                    "_validate_exact_sets",
                    return_value=({}, {}),
                ),
                mock.patch.object(
                    target,
                    "_run_metrics",
                    side_effect=lambda checkpoint, *_: {
                        "seed": checkpoint["seed"],
                        "arm": checkpoint["arm"],
                    },
                ),
                mock.patch.object(
                    target,
                    "_decision",
                    return_value=({}, {"frozen": True}, SUPPORTED),
                ),
            ):
                report = target.evaluate(
                    contract_path,
                    package_path,
                    truth_path,
                    completion_path,
                    predictions_path,
                    output_root,
                )
            self.assertEqual(report["terminal"], SUPPORTED)
            self.assertEqual(truth_read_count, 1)
            with (
                mock.patch.object(target, "_canonical_paths"),
                self.assertRaisesRegex(
                    FileExistsError, "already consumed"
                ),
            ):
                target.evaluate(
                    contract_path,
                    package_path,
                    truth_path,
                    completion_path,
                    predictions_path,
                    output_root,
                )


if __name__ == "__main__":
    unittest.main()
