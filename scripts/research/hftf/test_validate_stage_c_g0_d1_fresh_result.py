from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_stage_c_g0_d1_fresh_result as validator  # noqa: E402


class FreshResultValidatorTest(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _matrix(function) -> list[list[list[float]]]:
        return [
            [
                [function(height, row, column) for column in range(6)]
                for row in range(6)
            ]
            for height in range(2)
        ]

    def _fixture(self, root: Path) -> dict[str, Path]:
        paths = {
            "contract": root
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_"
            "EXECUTION_CONTRACT_D1_2026-08-01.json",
            "package": root
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-validation-20260801/"
            "validation.json",
            "prediction_authorization": root
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-validation-20260801/"
            "prediction_authorization.json",
            "truth": root
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-20260801/truth_labels.jsonl",
            "completion": root
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-predictions-20260801/completion.json",
            "predictions": root
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-predictions-20260801/predictions.jsonl",
            "join": root
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-effect-20260801/"
            "truth_join_receipt.json",
            "result": root
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-effect-20260801/effect_result.json",
            "output": root
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-effect-validation-20260801",
        }
        evaluator_path = root / validator.EVALUATOR_PATH
        evaluator_path.parent.mkdir(parents=True, exist_ok=True)
        evaluator_path.write_text("# frozen evaluator\n", encoding="utf-8")
        predictor_path = root / validator.PREDICTOR_PATH
        predictor_path.write_text("# frozen predictor\n", encoding="utf-8")
        package_validator_path = root / validator.PACKAGE_VALIDATOR_PATH
        package_validator_path.write_text(
            "# frozen package validator\n", encoding="utf-8"
        )
        checkpoints = [
            {
                "seed": seed,
                "arm": arm,
                "sha256": f"{index + 1:064x}",
            }
            for index, (seed, arm) in enumerate(
                (seed, arm)
                for seed in validator.SEEDS
                for arm in validator.ARMS
            )
        ]
        contract = {
            "schema": validator.CONTRACT_SCHEMA,
            "status": validator.CONTRACT_STATUS,
            "implementations": {
                "fresh_evaluator": {
                    "path": validator.EVALUATOR_PATH,
                    "sha256": validator._sha256(evaluator_path),
                    "execution_authorized": True,
                },
                "fresh_predictor": {
                    "path": validator.PREDICTOR_PATH,
                    "sha256": validator._sha256(predictor_path),
                    "execution_authorized": True,
                },
                "fresh_package_validator": {
                    "path": validator.PACKAGE_VALIDATOR_PATH,
                    "sha256": validator._sha256(
                        package_validator_path
                    ),
                    "execution_authorized": True,
                },
                validator.IMPLEMENTATION_KEY: {
                    "path": validator.IMPLEMENTATION_PATH,
                    "sha256": validator._sha256(
                        Path(validator.__file__).resolve()
                    ),
                    "execution_authorized": True,
                },
            },
            "checkpoint_contract": {"checkpoints": checkpoints},
        }
        self._write_json(paths["contract"], contract)
        source_order = ["fresh-a", "fresh-b", "fresh-c"]
        source_frames = {
            source: list(range(index * 100, index * 100 + 25))
            for index, source in enumerate(source_order)
        }
        package = {
            "schema": validator.PACKAGE_SCHEMA,
            "terminal": validator.PACKAGE_READY,
            "contract_sha256": validator._sha256(paths["contract"]),
            "package_validator_sha256": validator._sha256(
                package_validator_path
            ),
            "source_order": source_order,
            "source_frame_indices": source_frames,
            "unknown_to_safe_violation_count": 0,
        }
        truth: list[dict] = []
        for source in source_order:
            for frame in source_frames[source]:
                sample_id = f"{source}-{frame}"
                known = self._matrix(
                    lambda height, row, column: int(
                        (height, row, column) != (0, 0, 0)
                    )
                )
                risk = self._matrix(
                    lambda height, row, column: (
                        None
                        if (height, row, column) == (0, 0, 0)
                        else int((height + row + column) % 2 == 0)
                    )
                )
                clearance = self._matrix(
                    lambda height, row, column: (
                        None
                        if (height, row, column) == (0, 0, 0)
                        else (
                            -0.1
                            if (height + row + column) % 2 == 0
                            else 0.1
                        )
                    )
                )
                truth.append(
                    {
                        "schema": validator.TRUTH_SCHEMA,
                        "sample_id": sample_id,
                        "session_id": source,
                        "source_frame_index": frame,
                        "manifest_id": f"manifest-{sample_id}",
                        "labels": {
                            "known_target": known,
                            "risk_target_nullable": risk,
                            "clearance_target_m_nullable": clearance,
                        },
                    }
                )
        self._write_jsonl(paths["truth"], truth)
        package.update(
            {
                "truth_labels_path": str(paths["truth"].resolve()),
                "truth_labels_sha256": validator._sha256(paths["truth"]),
                "truth_label_count": 75,
                "prediction_inputs_path": str(
                    (
                        paths["truth"].parent
                        / "prediction_inputs.jsonl"
                    ).resolve()
                ),
                "prediction_inputs_sha256": "1" * 64,
                "prediction_input_count": 75,
            }
        )
        self._write_json(paths["package"], package)
        prediction_authorization = {
            "schema": validator.PREDICTION_AUTHORIZATION_SCHEMA,
            "terminal": validator.PREDICTION_AUTHORIZED,
            "contract_sha256": validator._sha256(paths["contract"]),
            "package_validator_sha256": validator._sha256(
                package_validator_path
            ),
            "prediction_inputs_path": package["prediction_inputs_path"],
            "prediction_inputs_sha256": package[
                "prediction_inputs_sha256"
            ],
            "prediction_input_count": 75,
            "source_order": source_order,
            "source_frame_indices": source_frames,
            "authorization": {
                "fresh_prediction_authorized": True,
                "truth_join_authorized_before_predictions_frozen": False,
                "source_replacement_or_package_rematerialization": False,
            },
        }
        self._write_json(
            paths["prediction_authorization"],
            prediction_authorization,
        )
        predictions: list[dict] = []
        for checkpoint in checkpoints:
            for truth_record in truth:
                labels = truth_record["labels"]
                known = labels["known_target"]
                risk = labels["risk_target_nullable"]
                clearance = labels["clearance_target_m_nullable"]
                direct = checkpoint["arm"] == validator.ARMS[0]
                predictions.append(
                    {
                        "schema": validator.PREDICTION_SCHEMA,
                        "prediction_index": len(predictions),
                        "seed": checkpoint["seed"],
                        "arm": checkpoint["arm"],
                        "checkpoint_sha256": checkpoint["sha256"],
                        "sample_id": truth_record["sample_id"],
                        "session_id": truth_record["session_id"],
                        "source_frame_index": truth_record[
                            "source_frame_index"
                        ],
                        "manifest_id": truth_record["manifest_id"],
                        "raw_task_output": self._matrix(
                            lambda height, row, column: (
                                0.0
                                if direct
                                or clearance[height][row][column] is None
                                else clearance[height][row][column]
                            )
                        ),
                        "risk_probability": self._matrix(
                            lambda height, row, column: (
                                0.5
                                if direct
                                else 0.0
                                if risk[height][row][column] is None
                                else float(risk[height][row][column])
                            )
                        ),
                        "known_probability": self._matrix(
                            lambda height, row, column: float(
                                known[height][row][column]
                            )
                        ),
                    }
                )
        self._write_jsonl(paths["predictions"], predictions)
        normalized_checkpoints = [
            {
                "seed": item["seed"],
                "arm": item["arm"],
                "checkpoint_sha256": item["sha256"],
            }
            for item in checkpoints
        ]
        prediction_receipt_path = (
            paths["completion"].parent / "execution_receipt.json"
        )
        prediction_receipt = {
            "schema": validator.PREDICTION_RECEIPT_SCHEMA,
            "status": "STARTED_BEFORE_FIRST_FRESH_FORWARD",
            "contract_sha256": validator._sha256(paths["contract"]),
            "prediction_authorization_sha256": validator._sha256(
                paths["prediction_authorization"]
            ),
            "prediction_inputs_sha256": "1" * 64,
            "training_validation_sha256": "2" * 64,
            "predictor_sha256": validator._sha256(predictor_path),
            "checkpoint_receipts": normalized_checkpoints,
            "truth_files_opened": False,
            "teacher_files_opened": False,
        }
        self._write_json(prediction_receipt_path, prediction_receipt)
        completion = {
            "schema": validator.COMPLETION_SCHEMA,
            "terminal": validator.PREDICTIONS_READY,
            "contract_sha256": validator._sha256(paths["contract"]),
            "prediction_authorization_sha256": validator._sha256(
                paths["prediction_authorization"]
            ),
            "prediction_inputs_sha256": "1" * 64,
            "training_validation_sha256": "2" * 64,
            "predictor_sha256": validator._sha256(predictor_path),
            "execution_receipt_sha256": validator._sha256(
                prediction_receipt_path
            ),
            "predictions_path": str(paths["predictions"].resolve()),
            "predictions_sha256": validator._sha256(paths["predictions"]),
            "prediction_count": 450,
            "ordered_prediction_key_sha256": (
                validator._ordered_prediction_key_sha256(predictions)
            ),
            "checkpoint_receipts": normalized_checkpoints,
            "raw_task_output_shape": [2, 6, 6],
            "risk_probability_shape": [2, 6, 6],
            "known_probability_shape": [2, 6, 6],
            "all_outputs_finite": True,
            "truth_files_opened": False,
            "teacher_files_opened": False,
            "truth_join_authorized": True,
            "second_prediction_run_authorized": False,
        }
        self._write_json(paths["completion"], completion)
        join = {
            "schema": validator.JOIN_SCHEMA,
            "status": (
                "FROZEN_PREDICTIONS_GLOBALLY_CONSUMED_BEFORE_TRUTH_OPEN"
            ),
            "execution_contract_sha256": validator._sha256(
                paths["contract"]
            ),
            "package_validation_sha256": validator._sha256(
                paths["package"]
            ),
            "completion_sha256": validator._sha256(paths["completion"]),
            "predictions_sha256": validator._sha256(paths["predictions"]),
            "expected_truth_sha256": validator._sha256(paths["truth"]),
            "truth_join_exactly_once": True,
            "second_model_forward_authorized": False,
            "source_replacement_authorized": False,
        }
        self._write_json(paths["join"], join)
        truth_order, by_key = validator._exact_records(
            contract, package, truth, predictions
        )
        runs = [
            validator._recompute_run(
                checkpoint, source_order, truth_order, by_key
            )
            for checkpoint in checkpoints
        ]
        aggregates, gates, terminal = validator._recompute_decision(
            runs, source_order, 0
        )
        result = {
            "schema": validator.RESULT_SCHEMA,
            "terminal": terminal,
            "workflow_profile": "FORMAL_ONE_SHOT_FRESH_EVALUATION",
            "claim_ceiling": (
                "FRESH_CURRENT_SYNTHETIC_PROXY_LEARNABILITY_ONLY"
            ),
            "parents": {
                "execution_contract_sha256": validator._sha256(
                    paths["contract"]
                ),
                "package_validation_sha256": validator._sha256(
                    paths["package"]
                ),
                "completion_sha256": validator._sha256(
                    paths["completion"]
                ),
                "predictions_sha256": validator._sha256(
                    paths["predictions"]
                ),
                "truth_sha256": validator._sha256(paths["truth"]),
                "truth_join_receipt_sha256": validator._sha256(
                    paths["join"]
                ),
            },
            "run_metrics": runs,
            "aggregates": aggregates,
            "gates": gates,
            "all_gates_pass": True,
            "fresh_firewall": {
                "prediction_forward_count": 450,
                "truth_join_count": 1,
                "second_model_forward_executed": False,
                "checkpoint_substitution_executed": False,
                "threshold_change_executed": False,
                "source_replacement_executed": False,
                "reserved_heldout_opened": False,
            },
            "authorization": {
                "causal_transport_contract_may_be_frozen": True,
                "same_cohort_rescue_authorized": False,
                "reserved_official_test_authorized": False,
                "future_or_temporal_experiment_authorized": False,
                "mainline_promotion_authorized": False,
            },
        }
        self._write_json(paths["result"], result)
        return paths

    def test_full_independent_recomputation_accepts_exact_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                validator, "_repository_root", return_value=root
            ):
                paths = self._fixture(root)
                report = validator.validate(
                    paths["contract"],
                    paths["package"],
                    paths["truth"],
                    paths["completion"],
                    paths["predictions"],
                    paths["join"],
                    paths["result"],
                    paths["output"],
                )
            self.assertEqual(report["terminal"], validator.VALIDATED)
            self.assertEqual(report["effect_terminal"], validator.SUPPORTED)
            self.assertTrue(
                all(
                    value
                    for key, value in report["checks"].items()
                    if key != "second_model_forward_performed"
                )
            )
            self.assertFalse(
                report["checks"]["second_model_forward_performed"]
            )

    def test_prediction_drift_not_hidden_by_hash_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                validator, "_repository_root", return_value=root
            ):
                paths = self._fixture(root)
                predictions = validator._load_jsonl(paths["predictions"])
                predictions[75]["risk_probability"][0][0][2] = 0.0
                self._write_jsonl(paths["predictions"], predictions)
                completion = validator._load_json(paths["completion"])
                completion["predictions_sha256"] = validator._sha256(
                    paths["predictions"]
                )
                self._write_json(paths["completion"], completion)
                join = validator._load_json(paths["join"])
                join["predictions_sha256"] = validator._sha256(
                    paths["predictions"]
                )
                join["completion_sha256"] = validator._sha256(
                    paths["completion"]
                )
                self._write_json(paths["join"], join)
                result = validator._load_json(paths["result"])
                result["parents"]["predictions_sha256"] = validator._sha256(
                    paths["predictions"]
                )
                result["parents"]["completion_sha256"] = validator._sha256(
                    paths["completion"]
                )
                result["parents"][
                    "truth_join_receipt_sha256"
                ] = validator._sha256(paths["join"])
                self._write_json(paths["result"], result)
                with self.assertRaisesRegex(
                    ValueError, "risk derivation|recomputation mismatch"
                ):
                    validator.validate(
                        paths["contract"],
                        paths["package"],
                        paths["truth"],
                        paths["completion"],
                        paths["predictions"],
                        paths["join"],
                        paths["result"],
                        paths["output"],
                    )

    def test_mae_gate_uses_worst_seed_not_median_seed(self) -> None:
        runs: list[dict] = []
        for seed, mae in zip(validator.SEEDS, (0.05, 0.05, 0.11)):
            for arm, f1 in (
                (validator.ARMS[0], 0.6),
                (validator.ARMS[1], 0.7),
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
                        for height in validator.HEIGHTS
                    },
                    "risk_by_source": {
                        source: {"f1": f1}
                        for source in ("a", "b", "c")
                    },
                }
                if arm == validator.ARMS[1]:
                    run["clearance_source_height_macro_mae_m"] = {
                        "overall": mae,
                        "risk": 0.05,
                        "safe": 0.05,
                        "near": mae,
                    }
                    run[
                        "raw_prediction_out_of_target_range_fraction"
                    ] = 0.0
                runs.append(run)
        aggregates, gates, terminal = validator._recompute_decision(
            runs, ["a", "b", "c"], 0
        )
        self.assertEqual(
            aggregates["max_seed_clearance_source_height_macro_mae_m"][
                "overall"
            ],
            0.11,
        )
        self.assertFalse(gates["overall_clearance_mae"])
        self.assertFalse(gates["near_boundary_clearance_mae"])
        self.assertEqual(terminal, validator.NOT_SUPPORTED)

    def test_validator_does_not_import_fresh_evaluator(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from evaluate_stage_c_g0_d1_fresh", source)
        self.assertNotIn("import evaluate_stage_c_g0_d1_fresh", source)

    def test_atomic_publication_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            validator._atomic_json(path, {"terminal": "first"})
            with self.assertRaises(FileExistsError):
                validator._atomic_json(path, {"terminal": "second"})
            self.assertEqual(
                validator._load_json(path), {"terminal": "first"}
            )


if __name__ == "__main__":
    unittest.main()
