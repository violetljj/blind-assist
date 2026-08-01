from __future__ import annotations

import copy
import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_stage_c_g0_d1_fresh import (
    ARMS,
    INPUT_KEYS,
    INPUT_SCHEMA,
    PREDICTION_AUTHORIZATION_SCHEMA,
    PREDICTION_AUTHORIZED,
    PREDICTION_SCHEMA,
    SEEDS,
    _atomic_json,
    _dependency_receipts,
    _finite,
    _matrix,
    _validate_checkpoint_contract,
    _validate_package_authority,
    _validate_inputs,
    predict,
)


def _records() -> tuple[list[dict], list[str], dict[str, list[int]]]:
    sources = ["source-a", "source-b", "source-c"]
    frames = {source: list(range(25)) for source in sources}
    records = []
    for source in sources:
        for frame in frames[source]:
            records.append(
                {
                    "schema": INPUT_SCHEMA,
                    "sample_id": f"{source}-{frame}",
                    "session_id": source,
                    "source_frame_index": frame,
                    "manifest_id": f"manifest-{source}",
                    "current_rgb": {
                        "path": f"C:/{source}/{frame}.jpg",
                        "sha256": "a" * 64,
                    },
                }
            )
    return records, sources, frames


class StageCG0D1FreshPredictorTest(unittest.TestCase):
    def test_frozen_dependency_receipts_reject_drift(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        contract = json.loads(
            (
                repository
                / "docs/research/hftf/"
                "HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_"
                "EXECUTION_CONTRACT_D1_2026-08-01.json"
            ).read_text(encoding="utf-8")
        )
        _dependency_receipts(contract)
        drifted = copy.deepcopy(contract)
        drifted["implementations"]["student_module"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ValueError, "dependency receipt mismatch"
        ):
            _dependency_receipts(drifted)

    def test_exact_truth_free_input_contract_passes(self) -> None:
        records, sources, frames = _records()
        _validate_inputs(records, sources, frames)
        self.assertEqual(75, len(records))
        self.assertTrue(all(set(record) == INPUT_KEYS for record in records))

    def test_input_contract_rejects_order_duplicate_and_extra_key(self) -> None:
        records, sources, frames = _records()
        swapped = records.copy()
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaisesRegex(ValueError, "order"):
            _validate_inputs(swapped, sources, frames)
        duplicated = records.copy()
        duplicated[-1] = duplicated[-2]
        with self.assertRaisesRegex(ValueError, "order"):
            _validate_inputs(duplicated, sources, frames)
        extra = [dict(record) for record in records]
        extra[0]["labels"] = {}
        with self.assertRaisesRegex(ValueError, "schema"):
            _validate_inputs(extra, sources, frames)

    def test_raw_and_probability_matrix_contracts(self) -> None:
        raw = torch.linspace(-2.0, 2.0, 72).reshape(2, 6, 6)
        self.assertEqual((2, 6, 6), np.asarray(
            _matrix(raw, require_unit_interval=False)
        ).shape)
        direct = torch.sigmoid(raw)
        derived = (raw < 0.0).to(torch.float32)
        direct_values = np.asarray(
            _matrix(direct, require_unit_interval=True)
        )
        derived_values = np.asarray(
            _matrix(derived, require_unit_interval=True)
        )
        self.assertTrue(((direct_values > 0) & (direct_values < 1)).all())
        self.assertEqual({0.0, 1.0}, set(derived_values.ravel()))
        with self.assertRaisesRegex(ValueError, "shape"):
            _matrix(torch.zeros(72), require_unit_interval=False)
        with self.assertRaisesRegex(ValueError, "finite"):
            _matrix(
                torch.full((2, 6, 6), math.nan),
                require_unit_interval=False,
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            _matrix(
                torch.full((2, 6, 6), 1.01),
                require_unit_interval=True,
            )

    def test_finite_checkpoint_guard_recurses(self) -> None:
        self.assertTrue(_finite({"x": torch.zeros(2), "epoch": 3}))
        self.assertFalse(
            _finite({"state": {"weight": torch.tensor([math.inf])}})
        )

    def test_checkpoint_contract_must_exactly_bind_six_receipts(self) -> None:
        receipts = [
            {
                "seed": seed,
                "arm": arm,
                "checkpoint_sha256": str(index) * 64,
            }
            for index, (seed, arm) in enumerate(
                (
                    (seed, arm)
                    for seed in SEEDS
                    for arm in ARMS
                ),
                start=1,
            )
        ]
        contract = {
            "checkpoint_contract": {
                "checkpoints": [
                    {
                        "seed": item["seed"],
                        "arm": item["arm"],
                        "sha256": item["checkpoint_sha256"],
                    }
                    for item in receipts
                ]
            }
        }
        _validate_checkpoint_contract(contract, receipts)
        contract["checkpoint_contract"]["checkpoints"][-1][
            "sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(ValueError, "exactly bind"):
            _validate_checkpoint_contract(contract, receipts)

    def test_atomic_json_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            _atomic_json(path, {"value": 1})
            self.assertEqual({"value": 1}, json.loads(
                path.read_text(encoding="utf-8")
            ))
            with self.assertRaises(FileExistsError):
                _atomic_json(path, {"value": 2})
            self.assertEqual({"value": 1}, json.loads(
                path.read_text(encoding="utf-8")
            ))

    def test_public_prediction_signature_has_no_truth_or_teacher_input(
        self,
    ) -> None:
        parameters = set(inspect.signature(predict).parameters)
        self.assertEqual(
            {
                "contract_path",
                "prediction_authorization_path",
                "prediction_inputs_path",
                "training_validation_path",
                "output_root",
            },
            parameters,
        )
        self.assertNotIn("truth_path", parameters)
        self.assertNotIn("teacher_receipts_path", parameters)

    def test_prediction_authorization_is_truth_free_and_exact(self) -> None:
        records, sources, frames = _records()
        authorization_path = Path("authorization.json").resolve()
        inputs_path = Path("prediction_inputs.jsonl").resolve()
        contract = {
            "implementations": {
                "fresh_package_validator": {
                    "sha256": "b" * 64
                }
            },
            "fresh_source_contract": {"source_order": sources},
        }
        receipt = {
            "schema": PREDICTION_AUTHORIZATION_SCHEMA,
            "terminal": PREDICTION_AUTHORIZED,
            "contract_sha256": "a" * 64,
            "package_validator_sha256": "b" * 64,
            "prediction_inputs_path": str(inputs_path),
            "prediction_inputs_sha256": "c" * 64,
            "prediction_input_count": 75,
            "source_order": sources,
            "source_frame_indices": frames,
            "authorization": {
                "fresh_prediction_authorized": True,
                "truth_join_authorized_before_predictions_frozen": False,
                "source_replacement_or_package_rematerialization": False,
            },
        }
        with (
            mock.patch(
                "predict_stage_c_g0_d1_fresh._canonical_path",
                side_effect=lambda _, key: (
                    authorization_path
                    if key == "fresh_prediction_authorization"
                    else inputs_path
                ),
            ),
            mock.patch(
                "predict_stage_c_g0_d1_fresh._load_json",
                return_value=receipt,
            ),
            mock.patch(
                "predict_stage_c_g0_d1_fresh._load_jsonl",
                return_value=records,
            ),
            mock.patch(
                "predict_stage_c_g0_d1_fresh._sha256",
                side_effect=lambda path: (
                    "a" * 64
                    if path.name.endswith(".json")
                    else "c" * 64
                ),
            ),
        ):
            validated, loaded = _validate_package_authority(
                authorization_path,
                contract,
                authorization_path,
                inputs_path,
            )
            self.assertEqual(validated, receipt)
            self.assertEqual(loaded, records)
            receipt["package_validator_sha256"] = "e" * 64
            with self.assertRaisesRegex(
                ValueError, "authorization mismatch"
            ):
                _validate_package_authority(
                    authorization_path,
                    contract,
                    authorization_path,
                    inputs_path,
                )
            receipt["package_validator_sha256"] = "b" * 64
            receipt["truth_labels_sha256"] = "d" * 64
            with self.assertRaisesRegex(
                ValueError, "authorization mismatch"
            ):
                _validate_package_authority(
                    authorization_path,
                    contract,
                    authorization_path,
                    inputs_path,
                )

    def test_frozen_cartesian_order_and_prediction_schema(self) -> None:
        self.assertEqual(
            [
                (17, "DIRECT_RISK_CURRENT"),
                (17, "SIGNED_CLEARANCE_CURRENT"),
                (29, "DIRECT_RISK_CURRENT"),
                (29, "SIGNED_CLEARANCE_CURRENT"),
                (43, "DIRECT_RISK_CURRENT"),
                (43, "SIGNED_CLEARANCE_CURRENT"),
            ],
            [(seed, arm) for seed in SEEDS for arm in ARMS],
        )
        self.assertEqual(
            "blindassist_hftf_stage_c_g0_d1_fresh_prediction",
            PREDICTION_SCHEMA,
        )
        self.assertEqual(450, len(SEEDS) * len(ARMS) * 75)


if __name__ == "__main__":
    unittest.main()
