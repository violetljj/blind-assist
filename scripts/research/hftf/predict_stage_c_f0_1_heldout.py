#!/usr/bin/env python3
"""Run the one-shot truth-blind F0.1 heldout predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from audit_stage_c_f0_1_teacher_opportunity import _canonical_bytes
from materialize_stage_c_f0_1_heldout_package import (
    CONTRACT_SCHEMA,
    EXPECTED_CONTRACT_STATUS,
    _canonical_artifact_path,
    _implementation_receipt,
)
from run_geometry_teacher_canary import _sha256
from train_stage_c_f0_1_student import (
    ARMS,
    SEEDS,
    TemporalStudent,
    _arm_target,
    _resize_image,
    _transform_resized_image,
    _validate_arm_history_images,
)
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl
from validate_stage_c_f0_1_heldout_package import (
    _expected_sample_rows,
    _validate_probability_free_input,
)


SCHEMA = "blindassist_hftf_stage_c_f0_1_heldout_prediction"
RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_heldout_prediction_execution_receipt"
)
COMPLETION_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_heldout_prediction_completion"
)
READY = "F0_1_SANPO_HELDOUT_PREDICTIONS_FROZEN"
NOT_EVALUABLE = (
    "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_STUDENT_CANARY_NOT_EVALUABLE"
)
PACKAGE_VALIDATED = "F0_1_SANPO_HELDOUT_PACKAGE_VALIDATED"
PACKAGE_VALIDATION_KEYS = {
    "schema",
    "terminal",
    "contract_sha256",
    "package_validator_sha256",
    "source_lock_sha256",
    "teacher_opportunity_sha256",
    "package_manifest_sha256",
    "package_root",
    "files",
    "checks",
    "authorization",
}


def _seed_inference(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HeldoutInferenceDataset(
    Dataset[tuple[torch.Tensor, int]]
):
    def __init__(self, records: list[dict[str, Any]], arm: str) -> None:
        self.records = records
        _, self.repeat_anchor = _arm_target(arm)
        self._cache: dict[str, tuple[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _image(self, item: dict[str, Any]) -> Any:
        from PIL import Image

        path = str(item["image_path"])
        digest = str(item["image_sha256"])
        cached = self._cache.get(path)
        if cached is None:
            if _sha256(Path(path)) != digest:
                raise ValueError(f"Heldout input RGB hash mismatch: {path}")
            with Image.open(path) as image:
                resized = _resize_image(image)
            cached = (digest, resized)
            self._cache[path] = cached
        elif cached[0] != digest:
            raise ValueError(f"Heldout input RGB digest conflict: {path}")
        return cached[1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        history = self.records[index]["history_rgb"]
        selected = [history[-1]] * 5 if self.repeat_anchor else history
        if self.repeat_anchor:
            frame = _transform_resized_image(
                self._image(selected[0]), augmentation=None
            )
            frames = [frame] * 5
        else:
            frames = [
                _transform_resized_image(
                    self._image(item), augmentation=None
                )
                for item in selected
            ]
        return torch.stack(frames), index


def _matrix(value: torch.Tensor) -> list[list[list[float]]]:
    array = value.detach().cpu().numpy()
    if array.shape != (2, 6, 6) or not np.isfinite(array).all():
        raise ValueError("Heldout probability shape/finite mismatch")
    if (array < 0.0).any() or (array > 1.0).any():
        raise ValueError("Heldout probability outside [0,1]")
    return array.astype(np.float64).tolist()


def predict(
    contract_path: Path,
    package_validation_path: Path,
    inference_inputs_path: Path,
    source_lock_path: Path,
    opportunity_path: Path,
    datasets_root: Path,
    checkpoints_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != EXPECTED_CONTRACT_STATUS
    ):
        raise ValueError("Frozen heldout contract identity mismatch")
    _implementation_receipt(
        contract, "heldout_predictor", Path(__file__).resolve()
    )
    _canonical_artifact_path(
        contract,
        "heldout_package_validation_root",
        package_validation_path.parent,
    )
    _canonical_artifact_path(
        contract, "heldout_predictions_root", output_root
    )
    _canonical_artifact_path(
        contract, "frozen_checkpoints_root", checkpoints_root
    )
    if package_validation_path.name != "validation.json":
        raise ValueError("Heldout package validation filename mismatch")
    if inference_inputs_path.name != "inference_inputs.jsonl":
        raise ValueError("Heldout inference input filename mismatch")
    validation = _load_json(package_validation_path)
    expected_validation_checks = {
        "exact_source_anchor_and_sample_order": True,
        "inference_schema_and_history_hashes_exact": True,
        "truth_schema_shape_null_mask_exact": True,
        "receipt_schema_causal_binding_exact": True,
        "truth_reaggregates_to_frozen_reference_opportunity": True,
        "student_output_computed": False,
    }
    if (
        set(validation) != PACKAGE_VALIDATION_KEYS
        or validation.get("schema")
        != "blindassist_hftf_stage_c_f0_1_heldout_package_validation"
        or validation.get("terminal") != PACKAGE_VALIDATED
        or validation.get("contract_sha256") != _sha256(contract_path)
        or validation.get("authorization", {}).get(
            "one_shot_prediction_authorized"
        )
        is not True
        or validation.get("authorization", {}).get(
            "truth_join_authorized_before_predictions_frozen"
        )
        is not False
        or validation.get("files", {})
        .get("inference_inputs.jsonl", {})
        .get("sha256")
        != _sha256(inference_inputs_path)
        or validation.get("package_validator_sha256")
        != contract["implementations"]["heldout_package_validator"][
            "sha256"
        ]
        or validation.get("source_lock_sha256")
        != contract["parents"]["source_lock"]["sha256"]
        or validation.get("teacher_opportunity_sha256")
        != contract["parents"]["teacher_opportunity_report"]["sha256"]
        or validation.get("checks") != expected_validation_checks
        or validation.get("authorization")
        != {
            "one_shot_prediction_authorized": True,
            "truth_join_authorized_before_predictions_frozen": False,
            "package_rematerialization_authorized": False,
        }
        or set(validation.get("files", {}))
        != {
            "inference_inputs.jsonl",
            "heldout_truth.jsonl",
            "teacher_receipts.jsonl",
        }
    ):
        raise ValueError("Heldout package validation authority mismatch")
    expected_input_path = (
        Path(str(validation["package_root"])) / "inference_inputs.jsonl"
    ).resolve()
    if inference_inputs_path.resolve() != expected_input_path:
        raise ValueError("Predictor input path is not validated inference input")
    _canonical_artifact_path(
        contract, "heldout_package_root", inference_inputs_path.parent
    )
    if (
        _sha256(source_lock_path)
        != contract["parents"]["source_lock"]["sha256"]
        or _sha256(opportunity_path)
        != contract["parents"]["teacher_opportunity_report"]["sha256"]
    ):
        raise ValueError("Predictor metadata parent hash mismatch")
    source_lock = _load_json(source_lock_path)
    opportunity = _load_json(opportunity_path)
    runtime = contract["inference_runtime_contract"]
    torchvision_version = __import__("torchvision").__version__
    if (
        torch.__version__ != runtime["torch_version"]
        or torchvision_version != runtime["torchvision_version"]
        or runtime["device"] != "cuda"
        or runtime["precision"] != "float32_no_amp"
        or not torch.cuda.is_available()
    ):
        raise ValueError("Frozen heldout prediction runtime mismatch")
    inputs = _load_jsonl(inference_inputs_path)
    if len(inputs) != 39 or len({item["sample_id"] for item in inputs}) != 39:
        raise ValueError("Heldout inference input count/identity mismatch")
    expected_sample_order = [
        f"hftf_f0_1_heldout_{session_id}_{anchor:02d}"
        for session_id in contract["heldout_source_contract"]["source_order"]
        for anchor in range(8, 21)
    ]
    if [item["sample_id"] for item in inputs] != expected_sample_order:
        raise ValueError("Heldout inference sample order mismatch")
    expected_rows = _expected_sample_rows(
        contract, source_lock, opportunity, datasets_root
    )
    for record, expected in zip(inputs, expected_rows, strict=True):
        _validate_probability_free_input(record, expected)
    if [
        (int(item["seed"]), str(item["arm"]))
        for item in contract["checkpoint_contract"]["checkpoints"]
    ] != [(seed, arm) for seed in SEEDS for arm in ARMS]:
        raise ValueError("Frozen heldout checkpoint order mismatch")
    checkpoint_paths: dict[tuple[int, str], Path] = {}
    for checkpoint in contract["checkpoint_contract"]["checkpoints"]:
        seed = int(checkpoint["seed"])
        arm = str(checkpoint["arm"])
        path = checkpoints_root / f"seed-{seed}" / arm / "checkpoint.pt"
        if _sha256(path) != checkpoint["sha256"]:
            raise ValueError(f"Frozen heldout checkpoint hash mismatch: {seed}:{arm}")
        checkpoint_paths[(seed, arm)] = path
    for arm in ARMS:
        _validate_arm_history_images(inputs, arm)
    repository_root = Path(__file__).resolve().parents[3]
    ledger_path = (
        repository_root
        / contract["canonical_artifact_paths"][
            "one_shot_consumption_ledger"
        ]
    ).resolve()
    if ledger_path.exists():
        raise FileExistsError("Heldout one-shot consumption ledger already exists")
    if output_root.exists():
        raise FileExistsError("Refusing to overwrite one-shot predictions")
    output_root.mkdir(parents=True)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "STARTED_BEFORE_FIRST_HELDOUT_FORWARD",
        "contract_sha256": _sha256(contract_path),
        "package_validation_sha256": _sha256(package_validation_path),
        "inference_inputs_sha256": _sha256(inference_inputs_path),
        "predictor_sha256": _sha256(Path(__file__).resolve()),
        "checkpoint_sha256_by_seed_arm": {
            f"{seed}:{arm}": _sha256(path)
            for (seed, arm), path in checkpoint_paths.items()
        },
        "first_forward_consumes_one_shot": True,
        "truth_or_teacher_receipt_opened": False,
    }
    with (output_root / "execution_receipt.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    predictions_path = output_root / "predictions.jsonl"
    count = 0
    consumed = False
    key_hasher = hashlib.sha256()
    try:
        with predictions_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            for checkpoint in contract["checkpoint_contract"][
                "checkpoints"
            ]:
                seed = int(checkpoint["seed"])
                arm = str(checkpoint["arm"])
                checkpoint_sha256 = str(checkpoint["sha256"])
                _seed_inference(seed)
                payload = torch.load(
                    checkpoint_paths[(seed, arm)],
                    map_location="cpu",
                    weights_only=False,
                )
                if (
                    payload.get("seed") != seed
                    or payload.get("arm") != arm
                    or payload.get("implementation_sha256")
                    != contract["parents"]["student_training_implementation"][
                        "sha256"
                    ]
                ):
                    raise ValueError("Heldout checkpoint metadata mismatch")
                model = TemporalStudent(pretrained_path=None)
                model.load_state_dict(payload["model_state_dict"], strict=True)
                model = model.cuda().eval()
                dataset = HeldoutInferenceDataset(inputs, arm)
                loader = DataLoader(
                    dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
                with torch.no_grad():
                    for frames, indices in loader:
                        frames = frames.cuda(non_blocking=True)
                        if not consumed:
                            ledger = {
                                "schema": (
                                    "blindassist_hftf_stage_c_f0_1_"
                                    "heldout_one_shot_consumption"
                                ),
                                "status": (
                                    "CONSUMED_CONSERVATIVELY_"
                                    "IMMEDIATELY_BEFORE_FIRST_FORWARD"
                                ),
                                "contract_sha256": _sha256(contract_path),
                                "package_validation_sha256": _sha256(
                                    package_validation_path
                                ),
                                "inference_inputs_sha256": _sha256(
                                    inference_inputs_path
                                ),
                                "predictor_sha256": _sha256(
                                    Path(__file__).resolve()
                                ),
                                "prediction_output_root": str(output_root),
                                "first_seed": seed,
                                "first_arm": arm,
                                "rerun_authorized": False,
                            }
                            ledger_path.parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            with ledger_path.open(
                                "x", encoding="utf-8", newline="\n"
                            ) as ledger_handle:
                                json.dump(
                                    ledger,
                                    ledger_handle,
                                    indent=2,
                                    ensure_ascii=False,
                                )
                                ledger_handle.write("\n")
                                ledger_handle.flush()
                                os.fsync(ledger_handle.fileno())
                            consumed = True
                        risk_logits, known_logits = model(frames)
                        risk_probabilities = torch.sigmoid(risk_logits)
                        known_probabilities = torch.sigmoid(known_logits)
                        for batch_index, input_index_tensor in enumerate(indices):
                            input_index = int(input_index_tensor)
                            sample = inputs[input_index]
                            key = {
                                "seed": seed,
                                "arm": arm,
                                "checkpoint_sha256": checkpoint_sha256,
                                "sample_id": sample["sample_id"],
                            }
                            record = {
                                "schema": SCHEMA,
                                "prediction_index": count,
                                **key,
                                "session_id": sample["session_id"],
                                "anchor_timeline_index": sample[
                                    "anchor_timeline_index"
                                ],
                                "risk_probability": _matrix(
                                    risk_probabilities[batch_index]
                                ),
                                "known_probability": _matrix(
                                    known_probabilities[batch_index]
                                ),
                            }
                            serialized = _canonical_bytes(record)
                            handle.write(serialized.decode("utf-8") + "\n")
                            key_hasher.update(_canonical_bytes(key) + b"\n")
                            count += 1
                        handle.flush()
                        os.fsync(handle.fileno())
                del model
                torch.cuda.empty_cache()
        if count != 351:
            raise ValueError("Heldout prediction record count mismatch")
        predictions_sha256 = _sha256(predictions_path)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "terminal": READY,
            "contract_sha256": _sha256(contract_path),
            "package_validation_sha256": _sha256(
                package_validation_path
            ),
            "inference_inputs_sha256": _sha256(inference_inputs_path),
            "predictor_sha256": _sha256(Path(__file__).resolve()),
            "execution_receipt_sha256": _sha256(
                output_root / "execution_receipt.json"
            ),
            "predictions_file": "predictions.jsonl",
            "predictions_sha256": predictions_sha256,
            "prediction_record_count": count,
            "ordered_join_key_sha256": key_hasher.hexdigest(),
            "all_probabilities_shape_2x6x6_finite_and_in_unit_interval": True,
            "truth_or_teacher_receipt_opened": False,
            "one_shot_consumed": True,
            "consumption_ledger_sha256": _sha256(ledger_path),
            "second_prediction_run_authorized": False,
            "truth_join_authorized": True,
        }
        with (output_root / "completion.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(completion, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return completion
    except Exception as error:
        failure = {
            "terminal": NOT_EVALUABLE,
            "error_type": type(error).__name__,
            "error": str(error),
            "prediction_records_serialized": count,
            "one_shot_consumed": consumed,
            "rerun_authorized": False,
        }
        try:
            with (output_root / "failure.json").open(
                "x", encoding="utf-8", newline="\n"
            ) as handle:
                json.dump(failure, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            raise RuntimeError(
                f"One-shot heldout prediction failed: {failure}"
            ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--package-validation", type=Path, required=True)
    parser.add_argument("--inference-inputs", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--opportunity", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = predict(
            args.contract.resolve(),
            args.package_validation.resolve(),
            args.inference_inputs.resolve(),
            args.source_lock.resolve(),
            args.opportunity.resolve(),
            args.datasets_root.resolve(),
            args.checkpoints_root.resolve(),
            args.output_root.resolve(),
        )
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "prediction_record_count": report[
                        "prediction_record_count"
                    ],
                    "predictions_sha256": report["predictions_sha256"],
                }
            )
        )
        return 0
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
