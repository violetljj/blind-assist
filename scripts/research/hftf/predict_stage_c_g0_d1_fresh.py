#!/usr/bin/env python3
"""Run the truth-blind HFTF G0-D1 fresh current-frame predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from audit_stage_c_f0_1_teacher_opportunity import _canonical_bytes
from run_geometry_teacher_canary import _sha256
from train_stage_c_f0_1_student import (
    TemporalStudent,
    _parameter_count,
    _resize_image,
    _transform_resized_image,
)
from verify_sanpo_pose_geometry_authority import _load_json


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_current_clearance_"
    "fresh_execution_contract_d1"
)
CONTRACT_STATUS = (
    "FROZEN_BEFORE_D1_FRESH_SOURCE_OPENING_OR_PREDICTION"
)
PREDICTION_AUTHORIZATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_prediction_authorization"
)
PREDICTION_AUTHORIZED = (
    "G0_D1_FRESH_PREDICTION_AUTHORIZATION_READY"
)
PREDICTION_AUTHORIZATION_KEYS = {
    "schema",
    "terminal",
    "contract_sha256",
    "package_validator_sha256",
    "prediction_inputs_path",
    "prediction_inputs_sha256",
    "prediction_input_count",
    "source_order",
    "source_frame_indices",
    "authorization",
}
PREDICTION_AUTHORIZATION_DECISION_KEYS = {
    "fresh_prediction_authorized",
    "truth_join_authorized_before_predictions_frozen",
    "source_replacement_or_package_rematerialization",
}
TRAINING_VALIDATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_training_validation"
)
TRAINING_VALIDATED = "G0_D1_SIX_FINAL_CHECKPOINTS_FROZEN"
CHECKPOINT_SCHEMA = "blindassist_hftf_stage_c_g0_d1_student_checkpoint"
INPUT_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_prediction_input"
PREDICTION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_prediction"
)
EXECUTION_RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_prediction_execution_receipt"
)
COMPLETION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_predictions_frozen"
)
READY = "G0_D1_FRESH_PREDICTIONS_FROZEN"
NOT_EVALUABLE = "G0_D1_FRESH_PREDICTIONS_NOT_EVALUABLE"
DEPENDENCY_RECEIPTS = {
    "teacher_opportunity_module": (
        "scripts/research/hftf/"
        "audit_stage_c_f0_1_teacher_opportunity.py"
    ),
    "geometry_teacher_module": (
        "scripts/research/hftf/run_geometry_teacher_canary.py"
    ),
    "student_module": (
        "scripts/research/hftf/train_stage_c_f0_1_student.py"
    ),
    "fresh_source_authority_verifier": (
        "scripts/research/hftf/"
        "verify_sanpo_pose_geometry_authority.py"
    ),
}
ARMS = ("DIRECT_RISK_CURRENT", "SIGNED_CLEARANCE_CURRENT")
SEEDS = (17, 29, 43)
INPUT_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "source_frame_index",
    "manifest_id",
    "current_rgb",
}
FROZEN_RUNTIME = {
    "torch_version": "2.11.0+cu128",
    "torchvision_version": "0.26.0+cu128",
    "device": "cuda",
    "precision": "float32_no_amp",
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "dataloader_workers": 0,
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"Fresh prediction input line {line_number} is not an object"
            )
        records.append(value)
    return records


def _resolve_receipt(owner_path: Path, receipt: dict[str, Any]) -> Path:
    raw = Path(str(receipt.get("path", "")))
    if raw.is_absolute():
        return raw.resolve()
    if raw.parts and raw.parts[0] == "artifacts.local":
        return (Path(__file__).resolve().parents[3] / raw).resolve()
    return (owner_path.parent / raw).resolve()


def _canonical_path(contract: dict[str, Any], key: str) -> Path:
    value = contract.get("canonical_artifacts", {}).get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing frozen fresh canonical artifact: {key}")
    return (Path(__file__).resolve().parents[3] / value).resolve()


def _require_canonical_paths(
    contract_path: Path,
    prediction_authorization_path: Path,
    prediction_inputs_path: Path,
    training_validation_path: Path,
    output_root: Path,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    expected = {
        "contract": (
            repository
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_"
            "EXECUTION_CONTRACT_D1_2026-08-01.json"
        ).resolve(),
        "prediction_authorization": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-validation-20260801/"
            "prediction_authorization.json"
        ).resolve(),
        "prediction_inputs": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-20260801/"
            "prediction_inputs.jsonl"
        ).resolve(),
        "training_validation": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-training-validation-20260801/"
            "validation.json"
        ).resolve(),
        "output": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-predictions-20260801"
        ).resolve(),
    }
    actual = {
        "contract": contract_path.resolve(),
        "prediction_authorization": prediction_authorization_path.resolve(),
        "prediction_inputs": prediction_inputs_path.resolve(),
        "training_validation": training_validation_path.resolve(),
        "output": output_root.resolve(),
    }
    for key in expected:
        if actual[key] != expected[key]:
            raise ValueError(f"Fresh predictor received noncanonical {key}")


def _implementation_receipt(
    contract: dict[str, Any], key: str, implementation_path: Path
) -> None:
    receipt = contract.get("implementations", {}).get(key)
    if (
        not isinstance(receipt, dict)
        or Path(str(receipt.get("path", ""))).as_posix()
        != (
            "scripts/research/hftf/"
            "predict_stage_c_g0_d1_fresh.py"
        )
        or receipt.get("sha256") != _sha256(implementation_path)
        or receipt.get("execution_authorized") is not True
    ):
        raise ValueError("Fresh predictor implementation receipt mismatch")


def _dependency_receipts(contract: dict[str, Any]) -> None:
    for key, relative in DEPENDENCY_RECEIPTS.items():
        receipt = contract.get("implementations", {}).get(key)
        path = (
            Path(__file__).resolve().parents[3] / relative
        ).resolve()
        if (
            not isinstance(receipt, dict)
            or receipt.get("path") != relative
            or receipt.get("sha256") != _sha256(path)
            or receipt.get("execution_authorized") is not True
        ):
            raise ValueError(
                f"Fresh predictor dependency receipt mismatch: {key}"
            )


def _validate_runtime(contract: dict[str, Any]) -> None:
    runtime = contract.get("runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("Fresh prediction runtime contract missing")
    if {
        key: runtime.get(key) for key in FROZEN_RUNTIME
    } != FROZEN_RUNTIME:
        raise ValueError("Fresh prediction frozen runtime drifted")
    if (
        torch.__version__ != FROZEN_RUNTIME["torch_version"]
        or torchvision.__version__ != FROZEN_RUNTIME["torchvision_version"]
        or not torch.cuda.is_available()
    ):
        raise ValueError("Fresh prediction runtime is unavailable")


def _validate_package_authority(
    contract_path: Path,
    contract: dict[str, Any],
    prediction_authorization_path: Path,
    prediction_inputs_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        prediction_authorization_path.resolve()
        != _canonical_path(contract, "fresh_prediction_authorization")
        or prediction_inputs_path.resolve()
        != _canonical_path(contract, "prediction_inputs")
        or prediction_inputs_path.name != "prediction_inputs.jsonl"
    ):
        raise ValueError("Fresh predictor received a noncanonical package path")
    validation = _load_json(prediction_authorization_path)
    authorization = validation.get("authorization", {})
    if (
        set(validation) != PREDICTION_AUTHORIZATION_KEYS
        or set(authorization) != PREDICTION_AUTHORIZATION_DECISION_KEYS
        or validation.get("schema") != PREDICTION_AUTHORIZATION_SCHEMA
        or validation.get("terminal") != PREDICTION_AUTHORIZED
        or validation.get("contract_sha256") != _sha256(contract_path)
        or validation.get("package_validator_sha256")
        != contract.get("implementations", {})
        .get("fresh_package_validator", {})
        .get("sha256")
        or validation.get("prediction_inputs_path")
        != str(prediction_inputs_path.resolve())
        or validation.get("prediction_inputs_sha256")
        != _sha256(prediction_inputs_path)
        or int(validation.get("prediction_input_count", -1)) != 75
        or authorization.get("fresh_prediction_authorized") is not True
        or authorization.get(
            "truth_join_authorized_before_predictions_frozen"
        )
        is not False
        or authorization.get(
            "source_replacement_or_package_rematerialization"
        )
        is not False
    ):
        raise ValueError("Fresh prediction authorization mismatch")
    records = _load_jsonl(prediction_inputs_path)
    source_order = validation.get("source_order")
    source_frames = validation.get("source_frame_indices")
    if source_order != contract.get("fresh_source_contract", {}).get(
        "source_order"
    ):
        raise ValueError("Fresh authorization source order drifted")
    _validate_inputs(records, source_order, source_frames)
    return validation, records


def _validate_inputs(
    records: list[dict[str, Any]],
    source_order: Any,
    source_frames: Any,
) -> None:
    if (
        not isinstance(source_order, list)
        or len(source_order) != 3
        or len(set(source_order)) != 3
        or not all(isinstance(value, str) and value for value in source_order)
        or not isinstance(source_frames, dict)
        or set(source_frames) != set(source_order)
    ):
        raise ValueError("Fresh frozen source order is invalid")
    expected_pairs: list[tuple[str, int]] = []
    for session_id in source_order:
        frames = source_frames[session_id]
        if (
            not isinstance(frames, list)
            or len(frames) != 25
            or len(set(frames)) != 25
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in frames
            )
        ):
            raise ValueError("Fresh frozen source frames are invalid")
        expected_pairs.extend((session_id, value) for value in frames)
    if (
        len(records) != 75
        or len({str(record.get("sample_id", "")) for record in records})
        != 75
        or [
            (
                str(record.get("session_id", "")),
                record.get("source_frame_index"),
            )
            for record in records
        ]
        != expected_pairs
    ):
        raise ValueError("Fresh prediction input order or identity mismatch")
    for record in records:
        receipt = record.get("current_rgb")
        if (
            set(record) != INPUT_KEYS
            or record.get("schema") != INPUT_SCHEMA
            or not isinstance(record.get("sample_id"), str)
            or not record["sample_id"]
            or not isinstance(record.get("manifest_id"), str)
            or not record["manifest_id"]
            or not isinstance(receipt, dict)
            or set(receipt) != {"path", "sha256"}
            or not isinstance(receipt.get("path"), str)
            or not isinstance(receipt.get("sha256"), str)
            or len(receipt["sha256"]) != 64
        ):
            raise ValueError("Fresh prediction input schema mismatch")


def _training_checkpoints(
    contract_path: Path,
    contract: dict[str, Any],
    training_validation_path: Path,
) -> list[dict[str, Any]]:
    parent = contract.get("parents", {}).get("training_validation")
    if not isinstance(parent, dict):
        raise ValueError("Fresh training-validation parent receipt missing")
    parent_path = _resolve_receipt(contract_path, parent)
    if (
        parent_path != training_validation_path.resolve()
        or not parent_path.is_file()
        or _sha256(parent_path) != str(parent.get("sha256", ""))
    ):
        raise ValueError("Fresh training-validation parent hash mismatch")
    validation = _load_json(training_validation_path)
    checkpoints = validation.get("final_checkpoints")
    expected_order = [(seed, arm) for seed in SEEDS for arm in ARMS]
    if (
        validation.get("schema") != TRAINING_VALIDATION_SCHEMA
        or validation.get("terminal") != TRAINING_VALIDATED
        or int(validation.get("final_checkpoint_count", -1)) != 6
        or not isinstance(checkpoints, list)
        or len(checkpoints) != 6
        or [
            (item.get("seed"), item.get("arm"))
            for item in checkpoints
            if isinstance(item, dict)
        ]
        != expected_order
        or validation.get("fresh_firewall")
        != {
            "fresh_media_loaded": False,
            "fresh_teacher_target_loaded": False,
            "fresh_student_output_computed": False,
            "fresh_used_for_checkpoint_or_threshold": False,
            "reserved_heldout_opened": False,
        }
    ):
        raise ValueError("Fresh training-validation authority mismatch")
    normalized: list[dict[str, Any]] = []
    for receipt, (seed, arm) in zip(
        checkpoints, expected_order, strict=True
    ):
        path = Path(str(receipt.get("checkpoint_path", ""))).resolve()
        digest = str(receipt.get("checkpoint_sha256", ""))
        if (
            not path.is_file()
            or len(digest) != 64
            or _sha256(path) != digest
        ):
            raise ValueError(
                f"Fresh final checkpoint receipt mismatch: {seed}:{arm}"
            )
        normalized.append(
            {
                "seed": seed,
                "arm": arm,
                "checkpoint_path": path,
                "checkpoint_sha256": digest,
            }
        )
    _validate_checkpoint_contract(contract, normalized)
    return normalized


def _validate_checkpoint_contract(
    contract: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> None:
    frozen = contract.get("checkpoint_contract", {}).get("checkpoints")
    if (
        not isinstance(frozen, list)
        or len(frozen) != 6
        or [
            (
                item.get("seed"),
                item.get("arm"),
                item.get("sha256"),
            )
            for item in frozen
            if isinstance(item, dict)
        ]
        != [
            (
                item["seed"],
                item["arm"],
                item["checkpoint_sha256"],
            )
            for item in checkpoints
        ]
    ):
        raise ValueError(
            "Fresh checkpoint contract does not exactly bind validation"
        )


def _finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _load_model(receipt: dict[str, Any]) -> TemporalStudent:
    payload = torch.load(
        receipt["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("phase") != "phase-b"
        or payload.get("seed") != receipt["seed"]
        or payload.get("arm") != receipt["arm"]
        or not _finite(payload)
    ):
        raise ValueError("Fresh checkpoint payload identity or finite mismatch")
    model = TemporalStudent(None)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if _parameter_count(model) != 1_022_448 or not _finite(
        model.state_dict()
    ):
        raise ValueError("Fresh checkpoint strict-load validation failed")
    return model


def _seed_inference(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FreshCurrentDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.cache: dict[Path, tuple[str, Image.Image]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _image(self, record: dict[str, Any]) -> Image.Image:
        receipt = record["current_rgb"]
        path = Path(str(receipt["path"])).resolve()
        digest = str(receipt["sha256"])
        cached = self.cache.get(path)
        if cached is None:
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"Fresh current RGB hash mismatch: {path}")
            with Image.open(path) as image:
                resized = _resize_image(image.convert("RGB"))
            self.cache[path] = (digest, resized)
            cached = self.cache[path]
        elif cached[0] != digest:
            raise ValueError(f"Fresh current RGB digest conflict: {path}")
        return cached[1].copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        frame = _transform_resized_image(
            self._image(self.records[index]), augmentation=None
        )
        return torch.stack([frame.clone() for _ in range(5)]), index


def _matrix(
    value: torch.Tensor, *, require_unit_interval: bool
) -> list[list[list[float]]]:
    array = value.detach().cpu().numpy()
    if array.shape != (2, 6, 6) or not np.isfinite(array).all():
        raise ValueError("Fresh output shape/finite mismatch")
    if require_unit_interval and (
        (array < 0.0).any() or (array > 1.0).any()
    ):
        raise ValueError("Fresh probability outside [0,1]")
    return array.astype(np.float64).tolist()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_jsonl(path: Path, temporary: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    os.link(temporary, path)
    temporary.unlink()


def predict(
    contract_path: Path,
    prediction_authorization_path: Path,
    prediction_inputs_path: Path,
    training_validation_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    _require_canonical_paths(
        contract_path,
        prediction_authorization_path,
        prediction_inputs_path,
        training_validation_path,
        output_root,
    )
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
    ):
        raise ValueError("Frozen fresh execution contract identity mismatch")
    _implementation_receipt(
        contract, "fresh_predictor", Path(__file__).resolve()
    )
    _dependency_receipts(contract)
    if output_root.resolve() != _canonical_path(
        contract, "fresh_predictions_root"
    ):
        raise ValueError("Fresh prediction output root is not canonical")
    if output_root.exists():
        raise FileExistsError("Refusing to overwrite fresh predictions")
    _, records = _validate_package_authority(
        contract_path,
        contract,
        prediction_authorization_path,
        prediction_inputs_path,
    )
    checkpoints = _training_checkpoints(
        contract_path, contract, training_validation_path
    )
    _validate_runtime(contract)
    output_root.mkdir(parents=True)
    execution_receipt = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "status": "STARTED_BEFORE_FIRST_FRESH_FORWARD",
        "contract_sha256": _sha256(contract_path),
        "prediction_authorization_sha256": _sha256(
            prediction_authorization_path
        ),
        "prediction_inputs_sha256": _sha256(prediction_inputs_path),
        "training_validation_sha256": _sha256(training_validation_path),
        "predictor_sha256": _sha256(Path(__file__).resolve()),
        "checkpoint_receipts": [
            {
                "seed": item["seed"],
                "arm": item["arm"],
                "checkpoint_sha256": item["checkpoint_sha256"],
            }
            for item in checkpoints
        ],
        "truth_files_opened": False,
        "teacher_files_opened": False,
        "fresh_firewall": {
            "prediction_inputs_only": True,
            "truth_files_opened": False,
            "teacher_files_opened": False,
            "threshold_or_checkpoint_selection_changed": False,
            "reserved_heldout_opened": False,
        },
    }
    _atomic_json(output_root / "execution_receipt.json", execution_receipt)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".predictions.", suffix=".jsonl.tmp", dir=output_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    prediction_count = 0
    key_hasher = hashlib.sha256()
    try:
        dataset = FreshCurrentDataset(records)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for checkpoint in checkpoints:
                seed = int(checkpoint["seed"])
                arm = str(checkpoint["arm"])
                _seed_inference(seed)
                model = _load_model(checkpoint).cuda().eval()
                loader = DataLoader(
                    dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
                with torch.no_grad():
                    for frames, indices in loader:
                        raw_task, known_logits = model(
                            frames.cuda(non_blocking=True)
                        )
                        if arm == "DIRECT_RISK_CURRENT":
                            risk = torch.sigmoid(raw_task)
                        else:
                            risk = (raw_task < 0.0).to(torch.float32)
                        known = torch.sigmoid(known_logits)
                        for batch_index, input_index_tensor in enumerate(
                            indices
                        ):
                            sample = records[int(input_index_tensor)]
                            key = {
                                "seed": seed,
                                "arm": arm,
                                "checkpoint_sha256": checkpoint[
                                    "checkpoint_sha256"
                                ],
                                "sample_id": sample["sample_id"],
                            }
                            record = {
                                "schema": PREDICTION_SCHEMA,
                                "prediction_index": prediction_count,
                                **key,
                                "session_id": sample["session_id"],
                                "source_frame_index": sample[
                                    "source_frame_index"
                                ],
                                "manifest_id": sample["manifest_id"],
                                "raw_task_output": _matrix(
                                    raw_task[batch_index],
                                    require_unit_interval=False,
                                ),
                                "risk_probability": _matrix(
                                    risk[batch_index],
                                    require_unit_interval=True,
                                ),
                                "known_probability": _matrix(
                                    known[batch_index],
                                    require_unit_interval=True,
                                ),
                            }
                            handle.write(
                                _canonical_bytes(record).decode("utf-8")
                                + "\n"
                            )
                            key_hasher.update(
                                _canonical_bytes(key) + b"\n"
                            )
                            prediction_count += 1
                        handle.flush()
                        os.fsync(handle.fileno())
                del model
                torch.cuda.empty_cache()
        if prediction_count != 450:
            raise ValueError("Fresh prediction count mismatch")
        predictions_path = output_root / "predictions.jsonl"
        _publish_jsonl(predictions_path, temporary)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "terminal": READY,
            "contract_sha256": _sha256(contract_path),
            "prediction_authorization_sha256": _sha256(
                prediction_authorization_path
            ),
            "prediction_inputs_path": str(
                prediction_inputs_path.resolve()
            ),
            "prediction_inputs_sha256": _sha256(
                prediction_inputs_path
            ),
            "training_validation_sha256": _sha256(
                training_validation_path
            ),
            "predictor_sha256": _sha256(Path(__file__).resolve()),
            "execution_receipt_sha256": _sha256(
                output_root / "execution_receipt.json"
            ),
            "predictions_path": str(predictions_path.resolve()),
            "predictions_sha256": _sha256(predictions_path),
            "prediction_count": prediction_count,
            "ordered_prediction_key_sha256": key_hasher.hexdigest(),
            "checkpoint_receipts": [
                {
                    "seed": item["seed"],
                    "arm": item["arm"],
                    "checkpoint_sha256": item["checkpoint_sha256"],
                }
                for item in checkpoints
            ],
            "raw_task_output_shape": [2, 6, 6],
            "risk_probability_shape": [2, 6, 6],
            "known_probability_shape": [2, 6, 6],
            "all_outputs_finite": True,
            "truth_files_opened": False,
            "teacher_files_opened": False,
            "fresh_firewall": {
                "prediction_inputs_only": True,
                "truth_files_opened": False,
                "teacher_files_opened": False,
                "threshold_or_checkpoint_selection_changed": False,
                "reserved_heldout_opened": False,
            },
            "truth_join_authorized": True,
            "second_prediction_run_authorized": False,
        }
        _atomic_json(output_root / "completion.json", completion)
        return completion
    except Exception as error:
        temporary.unlink(missing_ok=True)
        failure = {
            "terminal": NOT_EVALUABLE,
            "error_type": type(error).__name__,
            "error": str(error),
            "prediction_records_serialized": prediction_count,
            "truth_files_opened": False,
            "teacher_files_opened": False,
            "rerun_authorized": False,
        }
        try:
            _atomic_json(output_root / "failure.json", failure)
        finally:
            raise RuntimeError(
                f"Fresh prediction failed: {failure}"
            ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--prediction-authorization", type=Path, required=True
    )
    parser.add_argument("--prediction-inputs", type=Path, required=True)
    parser.add_argument(
        "--training-validation", type=Path, required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = predict(
            args.contract.resolve(),
            args.prediction_authorization.resolve(),
            args.prediction_inputs.resolve(),
            args.training_validation.resolve(),
            args.output_root.resolve(),
        )
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "prediction_count": report["prediction_count"],
                    "predictions_sha256": report[
                        "predictions_sha256"
                    ],
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
