#!/usr/bin/env python3
"""Train one frozen HFTF G0-D1 current student arm, seed, and phase."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import numbers
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from run_geometry_teacher_canary import _sha256
from train_stage_c_f0_1_student import (
    TemporalStudent,
    _parameter_count,
    _resize_image,
    _sample_augmentation,
    _seed_everything,
    _transform_resized_image,
)
from verify_sanpo_pose_geometry_authority import _load_json


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_current_clearance_"
    "learnability_execution_contract_d1"
)
CONTRACT_STATUS = (
    "FROZEN_BEFORE_D1_DEVELOPMENT_CORPUS_OR_STUDENT_OUTCOME"
)
CORPUS_VALIDATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_development_corpus_validation"
)
CORPUS_VALIDATED = "G0_D1_DEVELOPMENT_CORPUS_VALIDATED"
SCHEMA = "blindassist_hftf_stage_c_g0_d1_current_student_training"
READY = "G0_D1_ARM_SEED_PHASE_CHECKPOINT_FROZEN"
NOT_EVALUABLE = "G0_D1_ARM_SEED_PHASE_TRAINING_NOT_EVALUABLE"
ARMS = ("DIRECT_RISK_CURRENT", "SIGNED_CLEARANCE_CURRENT")
PHASES = ("phase-a", "phase-b")
SEEDS = (17, 29, 43)
HEIGHTS = ("body", "head")
PRETRAINED_SHA256 = (
    "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
)
STUDENT_KEYS = {
    "sample_id",
    "session_id",
    "role",
    "source_frame_index",
    "manifest_id",
    "current_rgb",
    "labels",
}
LABEL_KEYS = {
    "known_target",
    "risk_target_nullable",
    "clearance_target_m_nullable",
}
FROZEN_RUNTIME = {
    "python_wrapper": "E:/codex-tools/bin/blindassist-python.cmd",
    "torch_version": "2.11.0+cu128",
    "torchvision_version": "0.26.0+cu128",
    "device": "cuda",
    "precision": "float32_no_amp",
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "dataloader_workers": 0,
}
CORPUS_VALIDATION_CHECKS = {
    "exact_file_set",
    "exact_six_train_three_model_selection_sources",
    "exact_25_frames_per_source",
    "student_teacher_receipts_one_to_one_and_ordered",
    "student_exact_schema_and_current_rgb_hashes",
    "risk_equals_clearance_strictly_below_zero",
    "unknown_targets_are_null_and_never_safe",
    "source_height_targets_nondegenerate",
    "fresh_and_reserved_sources_excluded",
    "teacher_receipts_not_student_authorized",
    "authoritative_manifest_teacher_and_label_rederived",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("D1 student sample JSONL must contain objects")
        records.append(value)
    return records


def _decode_targets(
    labels: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    known_values = np.asarray(labels["known_target"], dtype=object)
    risk_values = np.asarray(labels["risk_target_nullable"], dtype=object)
    clearance_values = np.asarray(
        labels["clearance_target_m_nullable"], dtype=object
    )
    if (
        known_values.shape != (2, 6, 6)
        or risk_values.shape != (2, 6, 6)
        or clearance_values.shape != (2, 6, 6)
    ):
        raise ValueError("D1 target shape must be 2x6x6")
    known = np.zeros((2, 6, 6), dtype=np.float32)
    risk = np.zeros((2, 6, 6), dtype=np.float32)
    clearance = np.zeros((2, 6, 6), dtype=np.float32)
    for index in np.ndindex(known_values.shape):
        known_value = known_values[index]
        if (
            isinstance(known_value, bool)
            or not isinstance(known_value, (int, float))
            or float(known_value) not in (0.0, 1.0)
        ):
            raise ValueError("D1 known target must be exact numeric 0 or 1")
        is_known = float(known_value) == 1.0
        known[index] = float(is_known)
        risk_value = risk_values[index]
        clearance_value = clearance_values[index]
        if not is_known:
            if risk_value is not None or clearance_value is not None:
                raise ValueError("D1 UNKNOWN targets must remain null")
            continue
        if (
            isinstance(risk_value, bool)
            or not isinstance(risk_value, (int, float))
            or float(risk_value) not in (0.0, 1.0)
        ):
            raise ValueError("D1 known risk target must be exact 0 or 1")
        if (
            isinstance(clearance_value, bool)
            or not isinstance(clearance_value, (int, float))
            or not np.isfinite(float(clearance_value))
            or not -0.5 <= float(clearance_value) <= 1.0
        ):
            raise ValueError("D1 known clearance target is invalid")
        risk[index] = float(risk_value)
        clearance[index] = float(clearance_value)
        if bool(risk[index] == 1.0) is not bool(clearance[index] < 0.0):
            raise ValueError("D1 risk and clearance sign disagree")
    return (
        torch.from_numpy(risk),
        torch.from_numpy(clearance),
        torch.from_numpy(known),
    )


def _flip_targets(
    risk: torch.Tensor,
    clearance: torch.Tensor,
    known: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.flip(risk, dims=(1,)),
        torch.flip(clearance, dims=(1,)),
        torch.flip(known, dims=(1,)),
    )


def _image_receipt(record: dict[str, Any]) -> tuple[Path, str]:
    receipt = record["current_rgb"]
    path = Path(str(receipt["path"])).resolve()
    expected = str(receipt["sha256"])
    return path, expected


class CurrentDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            str,
        ]
    ]
):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        train: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.train_mode = train
        self.seed = seed
        self.epoch = 0
        self.cache: dict[Path, Image.Image] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _resized(self, path: Path) -> Image.Image:
        if path not in self.cache:
            with Image.open(path) as image:
                self.cache[path] = _resize_image(image.convert("RGB"))
        return self.cache[path].copy()

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
    ]:
        record = self.records[index]
        path, _ = _image_receipt(record)
        augmentation = (
            _sample_augmentation(
                self.seed, self.epoch, str(record["sample_id"])
            )
            if self.train_mode
            else None
        )
        resized = self._resized(path)
        frame = _transform_resized_image(resized, augmentation)
        frames = torch.stack([frame.clone() for _ in range(5)])
        risk, clearance, known = _decode_targets(record["labels"])
        if augmentation is not None and augmentation["horizontal_flip"]:
            risk, clearance, known = _flip_targets(
                risk, clearance, known
            )
        return (
            frames,
            risk,
            clearance,
            known,
            str(record["session_id"]),
        )


def _validate_image_receipts(records: list[dict[str, Any]]) -> int:
    paths: set[Path] = set()
    for record in records:
        path, expected = _image_receipt(record)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError("D1 current RGB receipt mismatch")
        paths.add(path)
    if len(paths) != len(records):
        raise ValueError("D1 current samples must use one unique RGB each")
    return len(paths)


def _validate_source_partition(
    records: list[dict[str, Any]],
    expected_roles: dict[str, str],
    expected_frames: dict[str, list[int]] | None = None,
) -> None:
    if (
        len(expected_roles) != 9
        or sum(role == "train" for role in expected_roles.values()) != 6
        or sum(role == "model_selection" for role in expected_roles.values())
        != 3
    ):
        raise ValueError("D1 validation source-role receipt mismatch")
    counts: dict[tuple[str, str], int] = {}
    frames: dict[str, set[int]] = {}
    group_counts: dict[tuple[str, int, str], int] = {}
    sample_ids: set[str] = set()
    for record in records:
        if (
            set(record) != STUDENT_KEYS
            or not isinstance(record.get("labels"), dict)
            or set(record["labels"]) != LABEL_KEYS
            or not isinstance(record.get("current_rgb"), dict)
            or set(record["current_rgb"]) != {"path", "sha256"}
        ):
            raise ValueError("D1 student exact current-only schema mismatch")
        sample_id = str(record.get("sample_id", ""))
        session_id = str(record.get("session_id", ""))
        role = str(record.get("role", ""))
        frame = record.get("source_frame_index")
        if (
            not sample_id
            or sample_id in sample_ids
            or expected_roles.get(session_id) != role
            or isinstance(frame, bool)
            or not isinstance(frame, int)
            or frame < 0
        ):
            raise ValueError("D1 student source, role, or frame drifted")
        sample_ids.add(sample_id)
        counts[(session_id, role)] = counts.get((session_id, role), 0) + 1
        frames.setdefault(session_id, set()).add(frame)
        risk, clearance, known = _decode_targets(record["labels"])
        for height in range(2):
            known_mask = known[height] >= 0.5
            groups = {
                "overall": known_mask,
                "risk": known_mask & (risk[height] >= 0.5),
                "safe": known_mask & (risk[height] < 0.5),
                "near": known_mask & (clearance[height].abs() <= 0.2),
            }
            for name, mask in groups.items():
                key = (session_id, height, name)
                group_counts[key] = group_counts.get(key, 0) + int(mask.sum())
    frozen_frames = expected_frames or {
        session_id: list(range(25)) for session_id in expected_roles
    }
    if (
        set(frames) != set(expected_roles)
        or set(frozen_frames) != set(expected_roles)
        or any(
            frames[session_id] != set(frozen_frames[session_id])
            for session_id in expected_roles
        )
        or any(
            counts.get((session_id, role)) != 25
            for session_id, role in expected_roles.items()
        )
        or any(
            group_counts.get((session_id, height, group), 0) <= 0
            for session_id in expected_roles
            for height in range(2)
            for group in ("overall", "risk", "safe", "near")
        )
    ):
        raise ValueError("D1 exact nondegenerate 6/3 source partition required")


def _validate_runtime_contract(
    design: dict[str, Any],
) -> None:
    runtime = design.get("runtime_and_model_contract", {}).get("runtime")
    if runtime != FROZEN_RUNTIME:
        raise ValueError("D1 frozen runtime contract drifted")
    if (
        torch.__version__ != FROZEN_RUNTIME["torch_version"]
        or torchvision.__version__
        != FROZEN_RUNTIME["torchvision_version"]
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("D1 frozen CUDA runtime is unavailable or drifted")


def _corpus_checks_pass(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == CORPUS_VALIDATION_CHECKS
        and all(item is True for item in value.values())
    )


def _training_counts(
    records: list[dict[str, Any]],
) -> dict[str, list[int]]:
    positive = [0, 0]
    negative = [0, 0]
    known = [0, 0]
    near = [0, 0]
    for record in records:
        risk, clearance, mask = _decode_targets(record["labels"])
        for height in range(2):
            height_mask = mask[height] >= 0.5
            positive[height] += int(
                ((risk[height] >= 0.5) & height_mask).sum()
            )
            negative[height] += int(
                ((risk[height] < 0.5) & height_mask).sum()
            )
            known[height] += int(height_mask.sum())
            near[height] += int(
                ((clearance[height].abs() <= 0.2) & height_mask).sum()
            )
    if any(value <= 0 for value in (*positive, *negative, *known, *near)):
        raise ValueError("D1 train-only target classes are degenerate")
    return {
        "positive": positive,
        "negative": negative,
        "known": known,
        "near": near,
    }


def _loss_parameters(
    train_records: list[dict[str, Any]],
) -> dict[str, list[float] | list[int]]:
    counts = _training_counts(train_records)
    positive_weight = [
        float(
            np.clip(
                counts["negative"][height]
                / counts["positive"][height],
                0.25,
                20.0,
            )
        )
        for height in range(2)
    ]
    risk_base = [
        counts["known"][height]
        / (2.0 * counts["positive"][height])
        for height in range(2)
    ]
    safe_base = [
        counts["known"][height]
        / (2.0 * counts["negative"][height])
        for height in range(2)
    ]
    weighted_sum = [0.0, 0.0]
    for record in train_records:
        risk, clearance, mask = _decode_targets(record["labels"])
        for height in range(2):
            height_mask = mask[height] >= 0.5
            base = torch.where(
                risk[height] >= 0.5,
                torch.tensor(risk_base[height]),
                torch.tensor(safe_base[height]),
            )
            boundary = torch.where(
                clearance[height].abs() <= 0.2,
                torch.tensor(2.0),
                torch.tensor(1.0),
            )
            weighted_sum[height] += float(
                (base * boundary * height_mask).sum()
            )
    normalization = [
        counts["known"][height] / weighted_sum[height]
        for height in range(2)
    ]
    return {
        **counts,
        "positive_weight": positive_weight,
        "risk_base_weight": risk_base,
        "safe_base_weight": safe_base,
        "clearance_weight_normalization": normalization,
    }


def _losses(
    arm: str,
    task_output: torch.Tensor,
    known_logits: torch.Tensor,
    risk: torch.Tensor,
    clearance: torch.Tensor,
    known: torch.Tensor,
    parameters: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    denominator = known.sum().clamp_min(1.0)
    positive_weight = parameters["positive_weight"]
    known_loss = nnf.binary_cross_entropy_with_logits(
        known_logits, known
    )
    if arm == "DIRECT_RISK_CURRENT":
        raw_task = nnf.binary_cross_entropy_with_logits(
            task_output,
            risk,
            pos_weight=positive_weight,
            reduction="none",
        )
        task_loss = (raw_task * known).sum() / denominator
        zero = task_loss.new_zeros(())
        total = task_loss + 0.25 * known_loss
        return {
            "total": total,
            "task": task_loss,
            "regression": zero,
            "sign": zero,
            "known": known_loss,
        }
    risk_base = parameters["risk_base_weight"]
    safe_base = parameters["safe_base_weight"]
    normalization = parameters["clearance_weight_normalization"]
    weights = torch.where(risk >= 0.5, risk_base, safe_base)
    weights = weights * torch.where(
        clearance.abs() <= 0.2,
        task_output.new_tensor(2.0),
        task_output.new_tensor(1.0),
    )
    weights = weights * normalization
    raw_regression = nnf.smooth_l1_loss(
        task_output, clearance, beta=0.1, reduction="none"
    )
    regression = (raw_regression * weights * known).sum() / denominator
    raw_sign = nnf.binary_cross_entropy_with_logits(
        -task_output / 0.1,
        risk,
        pos_weight=positive_weight,
        reduction="none",
    )
    sign = (raw_sign * known).sum() / denominator
    total = regression + 0.1 * sign + 0.25 * known_loss
    return {
        "total": total,
        "task": regression + 0.1 * sign,
        "regression": regression,
        "sign": sign,
        "known": known_loss,
    }


def _count_metrics(
    prediction: torch.Tensor,
    risk: torch.Tensor,
    known: torch.Tensor,
) -> dict[str, int]:
    mask = known >= 0.5
    truth = risk >= 0.5
    return {
        "tp": int((prediction & truth & mask).sum()),
        "fp": int((prediction & (~truth) & mask).sum()),
        "fn": int(((~prediction) & truth & mask).sum()),
        "tn": int(((~prediction) & (~truth) & mask).sum()),
    }


def _add_counts(
    destination: dict[str, int], source: dict[str, int]
) -> None:
    for key in ("tp", "fp", "fn", "tn"):
        destination[key] += int(source[key])


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    return {
        **counts,
        "f1": 2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _evaluate(
    arm: str,
    model: TemporalStudent,
    loader: DataLoader,
    device: torch.device,
    loss_parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    micro = _empty_counts()
    by_height = {height: _empty_counts() for height in HEIGHTS}
    by_source: dict[str, dict[str, int]] = {}
    clearance_sums = {
        "overall": 0.0,
        "risk": 0.0,
        "safe": 0.0,
        "near": 0.0,
    }
    clearance_denominators = {
        "overall": 0,
        "risk": 0,
        "safe": 0,
        "near": 0,
    }
    clearance_by_source: dict[
        str, dict[str, dict[str, float | int]]
    ] = {}
    out_of_range = 0
    known_count = 0
    loss_sums = {
        "total": 0.0,
        "task": 0.0,
        "regression": 0.0,
        "sign": 0.0,
        "known": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for frames, risk, clearance, known, session_ids in loader:
            frames = frames.to(device, non_blocking=True)
            risk = risk.to(device, non_blocking=True)
            clearance = clearance.to(device, non_blocking=True)
            known = known.to(device, non_blocking=True)
            task_output, known_logits = model(frames)
            losses = _losses(
                arm,
                task_output,
                known_logits,
                risk,
                clearance,
                known,
                loss_parameters,
            )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Non-finite D1 evaluation loss")
            for key in loss_sums:
                loss_sums[key] += float(losses[key])
            prediction = (
                torch.sigmoid(task_output) >= 0.5
                if arm == "DIRECT_RISK_CURRENT"
                else task_output < 0.0
            )
            _add_counts(micro, _count_metrics(prediction, risk, known))
            for height_index, height in enumerate(HEIGHTS):
                _add_counts(
                    by_height[height],
                    _count_metrics(
                        prediction[:, height_index],
                        risk[:, height_index],
                        known[:, height_index],
                    ),
                )
            for batch_index, session_id in enumerate(session_ids):
                source_id = str(session_id)
                counts = by_source.setdefault(source_id, _empty_counts())
                _add_counts(
                    counts,
                    _count_metrics(
                        prediction[batch_index],
                        risk[batch_index],
                        known[batch_index],
                    ),
                )
                if arm == "SIGNED_CLEARANCE_CURRENT":
                    source_clearance = clearance_by_source.setdefault(
                        source_id,
                        {
                            key: {"sum": 0.0, "count": 0}
                            for key in clearance_sums
                        },
                    )
                    source_mask = known[batch_index] >= 0.5
                    source_error = (
                        task_output[batch_index]
                        - clearance[batch_index]
                    ).abs()
                    source_groups = {
                        "overall": source_mask,
                        "risk": source_mask
                        & (risk[batch_index] >= 0.5),
                        "safe": source_mask
                        & (risk[batch_index] < 0.5),
                        "near": source_mask
                        & (clearance[batch_index].abs() <= 0.2),
                    }
                    for key, group in source_groups.items():
                        source_clearance[key]["sum"] = float(
                            source_clearance[key]["sum"]
                        ) + float(source_error[group].sum())
                        source_clearance[key]["count"] = int(
                            source_clearance[key]["count"]
                        ) + int(group.sum())
            if arm == "SIGNED_CLEARANCE_CURRENT":
                mask = known >= 0.5
                absolute_error = (task_output - clearance).abs()
                groups = {
                    "overall": mask,
                    "risk": mask & (risk >= 0.5),
                    "safe": mask & (risk < 0.5),
                    "near": mask & (clearance.abs() <= 0.2),
                }
                for key, group in groups.items():
                    clearance_sums[key] += float(
                        absolute_error[group].sum()
                    )
                    clearance_denominators[key] += int(group.sum())
                out_of_range += int(
                    (((task_output < -0.5) | (task_output > 1.0)) & mask).sum()
                )
                known_count += int(mask.sum())
            batches += 1
    source_metrics = {
        source: _metrics(counts) for source, counts in sorted(by_source.items())
    }
    source_f1_values = [
        float(metrics["f1"]) for metrics in source_metrics.values()
    ]
    result: dict[str, Any] = {
        "loss": {
            key: value / batches for key, value in loss_sums.items()
        },
        "risk_micro": _metrics(micro),
        "risk_by_height": {
            height: _metrics(counts)
            for height, counts in by_height.items()
        },
        "risk_by_source": source_metrics,
        "risk_source_macro_f1": float(np.mean(source_f1_values)),
        "risk_worst_source_f1": min(source_f1_values),
    }
    if arm == "SIGNED_CLEARANCE_CURRENT":
        if any(
            int(values[key]["count"]) <= 0
            for values in clearance_by_source.values()
            for key in clearance_sums
        ) or any(
            clearance_denominators[key] <= 0 for key in clearance_sums
        ):
            raise ValueError(
                "D1 clearance metric group denominator is zero"
            )
        result["clearance_mae_m"] = {
            key: clearance_sums[key] / clearance_denominators[key]
            for key in clearance_sums
        }
        result["raw_prediction_out_of_target_range_fraction"] = (
            out_of_range / known_count if known_count else 1.0
        )
        result["clearance_source_macro_mae_m"] = {
            key: float(
                np.mean(
                    [
                        float(values[key]["sum"])
                        / int(values[key]["count"])
                        for values in clearance_by_source.values()
                    ]
                )
            )
            for key in clearance_sums
        }
    return result


def _selection_key(
    arm: str,
    metrics: dict[str, Any],
    epoch: int,
) -> tuple[float, float, float, float, int]:
    clearance_tie = (
        -float(metrics["clearance_source_macro_mae_m"]["overall"])
        if arm == "SIGNED_CLEARANCE_CURRENT"
        else 0.0
    )
    return (
        float(metrics["risk_source_macro_f1"]),
        float(metrics["risk_worst_source_f1"]),
        float(metrics["risk_micro"]["f1"]),
        clearance_tie,
        -epoch,
    )


def _model_state_sha256(model: TemporalStudent) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_parameters(
    values: dict[str, list[float] | list[int]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "positive_weight": torch.tensor(
            values["positive_weight"],
            dtype=torch.float32,
            device=device,
        ).view(1, 2, 1, 1),
        "risk_base_weight": torch.tensor(
            values["risk_base_weight"],
            dtype=torch.float32,
            device=device,
        ).view(1, 2, 1, 1),
        "safe_base_weight": torch.tensor(
            values["safe_base_weight"],
            dtype=torch.float32,
            device=device,
        ).view(1, 2, 1, 1),
        "clearance_weight_normalization": torch.tensor(
            values["clearance_weight_normalization"],
            dtype=torch.float32,
            device=device,
        ).view(1, 2, 1, 1),
    }


def _resolve_parent(
    owner_path: Path,
    receipt: dict[str, Any],
) -> Path:
    raw = Path(str(receipt.get("path", "")))
    if raw.is_absolute():
        return raw.resolve()
    if raw.parts and raw.parts[0] == "artifacts.local":
        return (Path(__file__).resolve().parents[3] / raw).resolve()
    return (owner_path.parent / raw).resolve()


def _load_bound_parent(
    owner_path: Path,
    owner: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    receipt = owner.get("parents", {}).get(key)
    if not isinstance(receipt, dict):
        raise ValueError(f"Missing D1 parent receipt: {key}")
    path = _resolve_parent(owner_path, receipt)
    if not path.is_file() or _sha256(path) != str(receipt.get("sha256")):
        raise ValueError(f"D1 frozen parent hash mismatch: {key}")
    return path, _load_json(path)


def _expected_source_maps(
    design_path: Path,
    design: dict[str, Any],
) -> tuple[dict[str, str], dict[str, list[int]], set[str]]:
    _, source_plan = _load_bound_parent(
        design_path, design, "g0_source_plan"
    )
    roles = source_plan.get("roles", {})
    development = roles.get("development_reuse", [])
    fresh = roles.get("one_shot_fresh_evaluation", [])
    heldout = roles.get("reserved_fresh_heldout", [])
    if not (
        isinstance(development, list)
        and isinstance(fresh, list)
        and isinstance(heldout, list)
        and len(development) == 9
        and len(fresh) == 3
        and len(heldout) == 3
    ):
        raise ValueError("D1 source-plan role cardinality mismatch")
    expected_roles: dict[str, str] = {}
    expected_frames: dict[str, list[int]] = {}
    for index, source in enumerate(development):
        session_id = str(source.get("session_id", ""))
        role = "train" if index < 6 else "model_selection"
        frames = source.get("selected_source_frames")
        if (
            not session_id
            or source.get("role") != ("train" if index < 6 else "dev")
            or not isinstance(frames, list)
            or len(frames) != 25
            or len(set(frames)) != 25
        ):
            raise ValueError("D1 frozen development source map drifted")
        expected_roles[session_id] = role
        expected_frames[session_id] = frames
    forbidden = {
        str(source.get("session_id", "")) for source in [*fresh, *heldout]
    }
    if (
        len(expected_roles) != 9
        or len(forbidden) != 6
        or set(expected_roles) & forbidden
        or any(
            source.get("media_geometry_teacher_or_student_outcome_open")
            is not False
            for source in [*fresh, *heldout]
        )
    ):
        raise ValueError("D1 fresh/reserved source firewall drifted")
    return expected_roles, expected_frames, forbidden


def _require_canonical_inputs(
    contract_path: Path,
    corpus_validation_path: Path,
    student_samples_path: Path,
    pretrained_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    expected = {
        "contract": (
            repository
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_"
            "EXECUTION_CONTRACT_D1_2026-08-01.json"
        ).resolve(),
        "validation": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-development-corpus-validation-20260801/"
            "validation.json"
        ).resolve(),
        "samples": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-development-corpus-20260801/"
            "student_samples.jsonl"
        ).resolve(),
        "pretrained": Path(
            "C:/Users/26442/.cache/torch/hub/checkpoints/"
            "mobilenet_v3_small-047dcff4.pth"
        ).resolve(),
    }
    actual = {
        "contract": contract_path.resolve(),
        "validation": corpus_validation_path.resolve(),
        "samples": student_samples_path.resolve(),
        "pretrained": pretrained_path.resolve(),
    }
    for key in expected:
        if actual[key] != expected[key]:
            raise ValueError(f"D1 noncanonical {key} input path")


def _validate_inputs(
    contract_path: Path,
    corpus_validation_path: Path,
    student_samples_path: Path,
    pretrained_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _require_canonical_inputs(
        contract_path,
        corpus_validation_path,
        student_samples_path,
        pretrained_path,
    )
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
    ):
        raise ValueError("D1 execution contract identity mismatch")
    design_path, design = _load_bound_parent(
        contract_path, contract, "d1_scientific_design"
    )
    _validate_runtime_contract(design)
    frozen_roles, frozen_frames, forbidden_ids = _expected_source_maps(
        design_path, design
    )
    implementation = contract["implementations"]["current_student_trainer"]
    base_module = contract["implementations"]["f0_1_student_module"]
    corpus_validator = contract["implementations"][
        "development_corpus_validator"
    ]
    if (
        Path(str(implementation["path"])).as_posix()
        != "scripts/research/hftf/train_stage_c_g0_d1_current_student.py"
        or implementation.get("sha256")
        != _sha256(Path(__file__).resolve())
        or implementation.get("execution_authorized") is not True
        or Path(str(base_module["path"])).as_posix()
        != "scripts/research/hftf/train_stage_c_f0_1_student.py"
        or base_module.get("sha256")
        != _sha256(
            Path(__file__).resolve().parent
            / "train_stage_c_f0_1_student.py"
        )
        or Path(str(corpus_validator.get("path", ""))).as_posix()
        != (
            "scripts/research/hftf/"
            "validate_stage_c_g0_d1_development_corpus.py"
        )
        or corpus_validator.get("sha256")
        != _sha256(
            Path(__file__).resolve().parent
            / "validate_stage_c_g0_d1_development_corpus.py"
        )
        or corpus_validator.get("execution_authorized") is not True
    ):
        raise ValueError("D1 trainer implementation receipt mismatch")
    validation = _load_json(corpus_validation_path)
    dataset_spec_receipt = validation.get("parents", {}).get("dataset_spec")
    if not isinstance(dataset_spec_receipt, dict):
        raise ValueError("D1 corpus dataset-spec receipt is missing")
    dataset_spec_path = Path(str(dataset_spec_receipt.get("path", "")))
    teacher_receipt_path = Path(
        str(validation.get("teacher_receipts_path", ""))
    )
    if not dataset_spec_path.is_absolute():
        raise ValueError("D1 corpus dataset-spec path must be absolute")
    dataset_spec = _load_json(dataset_spec_path)
    if (
        validation.get("schema") != CORPUS_VALIDATION_SCHEMA
        or validation.get("terminal") != CORPUS_VALIDATED
        or validation.get("parents", {})
        .get("execution_contract", {})
        .get("sha256")
        != _sha256(contract_path)
        or _sha256(dataset_spec_path)
        != str(dataset_spec_receipt.get("sha256"))
        or validation.get("student_samples_sha256")
        != _sha256(student_samples_path)
        or Path(str(validation.get("student_samples_path", ""))).resolve()
        != student_samples_path.resolve()
        or not teacher_receipt_path.is_file()
        or _sha256(teacher_receipt_path)
        != validation.get("teacher_receipts_sha256")
        or dataset_spec.get("files", {})
        .get("student_samples.jsonl", {})
        .get("sha256")
        != _sha256(student_samples_path)
        or dataset_spec.get("files", {})
        .get("teacher_receipts.jsonl", {})
        .get("sha256")
        != _sha256(teacher_receipt_path)
        or validation.get("record_counts")
        != {"train": 150, "model_selection": 75}
        or validation.get("implementation")
        != {
            "path": corpus_validator["path"],
            "sha256": corpus_validator["sha256"],
        }
        or not _corpus_checks_pass(validation.get("checks"))
        or validation.get("authorization", {}).get(
            "development_training_authorized"
        )
        is not True
        or validation.get("authorization", {}).get(
            "fresh_source_opening_authorized"
        )
        is not False
    ):
        raise ValueError("D1 corpus validation receipt mismatch")
    if _sha256(pretrained_path) != PRETRAINED_SHA256:
        raise ValueError("D1 pretrained checkpoint receipt mismatch")
    records = _load_jsonl(student_samples_path)
    if len(records) != 225:
        raise ValueError("D1 student sample role or identity mismatch")
    expected_roles = validation.get("source_roles")
    expected_frames = validation.get("source_frame_indices")
    if not isinstance(expected_roles, dict) or not isinstance(
        expected_frames, dict
    ):
        raise ValueError("D1 validated source-role/frame map is missing")
    if (
        expected_roles != frozen_roles
        or expected_frames != frozen_frames
        or set(expected_roles) & forbidden_ids
    ):
        raise ValueError("D1 validated source map differs from source plan")
    _validate_source_partition(records, expected_roles, expected_frames)
    return contract, records, validation


def _optimizer(model: TemporalStudent) -> torch.optim.AdamW:
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 3e-5},
            {"params": head_parameters, "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )


def _validate_phase_a_preflight(
    phase_a_report_path: Path,
    *,
    arm: str,
    seed: int,
    contract_sha256: str,
    corpus_validation_sha256: str,
    student_samples_sha256: str,
    pretrained_sha256: str,
    implementation_sha256: str,
) -> tuple[dict[str, Any], int]:
    expected_path = (
        Path(__file__).resolve().parents[3]
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-training-20260801/phase-a"
        / str(seed)
        / arm.lower().replace("_", "-")
        / "training_report.json"
    ).resolve()
    if phase_a_report_path.resolve() != expected_path:
        raise ValueError("D1 Phase A report path is not canonical")
    checkpoint_path = phase_a_report_path.parent / "checkpoint.pt"
    if (
        not phase_a_report_path.parent.is_dir()
        or {path.name for path in phase_a_report_path.parent.iterdir()}
        != {"training_report.json", "checkpoint.pt"}
        or not checkpoint_path.is_file()
    ):
        raise ValueError("D1 Phase A run exact file set mismatch")
    report = _load_json(phase_a_report_path)
    history = report.get("history")
    runtime = report.get("runtime", {})
    if (
        report.get("schema") != SCHEMA
        or report.get("terminal") != READY
        or report.get("phase") != "phase-a"
        or report.get("arm") != arm
        or int(report.get("seed", -1)) != seed
        or report.get("contract_sha256") != contract_sha256
        or report.get("corpus_validation_sha256")
        != corpus_validation_sha256
        or report.get("student_samples_sha256")
        != student_samples_sha256
        or report.get("pretrained_checkpoint_sha256")
        != pretrained_sha256
        or report.get("implementation_sha256")
        != implementation_sha256
        or report.get("phase_a_report_sha256") is not None
        or report.get("fit_record_count") != 150
        or report.get("model_selection_record_count") != 75
        or report.get("epochs_completed") != 30
        or report.get("checkpoint_sha256") != _sha256(checkpoint_path)
        or {
            key: runtime.get(key)
            for key in (
                "torch_version",
                "torchvision_version",
                "device",
                "float32_no_amp",
                "deterministic_algorithms",
            )
        }
        != {
            "torch_version": FROZEN_RUNTIME["torch_version"],
            "torchvision_version": FROZEN_RUNTIME["torchvision_version"],
            "device": "cuda",
            "float32_no_amp": True,
            "deterministic_algorithms": True,
        }
        or report.get("fresh_firewall")
        != {
            "fresh_media_loaded": False,
            "fresh_teacher_target_loaded": False,
            "fresh_student_output_computed": False,
            "fresh_used_for_checkpoint_or_threshold": False,
            "reserved_heldout_opened": False,
        }
        or not isinstance(history, list)
        or len(history) != 30
        or [item.get("epoch") for item in history] != list(range(1, 31))
        or any(item.get("model_selection") is None for item in history)
    ):
        raise ValueError("D1 Phase A report identity or history mismatch")
    expected = max(
        history,
        key=lambda item: _selection_key(
            arm,
            item["model_selection"],
            int(item["epoch"]),
        ),
    )
    selected_epoch = int(report.get("selected_epoch", -1))
    if (
        selected_epoch != int(expected["epoch"])
        or report.get("selected_model_selection_metrics")
        != expected["model_selection"]
    ):
        raise ValueError("D1 Phase A selected epoch was not rederived")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    def finite(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return math.isfinite(float(value))
        return True

    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema")
        != "blindassist_hftf_stage_c_g0_d1_student_checkpoint"
        or checkpoint.get("phase") != "phase-a"
        or checkpoint.get("arm") != arm
        or int(checkpoint.get("seed", -1)) != seed
        or int(checkpoint.get("selected_epoch", -1)) != selected_epoch
        or checkpoint.get("selected_model_selection_metrics")
        != report["selected_model_selection_metrics"]
        or checkpoint.get("contract_sha256") != contract_sha256
        or checkpoint.get("corpus_validation_sha256")
        != corpus_validation_sha256
        or checkpoint.get("student_samples_sha256")
        != student_samples_sha256
        or checkpoint.get("pretrained_checkpoint_sha256")
        != pretrained_sha256
        or checkpoint.get("implementation_sha256")
        != implementation_sha256
        or checkpoint.get("parameter_count") != 1_022_448
        or checkpoint.get("parameter_count") != report.get("parameter_count")
        or checkpoint.get("initial_state_sha256")
        != report.get("initial_state_sha256")
        or checkpoint.get("loss_parameters")
        != report.get("loss_parameters")
        or not finite(checkpoint)
    ):
        raise ValueError("D1 Phase A checkpoint identity mismatch")
    model = TemporalStudent(None)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if _parameter_count(model) != 1_022_448:
        raise ValueError("D1 Phase A checkpoint model structure drifted")
    return report, selected_epoch


def train(
    contract_path: Path,
    corpus_validation_path: Path,
    student_samples_path: Path,
    pretrained_path: Path,
    phase: str,
    arm: str,
    seed: int,
    output_root: Path,
    phase_a_report_path: Path | None,
) -> dict[str, Any]:
    contract, records, validation = _validate_inputs(
        contract_path,
        corpus_validation_path,
        student_samples_path,
        pretrained_path,
    )
    if phase not in PHASES or arm not in ARMS or seed not in SEEDS:
        raise ValueError("D1 phase, arm, or seed is outside frozen set")
    if output_root.exists():
        raise FileExistsError("Refusing to overwrite D1 training run")
    if phase == "phase-a" and phase_a_report_path is not None:
        raise ValueError("Phase A must not receive a Phase A report")
    selected_epoch_from_phase_a: int | None = None
    if phase == "phase-b":
        if phase_a_report_path is None:
            raise ValueError("Phase B requires its exact Phase A report")
        _, selected_epoch_from_phase_a = _validate_phase_a_preflight(
            phase_a_report_path,
            arm=arm,
            seed=seed,
            contract_sha256=_sha256(contract_path),
            corpus_validation_sha256=_sha256(corpus_validation_path),
            student_samples_sha256=_sha256(student_samples_path),
            pretrained_sha256=_sha256(pretrained_path),
            implementation_sha256=_sha256(Path(__file__).resolve()),
        )
    unique_input_image_count = _validate_image_receipts(records)
    train_reference = [
        record for record in records if record["role"] == "train"
    ]
    if phase == "phase-a":
        fit_records = train_reference
        evaluation_records = [
            record
            for record in records
            if record["role"] == "model_selection"
        ]
    else:
        fit_records = records
        evaluation_records = []
    loss_parameter_values = _loss_parameters(train_reference)
    _seed_everything(seed)
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
    ):
        raise RuntimeError("D1 deterministic runtime setup failed")
    device = torch.device("cuda")
    model = TemporalStudent(pretrained_path).to(device)
    torch.nn.init.zeros_(model.head.bias)
    initial_state_sha256 = _model_state_sha256(model)
    parameter_count = _parameter_count(model)
    if parameter_count != 1_022_448:
        raise ValueError("D1 student parameter count drifted")
    optimizer = _optimizer(model)
    loss_parameters = _tensor_parameters(loss_parameter_values, device)
    fit_dataset = CurrentDataset(
        fit_records, train=True, seed=seed
    )
    evaluation_loader = (
        DataLoader(
            CurrentDataset(
                evaluation_records, train=False, seed=seed
            ),
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        if evaluation_records
        else None
    )
    history: list[dict[str, Any]] = []
    selected_epoch = -1
    selected_metrics: dict[str, Any] | None = None
    selected_key: tuple[float, float, float, float, int] | None = None
    selected_model_state: dict[str, Any] | None = None
    selected_optimizer_state: dict[str, Any] | None = None
    for epoch in range(1, 31):
        fit_dataset.set_epoch(epoch)
        generator = torch.Generator()
        generator.manual_seed(seed * 1000 + epoch)
        fit_loader = DataLoader(
            fit_dataset,
            batch_size=8,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
        )
        model.train()
        sums = {
            "total": 0.0,
            "task": 0.0,
            "regression": 0.0,
            "sign": 0.0,
            "known": 0.0,
        }
        batches = 0
        for frames, risk, clearance, known, _ in fit_loader:
            frames = frames.to(device, non_blocking=True)
            risk = risk.to(device, non_blocking=True)
            clearance = clearance.to(device, non_blocking=True)
            known = known.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            task_output, known_logits = model(frames)
            losses = _losses(
                arm,
                task_output,
                known_logits,
                risk,
                clearance,
                known,
                loss_parameters,
            )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Non-finite D1 training loss")
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite D1 gradient norm")
            optimizer.step()
            if any(
                not torch.isfinite(parameter).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("Non-finite D1 model parameter")
            for key in sums:
                sums[key] += float(losses[key].detach())
            batches += 1
        evaluation_metrics = (
            _evaluate(
                arm,
                model,
                evaluation_loader,
                device,
                loss_parameters,
            )
            if evaluation_loader is not None
            else None
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": {
                key: value / batches for key, value in sums.items()
            },
            "model_selection": evaluation_metrics,
        }
        history.append(epoch_result)
        freeze = False
        if phase == "phase-a":
            assert evaluation_metrics is not None
            key = _selection_key(arm, evaluation_metrics, epoch)
            if selected_key is None or key > selected_key:
                selected_key = key
                selected_epoch = epoch
                selected_metrics = copy.deepcopy(evaluation_metrics)
                freeze = True
        elif epoch == selected_epoch_from_phase_a:
            selected_epoch = epoch
            freeze = True
        if freeze:
            selected_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            selected_optimizer_state = copy.deepcopy(
                optimizer.state_dict()
            )
        print(
            json.dumps(
                {
                    "phase": phase,
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "train_total_loss": epoch_result["train_loss"][
                        "total"
                    ],
                    "selected_epoch": selected_epoch,
                    "selection_macro_f1": (
                        evaluation_metrics["risk_source_macro_f1"]
                        if evaluation_metrics is not None
                        else None
                    ),
                }
            ),
            flush=True,
        )
    if (
        len(history) != 30
        or selected_model_state is None
        or selected_optimizer_state is None
        or not 1 <= selected_epoch <= 30
    ):
        raise RuntimeError("D1 frozen checkpoint selection failed")
    checkpoint = {
        "schema": "blindassist_hftf_stage_c_g0_d1_student_checkpoint",
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "phase_a_report_sha256": (
            _sha256(phase_a_report_path)
            if phase_a_report_path is not None
            else None
        ),
        "selected_model_selection_metrics": selected_metrics,
        "parameter_count": parameter_count,
        "initial_state_sha256": initial_state_sha256,
        "model_state_dict": selected_model_state,
        "optimizer_state_dict": selected_optimizer_state,
        "loss_parameters": loss_parameter_values,
        "contract_sha256": _sha256(contract_path),
        "corpus_validation_sha256": _sha256(corpus_validation_path),
        "student_samples_sha256": _sha256(student_samples_path),
        "pretrained_checkpoint_sha256": _sha256(pretrained_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(
            prefix=f"{output_root.name}.partial-",
            dir=output_root.parent,
        )
    )
    checkpoint_path = partial / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": SCHEMA,
        "terminal": READY,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "DEVELOPMENT_CHECKPOINT_ONLY",
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "contract_sha256": _sha256(contract_path),
        "corpus_validation_sha256": _sha256(corpus_validation_path),
        "student_samples_sha256": _sha256(student_samples_path),
        "pretrained_checkpoint_sha256": _sha256(pretrained_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "phase_a_report_sha256": (
            _sha256(phase_a_report_path)
            if phase_a_report_path is not None
            else None
        ),
        "runtime": {
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
            "device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
            "float32_no_amp": True,
            "deterministic_algorithms": True,
        },
        "fit_record_count": len(fit_records),
        "model_selection_record_count": len(evaluation_records),
        "unique_development_input_image_count": unique_input_image_count,
        "parameter_count": parameter_count,
        "initial_state_sha256": initial_state_sha256,
        "loss_parameters": loss_parameter_values,
        "epochs_completed": len(history),
        "selected_epoch": selected_epoch,
        "selected_model_selection_metrics": selected_metrics,
        "checkpoint_file": "checkpoint.pt",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "history": history,
        "fresh_firewall": {
            "fresh_media_loaded": False,
            "fresh_teacher_target_loaded": False,
            "fresh_student_output_computed": False,
            "fresh_used_for_checkpoint_or_threshold": False,
            "reserved_heldout_opened": False,
        },
        "research_mainline_changed": False,
        "default_app_changed": False,
    }
    with (partial / "training_report.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    partial.replace(output_root)
    return report


def _canonical_output(
    path: Path, phase: str, seed: int, arm: str
) -> Path:
    expected = (
        Path(__file__).resolve().parents[3]
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-training-20260801"
        / phase
        / str(seed)
        / arm.lower().replace("_", "-")
    ).resolve()
    if path.resolve() != expected:
        raise ValueError("D1 training output path is not canonical")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--corpus-validation", type=Path, required=True)
    parser.add_argument("--student-samples", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--phase-a-report", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        output_root = _canonical_output(
            args.output_root, args.phase, args.seed, args.arm
        )
        report = train(
            args.contract.resolve(),
            args.corpus_validation.resolve(),
            args.student_samples.resolve(),
            args.pretrained_checkpoint.resolve(),
            args.phase,
            args.arm,
            args.seed,
            output_root,
            (
                args.phase_a_report.resolve()
                if args.phase_a_report is not None
                else None
            ),
        )
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "phase": report["phase"],
                    "arm": report["arm"],
                    "seed": report["seed"],
                    "selected_epoch": report["selected_epoch"],
                    "checkpoint_sha256": report["checkpoint_sha256"],
                }
            )
        )
        return 0
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
