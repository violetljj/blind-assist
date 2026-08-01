#!/usr/bin/env python3
"""Validate the complete frozen F0.1 SANPO nine-checkpoint training set."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch

import train_stage_c_f0_1_student as training_module
from train_stage_c_f0_1_student import (
    ARMS,
    CONTRACT_SCHEMA,
    CONTRACT_SHA256,
    CORPUS_VALIDATION_SHA256,
    PRETRAINED_SHA256,
    READY,
    SCHEMA as RUN_SCHEMA,
    SEEDS,
    STUDENT_SAMPLES_SHA256,
    TemporalStudent,
    _contract_parent_hashes,
    _parameter_count,
)
from verify_sanpo_pose_geometry_authority import _load_json
from run_geometry_teacher_canary import _sha256


SCHEMA = "blindassist_hftf_stage_c_f0_1_student_training_validation"
SUCCESS = "F0_1_SANPO_NINE_CHECKPOINTS_FROZEN"
FAILURE = "F0_1_SANPO_STUDENT_TRAINING_NOT_EVALUABLE"
EXPECTED_PARAMETER_COUNT = 1_022_448
EXPECTED_HELDOUT_FIREWALL = {
    "heldout_teacher_target_loaded": False,
    "heldout_rgb_loaded": False,
    "heldout_student_output_computed": False,
    "heldout_used_for_checkpoint_or_threshold": False,
}


def _expected_runs() -> list[tuple[int, str]]:
    return [(seed, arm) for seed in SEEDS for arm in ARMS]


def _validate_implementation_path(implementation_path: Path) -> None:
    imported_implementation_path = Path(training_module.__file__).resolve()
    if implementation_path.resolve() != imported_implementation_path:
        raise ValueError(
            "Implementation receipt path does not match imported trainer"
        )


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite scalar at {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")
        return
    raise TypeError(f"Unsupported validation value at {path}: {type(value)}")


def _selected_epoch(history: list[dict[str, Any]]) -> tuple[int, float]:
    if len(history) != 30:
        raise ValueError("Every frozen run must contain 30 epochs")
    best_epoch = -1
    best_f1 = -1.0
    for expected_epoch, entry in enumerate(history, start=1):
        if entry.get("epoch") != expected_epoch:
            raise ValueError("Training history epochs must be exactly 1..30")
        f1 = float(entry["dev"]["risk_micro"]["f1"])
        if f1 > best_f1:
            best_epoch = expected_epoch
            best_f1 = f1
    return best_epoch, best_f1


def _assert_tensor_tree_finite(value: Any, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite checkpoint tensor at {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_tensor_tree_finite(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_tensor_tree_finite(child, f"{path}[{index}]")
        return
    _assert_finite(value, path)


def _model_parameter_groups(
    model: TemporalStudent,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    return encoder_parameters, head_parameters


def _validate_model_state(
    model: TemporalStudent, model_state: dict[str, Any]
) -> None:
    expected_state = model.state_dict()
    if not isinstance(model_state, dict) or set(model_state) != set(
        expected_state
    ):
        raise ValueError("Model state key set mismatch")
    for key, expected in expected_state.items():
        actual = model_state[key]
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
        ):
            raise ValueError(f"Model state shape/dtype mismatch: {key}")


def _validate_optimizer_state(
    model: TemporalStudent,
    optimizer_state: dict[str, Any],
    selected_epoch: int,
) -> None:
    if not isinstance(optimizer_state, dict):
        raise ValueError("Optimizer state must be an object")
    encoder_parameters, head_parameters = _model_parameter_groups(model)
    expected_optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 3e-5},
            {"params": head_parameters, "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    expected_groups = expected_optimizer.state_dict()["param_groups"]
    actual_groups = optimizer_state.get("param_groups")
    actual_state = optimizer_state.get("state")
    if (
        set(optimizer_state) != {"state", "param_groups"}
        or not isinstance(actual_groups, list)
        or len(actual_groups) != 2
        or not isinstance(actual_state, dict)
    ):
        raise ValueError("Optimizer state must contain two groups and state")
    expected_parameters = [encoder_parameters, head_parameters]
    flattened_ids: list[int] = []
    flattened_parameters: list[torch.nn.Parameter] = []
    for index, (actual, expected, parameters) in enumerate(
        zip(actual_groups, expected_groups, expected_parameters, strict=True)
    ):
        if set(actual) != set(expected):
            raise ValueError(f"Optimizer group {index} key set mismatch")
        actual_metadata = {
            key: value for key, value in actual.items() if key != "params"
        }
        expected_metadata = {
            key: value for key, value in expected.items() if key != "params"
        }
        if (
            actual_metadata != expected_metadata
            or not isinstance(actual["params"], list)
            or len(actual["params"]) != len(parameters)
            or any(
                isinstance(parameter_id, bool)
                or not isinstance(parameter_id, int)
                for parameter_id in actual["params"]
            )
        ):
            raise ValueError(f"Optimizer group {index} contract mismatch")
        flattened_ids.extend(actual["params"])
        flattened_parameters.extend(parameters)
    if (
        len(flattened_ids) != len(set(flattened_ids))
        or set(actual_state) != set(flattened_ids)
        or len(actual_state) != len(flattened_parameters)
    ):
        raise ValueError("Optimizer state does not cover every model parameter")
    expected_step = selected_epoch * math.ceil(90 / 8)
    for parameter_id, parameter in zip(
        flattened_ids, flattened_parameters, strict=True
    ):
        state = actual_state[parameter_id]
        if not isinstance(state, dict) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError(f"AdamW state fields mismatch: {parameter_id}")
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or float(step) != expected_step
            or not isinstance(state["exp_avg"], torch.Tensor)
            or not isinstance(state["exp_avg_sq"], torch.Tensor)
            or state["exp_avg"].shape != parameter.shape
            or state["exp_avg_sq"].shape != parameter.shape
            or state["exp_avg"].dtype != parameter.dtype
            or state["exp_avg_sq"].dtype != parameter.dtype
            or step.dtype != torch.float32
        ):
            raise ValueError(f"AdamW state shape/step mismatch: {parameter_id}")
    _assert_tensor_tree_finite(optimizer_state, "checkpoint.optimizer")
    try:
        expected_optimizer.load_state_dict(optimizer_state)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Optimizer state cannot be loaded strictly") from error


def _validate_run(
    run_root: Path,
    seed: int,
    arm: str,
    contract_sha256: str,
    corpus_validation_sha256: str,
    implementation_sha256: str,
    parent_hashes: dict[str, str],
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    if not run_root.is_dir():
        raise ValueError(f"Missing frozen run directory: {run_root}")
    if {path.name for path in run_root.iterdir()} != {
        "checkpoint.pt",
        "training_report.json",
    }:
        raise ValueError(f"Frozen run directory has unexpected files: {run_root}")
    report_path = run_root / "training_report.json"
    checkpoint_path = run_root / "checkpoint.pt"
    report = _load_json(report_path)
    _assert_finite(report, f"report.{seed}.{arm}")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if (
        report.get("schema") != RUN_SCHEMA
        or report.get("terminal") != READY
        or report.get("arm") != arm
        or report.get("seed") != seed
        or report.get("contract_sha256") != contract_sha256
        or report.get("corpus_validation_sha256")
        != corpus_validation_sha256
        or report.get("student_samples_sha256")
        != STUDENT_SAMPLES_SHA256
        or report.get("pretrained_checkpoint_sha256")
        != PRETRAINED_SHA256
        or report.get("implementation_sha256")
        != implementation_sha256
        or report.get("frozen_parent_sha256") != parent_hashes
        or report.get("checkpoint_sha256") != checkpoint_sha256
        or report.get("checkpoint_file") != "checkpoint.pt"
        or report.get("epochs_completed") != 30
        or report.get("train_record_count") != 90
        or report.get("dev_record_count") != 39
        or report.get("heldout_record_count") != 0
        or report.get("all_losses_gradients_and_parameters_finite") is not True
        or report.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or report.get("research_mainline_changed") is not False
        or report.get("default_app_changed") is not False
    ):
        raise ValueError(f"Frozen training report mismatch: seed={seed} arm={arm}")
    expected_image_count = 193 if arm == "HIST_FUTURE" else 129
    if report.get("unique_input_image_count") != expected_image_count:
        raise ValueError(f"Frozen input-image count mismatch: seed={seed} arm={arm}")
    heldout = report.get("heldout_firewall", {})
    if heldout != EXPECTED_HELDOUT_FIREWALL:
        raise ValueError(f"Heldout firewall is not closed: seed={seed} arm={arm}")
    positive_weights = report.get("positive_weights_by_height")
    if (
        not isinstance(positive_weights, dict)
        or set(positive_weights) != {"body", "head"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.25 <= float(value) <= 20.0
            for value in positive_weights.values()
        )
    ):
        raise ValueError(f"Risk positive-weight gate failed: seed={seed} arm={arm}")
    runtime = report.get("runtime", {})
    if (
        runtime.get("torch_version") != runtime_contract.get("torch_version")
        or runtime.get("torchvision_version")
        != runtime_contract.get("torchvision_version")
        or runtime.get("device") != runtime_contract.get("device")
        or runtime.get("float32_no_amp") is not True
        or runtime.get("deterministic_algorithms") is not True
        or not runtime.get("cuda_device_name")
    ):
        raise ValueError(f"Frozen runtime mismatch: seed={seed} arm={arm}")
    history = report.get("history")
    if not isinstance(history, list):
        raise ValueError("Training history must be a list")
    _assert_finite(history, f"report.{seed}.{arm}.history")
    selected_epoch, selected_f1 = _selected_epoch(history)
    if (
        report.get("selected_epoch") != selected_epoch
        or report.get("selected_dev_metrics")
        != history[selected_epoch - 1]["dev"]
        or float(report["selected_dev_metrics"]["risk_micro"]["f1"])
        != selected_f1
    ):
        raise ValueError(f"Checkpoint selection mismatch: seed={seed} arm={arm}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    _assert_tensor_tree_finite(checkpoint, f"checkpoint.{seed}.{arm}")
    if (
        checkpoint.get("schema")
        != "blindassist_hftf_f0_1_student_checkpoint"
        or checkpoint.get("arm") != arm
        or checkpoint.get("seed") != seed
        or checkpoint.get("selected_epoch") != selected_epoch
        or checkpoint.get("selected_dev_metrics")
        != report.get("selected_dev_metrics")
        or checkpoint.get("selected_epoch_metrics")
        != history[selected_epoch - 1]
        or checkpoint.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or checkpoint.get("unique_input_image_count") != expected_image_count
        or checkpoint.get("contract_sha256") != contract_sha256
        or checkpoint.get("corpus_validation_sha256")
        != corpus_validation_sha256
        or checkpoint.get("student_samples_sha256")
        != STUDENT_SAMPLES_SHA256
        or checkpoint.get("pretrained_checkpoint_sha256")
        != PRETRAINED_SHA256
        or checkpoint.get("implementation_sha256")
        != implementation_sha256
        or checkpoint.get("frozen_parent_sha256") != parent_hashes
        or checkpoint.get("positive_weights_by_height")
        != report.get("positive_weights_by_height")
    ):
        raise ValueError(f"Frozen checkpoint metadata mismatch: seed={seed} arm={arm}")
    model = TemporalStudent(pretrained_path=None)
    _validate_model_state(model, checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if _parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"Checkpoint architecture mismatch: seed={seed} arm={arm}")
    _validate_optimizer_state(
        model, checkpoint["optimizer_state_dict"], selected_epoch
    )
    return {
        "seed": seed,
        "arm": arm,
        "report_sha256": _sha256(report_path),
        "checkpoint_sha256": checkpoint_sha256,
        "parameter_count": report["parameter_count"],
        "epochs_completed": report["epochs_completed"],
        "selected_epoch": selected_epoch,
        "selected_dev_f1": selected_f1,
        "selected_dev_recall": report["selected_dev_metrics"]["risk_micro"][
            "recall"
        ],
        "selected_dev_false_positive_rate": report[
            "selected_dev_metrics"
        ]["risk_micro"]["false_positive_rate"],
        "positive_weights_by_height": positive_weights,
    }


def validate(
    contract_path: Path,
    corpus_validation_path: Path,
    implementation_path: Path,
    training_root: Path,
) -> dict[str, Any]:
    contract_sha256 = _sha256(contract_path)
    corpus_validation_sha256 = _sha256(corpus_validation_path)
    implementation_sha256 = _sha256(implementation_path)
    _validate_implementation_path(implementation_path)
    if contract_sha256 != CONTRACT_SHA256:
        raise ValueError("Frozen training execution contract hash mismatch")
    if corpus_validation_sha256 != CORPUS_VALIDATION_SHA256:
        raise ValueError("Frozen train/dev corpus validation hash mismatch")
    contract = _load_json(contract_path)
    corpus_validation = _load_json(corpus_validation_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status")
        != "FROZEN_BEFORE_FIRST_F0_1_STUDENT_OPTIMIZATION_STEP"
        or corpus_validation.get("terminal")
        != "F0_1_SANPO_TRAIN_DEV_CORPUS_VALIDATED"
    ):
        raise ValueError("Training validation parent authority mismatch")
    parent_hashes = _contract_parent_hashes(contract)
    expected_seed_dirs = {f"seed-{seed}" for seed in SEEDS}
    if (
        not training_root.is_dir()
        or {path.name for path in training_root.iterdir()}
        != expected_seed_dirs
    ):
        raise ValueError("Training root must contain exactly three seed directories")
    for seed in SEEDS:
        seed_root = training_root / f"seed-{seed}"
        if {path.name for path in seed_root.iterdir()} != set(ARMS):
            raise ValueError(f"Seed directory arm set mismatch: {seed_root}")
    runs = [
        _validate_run(
            training_root / f"seed-{seed}" / arm,
            seed,
            arm,
            contract_sha256,
            corpus_validation_sha256,
            implementation_sha256,
            parent_hashes,
            contract["runtime_contract"],
        )
        for seed, arm in _expected_runs()
    ]
    checkpoint_hashes = [run["checkpoint_sha256"] for run in runs]
    parameter_counts = {run["parameter_count"] for run in runs}
    if len(set(checkpoint_hashes)) != 9 or parameter_counts != {
        EXPECTED_PARAMETER_COUNT
    }:
        raise ValueError("Nine-run checkpoint hash or parameter-count gate failed")
    current_weights = {
        tuple(sorted(run["positive_weights_by_height"].items()))
        for run in runs
        if run["arm"] == "SF_CURRENT"
    }
    future_weights = {
        tuple(sorted(run["positive_weights_by_height"].items()))
        for run in runs
        if run["arm"] in {"SF_FUTURE", "HIST_FUTURE"}
    }
    if len(current_weights) != 1 or len(future_weights) != 1:
        raise ValueError("Train-only class weights drifted across frozen runs")
    dev_by_seed = {
        str(seed): {
            run["arm"]: run["selected_dev_f1"]
            for run in runs
            if run["seed"] == seed
        }
        for seed in SEEDS
    }
    temporal_minus_single_frame = {
        str(seed): (
            dev_by_seed[str(seed)]["HIST_FUTURE"]
            - dev_by_seed[str(seed)]["SF_FUTURE"]
        )
        for seed in SEEDS
    }
    return {
        "schema": SCHEMA,
        "terminal": SUCCESS,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "TRAIN_DEV_CHECKPOINT_FREEZE_ONLY",
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": contract_sha256,
        "corpus_validation_path": str(corpus_validation_path.resolve()),
        "corpus_validation_sha256": corpus_validation_sha256,
        "implementation_path": str(implementation_path.resolve()),
        "implementation_sha256": implementation_sha256,
        "training_root": str(training_root.resolve()),
        "execution_order": [
            {"seed": seed, "arm": arm} for seed, arm in _expected_runs()
        ],
        "run_count": len(runs),
        "runs": runs,
        "gate_checks": {
            "all_nine_expected_runs_present": True,
            "all_runs_complete_30_epochs": True,
            "all_losses_metrics_model_and_optimizer_tensors_finite": True,
            "all_parameter_counts_equal": True,
            "all_checkpoint_hashes_recorded_and_unique": True,
            "all_selected_epochs_recomputed_from_frozen_dev_metric": True,
            "all_parent_and_implementation_hashes_match": True,
            "heldout_firewall_closed_for_all_runs": True,
            "optimizer_state_complete_loadable_and_finite": True,
            "risk_class_weights_finite_bounded_and_consistent": True,
        },
        "dev_checkpoint_selection_diagnostic_only": {
            "selected_f1_by_seed_and_arm": dev_by_seed,
            "hist_future_minus_sf_future_by_seed": (
                temporal_minus_single_frame
            ),
            "all_three_dev_deltas_positive": all(
                delta > 0.0 for delta in temporal_minus_single_frame.values()
            ),
            "effect_evidence": False,
            "used_to_change_threshold_source_augmentation_or_architecture": False,
        },
        "authorization": {
            "heldout_execution_contract_may_be_frozen": True,
            "heldout_target_materialization_authorized": False,
            "heldout_student_inference_authorized": False,
            "mainline_promotion_authorized": False,
        },
        "claim_boundary": {
            "dev_metrics_are_not_student_effect_evidence": True,
            "geometry_teacher_is_not_human_or_safety_truth": True,
            "research_mainline_changed": False,
            "default_app_changed": False,
            "android_execution_authorized": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
    }


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--corpus-validation", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root: Path | None = None
    try:
        output_root = _require_artifacts_output(args.output_root)
        if output_root.exists():
            raise FileExistsError(
                f"Refusing to overwrite validation output: {output_root}"
            )
        report = validate(
            args.contract.resolve(),
            args.corpus_validation.resolve(),
            args.implementation.resolve(),
            args.training_root.resolve(),
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        partial_root = Path(
            tempfile.mkdtemp(
                prefix=f"{output_root.name}.partial-",
                dir=output_root.parent,
            )
        )
        report_path = partial_root / "validation.json"
        with report_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        partial_root.replace(output_root)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "run_count": report["run_count"],
                    "validation_sha256": _sha256(
                        output_root / "validation.json"
                    ),
                    "output_root": str(output_root),
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
        print(
            json.dumps(
                {
                    "terminal": FAILURE,
                    "ok": False,
                    "error": str(error),
                    "output_root": (
                        str(output_root) if output_root is not None else None
                    ),
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
