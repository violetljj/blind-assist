#!/usr/bin/env python3
"""Validate all frozen HFTF G0-D1 Phase A and Phase B runs."""

from __future__ import annotations

import argparse
import json
import math
import numbers
import tempfile
from pathlib import Path
from typing import Any

import torch

from run_geometry_teacher_canary import _sha256
from train_stage_c_f0_1_student import TemporalStudent, _parameter_count
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
TRAINING_SCHEMA = "blindassist_hftf_stage_c_g0_d1_current_student_training"
TRAINING_READY = "G0_D1_ARM_SEED_PHASE_CHECKPOINT_FROZEN"
CHECKPOINT_SCHEMA = "blindassist_hftf_stage_c_g0_d1_student_checkpoint"
SCHEMA = "blindassist_hftf_stage_c_g0_d1_training_validation"
VALIDATED = "G0_D1_SIX_FINAL_CHECKPOINTS_FROZEN"
NOT_EVALUABLE = "G0_D1_TRAINING_VALIDATION_NOT_EVALUABLE"
ARMS = ("DIRECT_RISK_CURRENT", "SIGNED_CLEARANCE_CURRENT")
PHASES = ("phase-a", "phase-b")
SEEDS = (17, 29, 43)
PRETRAINED_SHA256 = (
    "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
)
EXPECTED_REPORT_RUNTIME = {
    "torch_version": "2.11.0+cu128",
    "torchvision_version": "0.26.0+cu128",
    "device": "cuda",
    "float32_no_amp": True,
    "deterministic_algorithms": True,
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


def _arm_directory(arm: str) -> str:
    return arm.lower().replace("_", "-")


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
        frames = source.get("selected_source_frames")
        if (
            not session_id
            or source.get("role") != ("train" if index < 6 else "dev")
            or not isinstance(frames, list)
            or len(frames) != 25
            or len(set(frames)) != 25
        ):
            raise ValueError("D1 frozen development source map drifted")
        expected_roles[session_id] = (
            "train" if index < 6 else "model_selection"
        )
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
    training_root: Path,
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
        "training": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-training-20260801"
        ).resolve(),
    }
    actual = {
        "contract": contract_path.resolve(),
        "validation": corpus_validation_path.resolve(),
        "samples": student_samples_path.resolve(),
        "pretrained": pretrained_path.resolve(),
        "training": training_root.resolve(),
    }
    for key in expected:
        if actual[key] != expected[key]:
            raise ValueError(f"D1 noncanonical {key} input path")


def _corpus_checks_pass(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == CORPUS_VALIDATION_CHECKS
        and all(item is True for item in value.values())
    )


def _selection_key(
    arm: str,
    metrics: dict[str, Any],
    epoch: int,
) -> tuple[float, float, float, float, int]:
    mae_tie = (
        -float(metrics["clearance_source_macro_mae_m"]["overall"])
        if arm == "SIGNED_CLEARANCE_CURRENT"
        else 0.0
    )
    return (
        float(metrics["risk_source_macro_f1"]),
        float(metrics["risk_worst_source_f1"]),
        float(metrics["risk_micro"]["f1"]),
        mae_tie,
        -epoch,
    )


def _expected_run_roots(root: Path) -> dict[tuple[str, int, str], Path]:
    return {
        (phase, seed, arm): (
            root / phase / str(seed) / _arm_directory(arm)
        ).resolve()
        for phase in PHASES
        for seed in SEEDS
        for arm in ARMS
    }


def _finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _validate_checkpoint(
    checkpoint_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    if _sha256(checkpoint_path) != report["checkpoint_sha256"]:
        raise ValueError("D1 checkpoint hash mismatch")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("phase") != report["phase"]
        or checkpoint.get("arm") != report["arm"]
        or int(checkpoint.get("seed", -1)) != int(report["seed"])
        or int(checkpoint.get("selected_epoch", -1))
        != int(report["selected_epoch"])
        or checkpoint.get("initial_state_sha256")
        != report["initial_state_sha256"]
        or checkpoint.get("loss_parameters") != report["loss_parameters"]
        or checkpoint.get("contract_sha256")
        != report["contract_sha256"]
        or checkpoint.get("corpus_validation_sha256")
        != report["corpus_validation_sha256"]
        or checkpoint.get("student_samples_sha256")
        != report["student_samples_sha256"]
        or checkpoint.get("pretrained_checkpoint_sha256")
        != report["pretrained_checkpoint_sha256"]
        or checkpoint.get("implementation_sha256")
        != report["implementation_sha256"]
        or checkpoint.get("selected_model_selection_metrics")
        != report["selected_model_selection_metrics"]
        or checkpoint.get("phase_a_report_sha256")
        != report["phase_a_report_sha256"]
        or not _finite(checkpoint)
    ):
        raise ValueError("D1 checkpoint identity or finite-state mismatch")
    model = TemporalStudent(None)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if _parameter_count(model) != 1_022_448:
        raise ValueError("D1 checkpoint parameter count mismatch")
    return checkpoint


def _validate_phase_a_history(report: dict[str, Any]) -> None:
    history = report["history"]
    if (
        len(history) != 30
        or [int(item["epoch"]) for item in history] != list(range(1, 31))
        or any(item["model_selection"] is None for item in history)
    ):
        raise ValueError("D1 Phase A history coverage mismatch")
    arm = str(report["arm"])
    expected = max(
        history,
        key=lambda item: _selection_key(
            arm, item["model_selection"], int(item["epoch"])
        ),
    )
    if (
        int(expected["epoch"]) != int(report["selected_epoch"])
        or expected["model_selection"]
        != report["selected_model_selection_metrics"]
    ):
        raise ValueError("D1 Phase A selected epoch was not rederived")


def _validate_phase_b_history(
    report: dict[str, Any],
    phase_a_report: dict[str, Any],
    phase_a_report_path: Path,
) -> None:
    history = report["history"]
    if (
        len(history) != 30
        or [int(item["epoch"]) for item in history] != list(range(1, 31))
        or any(item["model_selection"] is not None for item in history)
        or int(report["selected_epoch"])
        != int(phase_a_report["selected_epoch"])
        or report["selected_model_selection_metrics"] is not None
        or report["phase_a_report_sha256"]
        != _sha256(phase_a_report_path)
    ):
        raise ValueError("D1 Phase B fixed-epoch history mismatch")


def _validate_report_identity(
    report: dict[str, Any],
    *,
    phase: str,
    seed: int,
    arm: str,
    contract_sha256: str,
    corpus_validation_sha256: str,
    student_samples_sha256: str,
    trainer_sha256: str,
) -> None:
    runtime = report.get("runtime", {})
    if (
        report.get("schema") != TRAINING_SCHEMA
        or report.get("terminal") != TRAINING_READY
        or report.get("phase") != phase
        or int(report.get("seed", -1)) != seed
        or report.get("arm") != arm
        or report.get("contract_sha256") != contract_sha256
        or report.get("corpus_validation_sha256")
        != corpus_validation_sha256
        or report.get("student_samples_sha256")
        != student_samples_sha256
        or report.get("pretrained_checkpoint_sha256")
        != PRETRAINED_SHA256
        or report.get("implementation_sha256") != trainer_sha256
        or int(report.get("epochs_completed", -1)) != 30
        or int(report.get("parameter_count", -1)) != 1_022_448
        or int(report.get("unique_development_input_image_count", -1))
        != 225
        or report.get("fresh_firewall")
        != {
            "fresh_media_loaded": False,
            "fresh_teacher_target_loaded": False,
            "fresh_student_output_computed": False,
            "fresh_used_for_checkpoint_or_threshold": False,
            "reserved_heldout_opened": False,
        }
        or {
            key: runtime.get(key) for key in EXPECTED_REPORT_RUNTIME
        }
        != EXPECTED_REPORT_RUNTIME
        or not isinstance(runtime.get("cuda_device_name"), str)
        or not runtime["cuda_device_name"]
        or not _finite(report)
    ):
        raise ValueError("D1 training report identity mismatch")
    expected_counts = (
        (150, 75) if phase == "phase-a" else (225, 0)
    )
    if (
        int(report["fit_record_count"]),
        int(report["model_selection_record_count"]),
    ) != expected_counts:
        raise ValueError("D1 training report role count mismatch")


def validate(
    contract_path: Path,
    corpus_validation_path: Path,
    student_samples_path: Path,
    pretrained_path: Path,
    training_root: Path,
) -> dict[str, Any]:
    _require_canonical_inputs(
        contract_path,
        corpus_validation_path,
        student_samples_path,
        pretrained_path,
        training_root,
    )
    contract = _load_json(contract_path)
    corpus_validation = _load_json(corpus_validation_path)
    contract_sha256 = _sha256(contract_path)
    corpus_validation_sha256 = _sha256(corpus_validation_path)
    student_samples_sha256 = _sha256(student_samples_path)
    design_path, design = _load_bound_parent(
        contract_path, contract, "d1_scientific_design"
    )
    frozen_roles, frozen_frames, forbidden_ids = _expected_source_maps(
        design_path, design
    )
    corpus_validator_receipt = contract["implementations"][
        "development_corpus_validator"
    ]
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
        or corpus_validation.get("schema") != CORPUS_VALIDATION_SCHEMA
        or corpus_validation.get("terminal") != CORPUS_VALIDATED
        or corpus_validation.get("parents", {})
        .get("execution_contract", {})
        .get("sha256")
        != contract_sha256
        or corpus_validation.get("student_samples_sha256")
        != student_samples_sha256
        or Path(
            str(corpus_validation.get("student_samples_path", ""))
        ).resolve()
        != student_samples_path.resolve()
        or corpus_validation.get("record_counts")
        != {"train": 150, "model_selection": 75}
        or corpus_validation.get("source_roles") != frozen_roles
        or corpus_validation.get("source_frame_indices") != frozen_frames
        or set(corpus_validation.get("source_roles", {})) & forbidden_ids
        or corpus_validation.get("implementation")
        != {
            "path": corpus_validator_receipt.get("path"),
            "sha256": corpus_validator_receipt.get("sha256"),
        }
        or not _corpus_checks_pass(corpus_validation.get("checks"))
        or corpus_validation.get("authorization", {}).get(
            "fresh_source_opening_authorized"
        )
        is not False
        or _sha256(pretrained_path) != PRETRAINED_SHA256
    ):
        raise ValueError("D1 training validation parent mismatch")
    validator_receipt = contract["implementations"]["training_validator"]
    trainer_receipt = contract["implementations"]["current_student_trainer"]
    if (
        Path(str(validator_receipt.get("path", ""))).as_posix()
        != "scripts/research/hftf/validate_stage_c_g0_d1_training.py"
        or validator_receipt.get("sha256")
        != _sha256(Path(__file__).resolve())
        or validator_receipt.get("execution_authorized") is not True
        or Path(str(trainer_receipt.get("path", ""))).as_posix()
        != "scripts/research/hftf/train_stage_c_g0_d1_current_student.py"
        or trainer_receipt.get("execution_authorized") is not True
        or Path(
            str(corpus_validator_receipt.get("path", ""))
        ).as_posix()
        != (
            "scripts/research/hftf/"
            "validate_stage_c_g0_d1_development_corpus.py"
        )
        or corpus_validator_receipt.get("sha256")
        != _sha256(
            Path(__file__).resolve().parent
            / "validate_stage_c_g0_d1_development_corpus.py"
        )
    ):
        raise ValueError("D1 training validator receipt mismatch")
    trainer_sha256 = str(trainer_receipt["sha256"])
    expected_roots = _expected_run_roots(training_root)
    if (
        not training_root.is_dir()
        or {path.name for path in training_root.iterdir()}
        != set(PHASES)
        or any(
            {path.name for path in (training_root / phase).iterdir()}
            != {str(seed) for seed in SEEDS}
            for phase in PHASES
        )
        or any(
            {
                path.name
                for path in (training_root / phase / str(seed)).iterdir()
            }
            != {_arm_directory(arm) for arm in ARMS}
            for phase in PHASES
            for seed in SEEDS
        )
    ):
        raise ValueError("D1 training canonical directory tree mismatch")
    reports: dict[tuple[str, int, str], dict[str, Any]] = {}
    checkpoints: dict[tuple[str, int, str], dict[str, Any]] = {}
    for identity, run_root in expected_roots.items():
        phase, seed, arm = identity
        if (
            not run_root.is_dir()
            or {path.name for path in run_root.iterdir()}
            != {"training_report.json", "checkpoint.pt"}
        ):
            raise ValueError("D1 training run directory set mismatch")
        report_path = run_root / "training_report.json"
        report = _load_json(report_path)
        _validate_report_identity(
            report,
            phase=phase,
            seed=seed,
            arm=arm,
            contract_sha256=contract_sha256,
            corpus_validation_sha256=corpus_validation_sha256,
            student_samples_sha256=student_samples_sha256,
            trainer_sha256=trainer_sha256,
        )
        checkpoint = _validate_checkpoint(
            run_root / "checkpoint.pt", report
        )
        reports[identity] = report
        checkpoints[identity] = checkpoint
    for seed in SEEDS:
        initial_hashes = {
            reports[(phase, seed, arm)]["initial_state_sha256"]
            for phase in PHASES
            for arm in ARMS
        }
        if len(initial_hashes) != 1:
            raise ValueError("D1 same-seed initial model arrays drifted")
    if (
        len(
            {
                json.dumps(
                    report["loss_parameters"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for report in reports.values()
            }
        )
        != 1
    ):
        raise ValueError("D1 train-only loss parameters drifted across runs")
    final_receipts: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            phase_a_identity = ("phase-a", seed, arm)
            phase_b_identity = ("phase-b", seed, arm)
            phase_a_report = reports[phase_a_identity]
            phase_b_report = reports[phase_b_identity]
            _validate_phase_a_history(phase_a_report)
            phase_a_path = (
                expected_roots[phase_a_identity]
                / "training_report.json"
            )
            _validate_phase_b_history(
                phase_b_report, phase_a_report, phase_a_path
            )
            if (
                checkpoints[phase_b_identity]["phase_a_report_sha256"]
                != _sha256(phase_a_path)
            ):
                raise ValueError("D1 Phase B checkpoint parent mismatch")
            final_root = expected_roots[phase_b_identity]
            final_receipts.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "selected_epoch": int(
                        phase_b_report["selected_epoch"]
                    ),
                    "training_report_path": str(
                        (final_root / "training_report.json").resolve()
                    ),
                    "training_report_sha256": _sha256(
                        final_root / "training_report.json"
                    ),
                    "checkpoint_path": str(
                        (final_root / "checkpoint.pt").resolve()
                    ),
                    "checkpoint_sha256": _sha256(
                        final_root / "checkpoint.pt"
                    ),
                    "initial_state_sha256": phase_b_report[
                        "initial_state_sha256"
                    ],
                }
            )
    return {
        "schema": SCHEMA,
        "terminal": VALIDATED,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "DEVELOPMENT_CHECKPOINT_VALIDATION_ONLY",
        "claim_ceiling": "SIX_CURRENT_STUDENT_CHECKPOINTS_NOT_FRESH_EFFECT",
        "contract_sha256": contract_sha256,
        "corpus_validation_sha256": corpus_validation_sha256,
        "student_samples_sha256": student_samples_sha256,
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "run_count": len(reports),
        "phase_a_run_count": 6,
        "phase_b_run_count": 6,
        "all_runs_30_epochs": True,
        "same_seed_initial_arrays_match_between_arms_and_phases": True,
        "phase_a_selection_recomputed": True,
        "phase_b_selected_epochs_match_phase_a": True,
        "final_checkpoint_count": len(final_receipts),
        "final_checkpoints": final_receipts,
        "fresh_firewall": {
            "fresh_media_loaded": False,
            "fresh_teacher_target_loaded": False,
            "fresh_student_output_computed": False,
            "fresh_used_for_checkpoint_or_threshold": False,
            "reserved_heldout_opened": False,
        },
        "authorization": {
            "freeze_fresh_execution_contract": True,
            "fresh_media_acquisition_executed": False,
            "fresh_student_forward_executed": False,
            "reserved_heldout_acquisition_authorized": False,
            "future_or_temporal_experiment_authorized": False,
            "mainline_promotion_authorized": False,
        },
    }


def _canonical_output(path: Path) -> Path:
    expected = (
        Path(__file__).resolve().parents[3]
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-training-validation-20260801/"
        "validation.json"
    ).resolve()
    if path.resolve() != expected:
        raise ValueError("D1 training validation output is not canonical")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--corpus-validation", type=Path, required=True)
    parser.add_argument("--student-samples", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _canonical_output(args.output)
        if output.exists():
            raise FileExistsError(
                "Refusing to overwrite D1 training validation"
            )
        report = validate(
            args.contract.resolve(),
            args.corpus_validation.resolve(),
            args.student_samples.resolve(),
            args.pretrained_checkpoint.resolve(),
            args.training_root.resolve(),
        )
        output.parent.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(
            tempfile.mkdtemp(
                prefix=f"{output.parent.name}.partial-",
                dir=output.parent.parent,
            )
        )
        with (partial / output.name).open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        if output.parent.exists():
            raise FileExistsError(
                "D1 training validation output root appeared"
            )
        partial.replace(output.parent)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "final_checkpoint_count": report[
                        "final_checkpoint_count"
                    ],
                    "validation_sha256": _sha256(output),
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
