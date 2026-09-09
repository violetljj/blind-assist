from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import (
    _write_exclusive_json,
    build_dataset_governed,
)
from .evaluator import evaluate_dataset
from .io_utils import read_json, sha256_file, write_json
from .validator import validate_dataset


SEALED_OUTCOME_FILES = (
    "algorithm_pair_ledger.jsonl",
    "truth_pair_ledger.jsonl",
    "old_r1_comparison.jsonl",
    "scientific_summary.json",
    "validation.json",
)


def _terminal_path(claim_path: Path) -> Path:
    return claim_path.with_name(claim_path.stem + ".terminal.json")


def _write_failure_terminal(
    claim_path: Path,
    activation: dict[str, Any],
    output_root: Path,
    error: Exception,
) -> None:
    terminal_path = _terminal_path(claim_path)
    if terminal_path.exists():
        return
    _write_exclusive_json(
        terminal_path,
        {
            "schema": "rcle.synthetic_scene.sealed_terminal.v1",
            "protocol_id": activation.get("protocol_id"),
            "claim_id": activation.get("claim_id"),
            "status": "INVALID_SEALED_EXECUTION_FAIL_CLOSED",
            "claim_consumed": True,
            "canonical_output_root": str(output_root.resolve()),
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
            "retry_or_replacement_forbidden": True,
        },
    )


def execute_sealed_once(
    spec_path: Path, output_root: Path, activation_path: Path
) -> dict[str, Any]:
    """Consume one claim across generation, validation, evaluation, and terminal."""
    spec_path = spec_path.resolve()
    output_root = output_root.resolve()
    activation_path = activation_path.resolve()
    activation = read_json(activation_path)
    claim_path = Path(activation["claim_path"]).resolve()
    terminal_path = _terminal_path(claim_path)
    if claim_path.exists() or terminal_path.exists():
        raise PermissionError("SEALED_CLAIM_ALREADY_CONSUMED")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("SEALED_OUTPUT_ROOT_NOT_EMPTY")
    try:
        build_dataset_governed(
            spec_path,
            output_root,
            split="sealed",
            activation=activation_path,
        )
        initial_validation = validate_dataset(
            output_root, check_algorithm=False
        )
        if initial_validation["status"] != "PASS":
            raise RuntimeError("SEALED_DATASET_VALIDATION_FAILED")
        preexisting = [
            name
            for name in SEALED_OUTCOME_FILES
            if (output_root / name).exists()
        ]
        if preexisting:
            raise FileExistsError(
                "SEALED_OUTCOME_ALREADY_EXISTS:" + ",".join(preexisting)
            )
        summary = evaluate_dataset(
            output_root, sealed_execution_authorized=True
        )
        final_validation = validate_dataset(
            output_root, check_algorithm=True
        )
        if final_validation["status"] != "PASS":
            raise RuntimeError("SEALED_RESULT_VALIDATION_FAILED")
        validation_sha = write_json(
            output_root / "validation.json", final_validation
        )
        summary_path = output_root / "scientific_summary.json"
        receipt_path = output_root / "receipt.json"
        terminal = {
            "schema": "rcle.synthetic_scene.sealed_terminal.v1",
            "protocol_id": activation["protocol_id"],
            "claim_id": activation["claim_id"],
            "status": "SYNTHETIC_SEALED_EXECUTION_COMPLETE",
            "claim_consumed": True,
            "canonical_output_root": str(output_root),
            "activation_sha256": sha256_file(activation_path),
            "dataset_receipt_sha256": sha256_file(receipt_path),
            "scientific_summary_sha256": sha256_file(summary_path),
            "validation_sha256": validation_sha,
            "scientific_outcome": summary["scientific_outcome"],
            "all_local_gates_pass": summary["all_local_gates_pass"],
            "retry_or_replacement_forbidden": True,
            "authority_ceiling": summary["claim_ceiling"],
        }
        _write_exclusive_json(terminal_path, terminal)
        return terminal
    except Exception as exc:
        if claim_path.is_file():
            _write_failure_terminal(
                claim_path, activation, output_root, exc
            )
        raise
