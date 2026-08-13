#!/usr/bin/env python3
"""Read-only R11 Phase-A audit with the frozen round-12 receipt representation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from scripts.research.taro_o1r_r11_abstention_runtime import (
    validate_pool_phase_a as original,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ATTEMPT_01_REPAIR_RECEIPT_RELATIVE = (
    "docs/research/taro/"
    "TARO_O1R_R11_PHASE_A_INDEPENDENT_VALIDATOR_ROUND12_REPRESENTATION_REPAIR_2026-08-13.json"
)
REPAIR_RECEIPT_RELATIVE = (
    "docs/research/taro/"
    "TARO_O1R_R11_PHASE_A_INDEPENDENT_VALIDATOR_ROUND12_REPRESENTATION_REPAIR_ATTEMPT_02_2026-08-13.json"
)
ORIGINAL_LOCK_RELATIVE = original.LOCK_RELATIVE
ORIGINAL_VALIDATOR_RELATIVE = (
    "scripts/research/taro_o1r_r11_abstention_runtime/validate_pool_phase_a.py"
)
AUDITOR_RELATIVE = (
    "scripts/research/taro_o1r_r11_abstention_runtime/audit_pool_phase_a_round12_terminal.py"
)
AUDITOR_TEST_RELATIVE = (
    "scripts/research/taro_o1r_r11_abstention_runtime/test_audit_pool_phase_a_round12_terminal.py"
)
ORIGINAL_TERMINAL_RELATIVE = f"{original.EVIDENCE_ROOT}/terminal.json"
OUTPUT_ROOT_RELATIVE = (
    "artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-validator-round12-repair-r0"
)
OUTPUT_RELATIVE = f"{OUTPUT_ROOT_RELATIVE}/post-result-audit.json"
REPAIR_SCHEMA = "blindassist.taro.o1r.r11_phase_a_validator_round12_repair_attempt_02.v1"
RESULT_SCHEMA = "blindassist.taro.o1r.r11_phase_a_validator_round12_audit.v1"
REPAIR_STATUS = "R11_PHASE_A_VALIDATOR_ROUND12_REPRESENTATION_REPAIR_ATTEMPT_02_FROZEN"
PASS_STATUS = "TARO_O1R_R11_PHASE_A_OFFLINE_VALIDATOR_ROUND12_REPAIR_PASS"
FLOAT_DECIMALS = 12
REPAIRED_FIELDS = ("camera_to_world_4x4", "gravity_up_camera_xyz")

_ORIGINAL_INVENTORY_MEMBER_BINDINGS = original._inventory_member_bindings


class RepairAuditError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise RepairAuditError(code, message)


def _repo_path(relative: str) -> Path:
    return original.materializer.safe_join(REPO_ROOT, relative)


def _resolved_cli_path(value: Path | None, expected_relative: str) -> Path:
    candidate = _repo_path(expected_relative) if value is None else value
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RepairAuditError("R11_PHASE_A_REPAIR_JSON", f"JSON cannot be read: {path}") from error
    require(isinstance(value, dict), "R11_PHASE_A_REPAIR_JSON", f"JSON object required: {path}")
    return value


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema
        and isinstance(observed, str)
        and original.adapter.canonical_sha256(record) == observed,
        "R11_PHASE_A_REPAIR_SEAL",
        f"record seal/schema drift: {schema}",
    )
    record["content_sha256"] = observed
    return record


def _round12_float(value: float) -> float:
    require(math.isfinite(value), "R11_PHASE_A_REPAIR_NONFINITE", "non-finite trajectory value")
    rounded = round(float(value), FLOAT_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def canonicalize_round12_numeric(value: Any) -> Any:
    """Reproduce only the frozen JSON float representation used by the producer."""

    if isinstance(value, float):
        return _round12_float(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, list):
        return [canonicalize_round12_numeric(child) for child in value]
    if isinstance(value, tuple):
        return [canonicalize_round12_numeric(child) for child in value]
    raise RepairAuditError(
        "R11_PHASE_A_REPAIR_NUMERIC_TYPE",
        f"trajectory numeric structure contains {type(value).__name__}",
    )


def canonical_representation_matches(stored: Any, independently_reconstructed: Any) -> bool:
    """Compare exactly after round-12 serialization; no epsilon or tolerance is used."""

    return stored == canonicalize_round12_numeric(independently_reconstructed)


def _canonicalize_trajectory_member_bindings(
    members: MutableMapping[tuple[str, str, str, str], dict[str, Any]],
) -> int:
    repaired = 0
    for key, expected in members.items():
        if len(key) != 4 or key[-1] != "trajectory":
            continue
        require(
            all(field in expected for field in REPAIRED_FIELDS),
            "R11_PHASE_A_REPAIR_TRAJECTORY_FIELDS",
            "independent trajectory reconstruction lacks a frozen numeric field",
        )
        for field in REPAIRED_FIELDS:
            expected[field] = canonicalize_round12_numeric(expected[field])
        repaired += 1
    require(
        repaired == original.FRAME_COUNT,
        "R11_PHASE_A_REPAIR_TRAJECTORY_COUNT",
        "independent trajectory reconstruction count drift",
    )
    return repaired


@contextmanager
def _round12_inventory_binding_repair(stats: dict[str, int]) -> Iterator[None]:
    require(
        original._inventory_member_bindings is _ORIGINAL_INVENTORY_MEMBER_BINDINGS,
        "R11_PHASE_A_REPAIR_PATCH_PRECONDITION",
        "original validator inventory binding function was already replaced",
    )
    require(
        original.adapter.FLOAT_DECIMALS == FLOAT_DECIMALS,
        "R11_PHASE_A_REPAIR_DECIMAL_CONTRACT",
        "frozen producer decimal representation drift",
    )

    def repaired_inventory_member_bindings(
        inventory: Mapping[str, Any],
    ) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        members = _ORIGINAL_INVENTORY_MEMBER_BINDINGS(inventory)
        stats["trajectory_records_canonicalized"] = _canonicalize_trajectory_member_bindings(members)
        stats["numeric_fields_per_trajectory"] = len(REPAIRED_FIELDS)
        return members

    original._inventory_member_bindings = repaired_inventory_member_bindings
    try:
        yield
    finally:
        original._inventory_member_bindings = _ORIGINAL_INVENTORY_MEMBER_BINDINGS


def _verify_file_binding(binding: Any, expected_relative: str, label: str) -> Path:
    require(
        isinstance(binding, dict)
        and set(binding) == {"path", "bytes", "sha256"}
        and binding.get("path") == expected_relative,
        "R11_PHASE_A_REPAIR_BINDING",
        f"{label} binding identity drift",
    )
    path = _repo_path(expected_relative)
    require(
        path.is_file()
        and path.stat().st_size == binding.get("bytes")
        and _sha256_file(path) == binding.get("sha256"),
        "R11_PHASE_A_REPAIR_BINDING",
        f"{label} bytes/SHA drift",
    )
    return path


def _validate_repair_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    repair = _validate_seal(value, REPAIR_SCHEMA)
    expected_bindings = {
        "attempt_01_repair": ATTEMPT_01_REPAIR_RECEIPT_RELATIVE,
        "original_execution_lock": ORIGINAL_LOCK_RELATIVE,
        "original_terminal": ORIGINAL_TERMINAL_RELATIVE,
        "original_validator": ORIGINAL_VALIDATOR_RELATIVE,
        "repair_auditor": AUDITOR_RELATIVE,
        "repair_auditor_test": AUDITOR_TEST_RELATIVE,
    }
    require(
        repair.get("repair_id")
        == "TARO_O1R_R11_PHASE_A_INDEPENDENT_VALIDATOR_ROUND12_REPRESENTATION_REPAIR_ATTEMPT_02"
        and repair.get("status") == REPAIR_STATUS
        and repair.get("repair_class") == "PROTOCOL_ONLY_PATH_ALIAS_AND_NUMERIC_REPRESENTATION"
        and repair.get("output")
        == {"path": OUTPUT_RELATIVE, "fresh_root_required": True, "overwrite_forbidden": True},
        "R11_PHASE_A_REPAIR_IDENTITY",
        "repair identity/output policy drift",
    )
    bindings = repair.get("bindings")
    require(
        isinstance(bindings, dict) and set(bindings) == set(expected_bindings),
        "R11_PHASE_A_REPAIR_BINDING",
        "repair binding set drift",
    )
    for role, relative in expected_bindings.items():
        _verify_file_binding(bindings[role], relative, role)
    require(
        repair.get("numeric_contract")
        == {
            "producer_representation": "CANONICAL_JSON_FLOAT_ROUND_12",
            "independent_reconstruction": "FLOAT64_TRAJECTORY_INTERPOLATION_AND_NORMALIZED_GRAVITY",
            "repaired_fields": list(REPAIRED_FIELDS),
            "comparison": "EXACT_EQUALITY_AFTER_RECONSTRUCTION_ROUND_12",
            "tolerance": None,
            "epsilon": None,
        },
        "R11_PHASE_A_REPAIR_NUMERIC_CONTRACT",
        "repair numeric contract drift",
    )
    require(
        repair.get("path_contract")
        == {
            "accepted_cli_paths": [
                "exact repository-relative repair receipt path",
                "exact repository-relative formal output root path",
            ],
            "comparison": "RESOLVED_EXACT_EQUALITY_TO_THE_SAME_AUTHORIZED_TARGET",
            "alternate_root_or_target_allowed": False,
            "junction_scope_expanded": False,
        },
        "R11_PHASE_A_REPAIR_PATH_CONTRACT",
        "repair path-alias contract drift",
    )
    require(
        repair.get("attempt_01")
        == {
            "failure_code": "R11_PHASE_A_REPAIR_PATH",
            "failure_message": "repair receipt or output root path drift",
            "output_root_created": False,
            "partial_root_created": False,
            "phase_a_frame_payload_reads": 0,
            "model_rerun": False,
            "faro_highres_truth_label_outcome_reads": 0,
            "scientific_or_selection_result_created": False,
        },
        "R11_PHASE_A_REPAIR_ATTEMPT_01",
        "repair Attempt 01 incident boundary drift",
    )
    require(
        repair.get("scientific_influence")
        == {
            "phase_a_root_modified": False,
            "model_rerun": False,
            "source_or_candidate_rewritten": False,
            "algorithm_changed": False,
            "selector_or_threshold_changed": False,
            "parent_or_frame_roster_changed": False,
            "faro_highres_truth_label_outcome_read": False,
            "training_network_device_deployment_product_safety_authority": False,
        },
        "R11_PHASE_A_REPAIR_AUTHORITY",
        "repair scientific influence drift",
    )
    require(
        repair.get("authority")
        == "Read-only post-terminal validation of the same sealed R11 Phase-A root only; no model rerun, "
        "source or evidence rewrite, parent scoring, top-24 selection, FARO, highres depth, truth, label, "
        "outcome, training, network, device, deployment, product or safety authority.",
        "R11_PHASE_A_REPAIR_AUTHORITY",
        "repair authority drift",
    )
    return repair


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> tuple[int, str]:
    record = copy.deepcopy(dict(value))
    record["content_sha256"] = original.adapter.canonical_sha256(record)
    payload = original.adapter.canonical_json_bytes(record) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return len(payload), hashlib.sha256(payload).hexdigest().upper()


def _publish_single_file_root_atomic(target_root: Path, value: Mapping[str, Any]) -> tuple[int, str]:
    require(
        target_root.parent.is_dir() and not target_root.exists(),
        "R11_PHASE_A_REPAIR_ROOT_COLLISION",
        "fresh repair output root unavailable",
    )
    partial_root = Path(
        tempfile.mkdtemp(
            prefix=f".{target_root.name}.partial-",
            dir=target_root.parent,
        )
    )
    output_path = partial_root / "post-result-audit.json"
    try:
        output_bytes, output_sha256 = _write_json_exclusive(output_path, value)
        payload = output_path.read_bytes()
        require(
            len(payload) == output_bytes
            and hashlib.sha256(payload).hexdigest().upper() == output_sha256
            and {path.name for path in partial_root.iterdir()} == {output_path.name}
            and output_path.is_file(),
            "R11_PHASE_A_REPAIR_OUTPUT_REPLAY",
            "partial audit output bytes/SHA/file-set drift",
        )
        _validate_seal(_load_json(output_path), RESULT_SCHEMA)
        os.replace(partial_root, target_root)
    except BaseException:
        output_path.unlink(missing_ok=True)
        try:
            partial_root.rmdir()
        except OSError:
            pass
        raise
    return output_bytes, output_sha256


def audit_same_sealed_root(
    repair_receipt_path: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    expected_repair_path = _repo_path(REPAIR_RECEIPT_RELATIVE).resolve()
    expected_output_root = _repo_path(OUTPUT_ROOT_RELATIVE).resolve()
    repair_path = _resolved_cli_path(repair_receipt_path, REPAIR_RECEIPT_RELATIVE)
    target_root = _resolved_cli_path(output_root, OUTPUT_ROOT_RELATIVE)
    require(
        repair_path == expected_repair_path and target_root == expected_output_root,
        "R11_PHASE_A_REPAIR_PATH",
        "repair receipt or output root path drift",
    )
    repair = _validate_repair_receipt(_load_json(repair_path))
    require(
        not target_root.exists(),
        "R11_PHASE_A_REPAIR_ROOT_COLLISION",
        "fresh repair output root already exists",
    )

    stats: dict[str, int] = {}
    with _round12_inventory_binding_repair(stats):
        original_result = original.validate_evidence(
            _repo_path(original.EVIDENCE_ROOT),
            _repo_path(ORIGINAL_LOCK_RELATIVE),
        )
    require(
        original._inventory_member_bindings is _ORIGINAL_INVENTORY_MEMBER_BINDINGS,
        "R11_PHASE_A_REPAIR_PATCH_RESTORE",
        "original validator function was not restored",
    )
    require(
        original_result.get("passed") is True
        and original_result.get("terminal") == original.PASS_TERMINAL
        and original_result.get("parent_count") == original.PARENT_COUNT
        and original_result.get("frame_count") == original.FRAME_COUNT
        and original_result.get("query_count") == original.QUERY_COUNT
        and original_result.get("root_file_count") == original.FINAL_FILE_COUNT
        and original_result.get("highres_depth_member_payload_reads") == 0
        and original_result.get("faro_reads") == 0
        and original_result.get("truth_reads") == 0
        and stats
        == {
            "trajectory_records_canonicalized": original.FRAME_COUNT,
            "numeric_fields_per_trajectory": len(REPAIRED_FIELDS),
        },
        "R11_PHASE_A_REPAIR_ORIGINAL_VALIDATION",
        "repaired original validator did not reproduce the sealed PASS",
    )

    result = {
        "schema": RESULT_SCHEMA,
        "status": PASS_STATUS,
        "execution_validity": "VALID_WITH_POST_TERMINAL_NUMERIC_REPRESENTATION_REPAIR",
        "scientific_terminal": original.PASS_TERMINAL,
        "repair_receipt": {
            "path": REPAIR_RECEIPT_RELATIVE,
            "bytes": repair_path.stat().st_size,
            "sha256": _sha256_file(repair_path),
            "content_sha256": repair["content_sha256"],
        },
        "same_sealed_phase_a_root": original.EVIDENCE_ROOT,
        "original_terminal": copy.deepcopy(repair["bindings"]["original_terminal"]),
        "original_execution_lock": copy.deepcopy(repair["bindings"]["original_execution_lock"]),
        "original_validator": copy.deepcopy(repair["bindings"]["original_validator"]),
        "numeric_contract": copy.deepcopy(repair["numeric_contract"]),
        "trajectory_records_canonicalized": stats["trajectory_records_canonicalized"],
        "numeric_fields_per_trajectory": stats["numeric_fields_per_trajectory"],
        "original_validator_result": original_result,
        "phase_a_root_modified": False,
        "model_rerun": False,
        "parent_scoring_performed": False,
        "top24_selection_performed": False,
        "highres_depth_member_payload_reads": 0,
        "faro_reads": 0,
        "truth_reads": 0,
        "label_reads": 0,
        "outcome_reads": 0,
        "training_steps": 0,
        "network_requests": 0,
        "next_gate": "R11_SOURCE_ONLY_TOP24_IMPLEMENTATION_LOCK",
        "authority": (
            "Read-only repaired independent validation PASS for the same sealed Phase-A root. This permits "
            "only a separately frozen source-only top-24 implementation/execution lock; it does not itself "
            "perform selection or authorize FARO, scientific promotion, deployment, product or safety use."
        ),
    }
    output_bytes, output_sha256 = _publish_single_file_root_atomic(target_root, result)
    return {
        "passed": True,
        "status": PASS_STATUS,
        "scientific_terminal": original.PASS_TERMINAL,
        "parent_count": original.PARENT_COUNT,
        "frame_count": original.FRAME_COUNT,
        "query_count": original.QUERY_COUNT,
        "root_file_count": original.FINAL_FILE_COUNT,
        "output": OUTPUT_RELATIVE,
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-receipt", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = audit_same_sealed_root(args.repair_receipt, args.output_root)
    except Exception as error:  # noqa: BLE001 - CLI must turn every validation failure into a receipt
        print(
            json.dumps(
                {
                    "passed": False,
                    "failure_code": str(getattr(error, "code", type(error).__name__)),
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
