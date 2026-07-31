"""Freeze R2-P0 identities and emit the fail-closed readiness terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import PROTOCOL_ID


R1_CRITICAL_EXPECTED = {
    "docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_RESULT_2026-07-31.md":
        "c51c171ff8651d824b8c5a127dd86f6b69462420270f40b1c9879aa3180460a8",
    "scripts/research/dual_loop_segmentation_model_selection/"
    "validate_model_selection_closeout.py":
        "d3e0ed40f088789899d53ab92d01ca09d27f599f924008b54b260ca0407fefdf",
    "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
    "independent_closeout_validation_v2.json":
        "7658a5c4504c766afc04b3573c8a6068c0c4302f6508dc05c62c02db64866b86",
    "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
    "formal/failure_receipt.json":
        "591ffe36ca75b4ba37d0b7f2f74abffc3f06ff79fe5a552c387bdbe70a91796b",
    "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
    "formal_freeze_receipt.json":
        "fb2f3329c5691b21b0a5d15a070987adb12af5a4868ceeca2d52ffbcff991e54",
}

R2_IDENTITY_PATHS = {
    "canonicalizer_code":
        "scripts/research/dual_loop_segmentation_r2_p0/canonicalizer.py",
    "canonicalization_contract":
        "configs/dual_loop_segmentation_r2_p0/canonicalization_contract.json",
    "canonical_view_schema":
        "configs/dual_loop_segmentation_r2_p0/materialized_canonical_view.schema.json",
    "canonical_view_manifest":
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "canonical-view/manifest.jsonl",
    "dev_frames":
        "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
        "dev/ddrnet23_slim/frames.jsonl",
    "dev_components":
        "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
        "dev/ddrnet23_slim/components.jsonl",
    "dev_report":
        "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
        "dev/ddrnet23_slim/report.json",
    "runtime_rows":
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "runtime-ddrnet-baseline/runtime_rows.jsonl",
    "yolo_trace":
        "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
        "dev/yolo_trace.jsonl",
    "checkpoint":
        "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
        "ddrnet23_slim/fp32/fp32_checkpoint.pt",
    "tflite":
        "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
        "ddrnet23_slim/int8/model_int8.tflite",
    "postprocess_config":
        "configs/dual_loop_segmentation_r2_p0/baseline_postprocess.json",
    "rehearsal_evaluator":
        "scripts/research/dual_loop_segmentation_r2_p0/run_rehearsal.py",
    "rehearsal_validator":
        "scripts/research/dual_loop_segmentation_r2_p0/validate_rehearsal.py",
    "runtime_harness":
        "scripts/research/dual_loop_segmentation_r2_p0/benchmark_runtime_rows.py",
    "runtime_validator":
        "scripts/research/dual_loop_segmentation_r2_p0/validate_runtime_rows.py",
    "refinement_search_config":
        "configs/dual_loop_segmentation_r2_p0/ddrnet_refinement_search.json",
    "refinement_report":
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "ddrnet-refinement/report.json",
    "candidate_gate_matrix":
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "candidate-gate-matrix.json",
    "canonical_view_validation":
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "canonical-view-validation.json",
    "rehearsal_validation":
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "rehearsal-ddrnet-baseline-v2-validation.json",
    "runtime_validation":
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "runtime-ddrnet-baseline-validation.json",
    "holdout_metadata_audit":
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "holdout-metadata-audit.json",
}


class ReadinessLockError(RuntimeError):
    """Raised when an identity or readiness prerequisite fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReadinessLockError(f"required identity file is missing: {relative_path}")
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _nested_identity_entries(value: Any) -> Iterator[dict[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("relative_path"), str) and isinstance(
            value.get("sha256"), str
        ):
            yield {
                "relative_path": value["relative_path"],
                "sha256": value["sha256"],
            }
        for child in value.values():
            yield from _nested_identity_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_identity_entries(child)


def verify_frozen_identities(
    repo_root: Path,
    freeze_receipt: dict[str, Any],
) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in _nested_identity_entries(freeze_receipt):
        relative_path = entry["relative_path"].replace("\\", "/")
        if relative_path in seen:
            continue
        seen.add(relative_path)
        path = repo_root / Path(relative_path)
        actual = sha256_file(path) if path.is_file() else None
        if actual != entry["sha256"]:
            raise ReadinessLockError(
                f"R1 frozen identity mismatch: {relative_path}; "
                f"expected={entry['sha256']} actual={actual}"
            )
        verified.append(
            {
                "relative_path": relative_path,
                "sha256": actual,
            }
        )
    if not verified:
        raise ReadinessLockError("R1 formal freeze receipt has no identity entries")
    return verified


def _load_json(repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = repo_root / relative_path
    if not path.is_file():
        raise ReadinessLockError(f"required JSON is missing: {relative_path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReadinessLockError(f"required JSON is not an object: {relative_path}")
    return value


def build(repo_root: Path) -> dict[str, Any]:
    formal_freeze_path = (
        "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/"
        "formal_freeze_receipt.json"
    )
    formal_freeze = _load_json(repo_root, formal_freeze_path)
    verified_r1_identities = verify_frozen_identities(repo_root, formal_freeze)

    critical: dict[str, dict[str, Any]] = {}
    for relative_path, expected in R1_CRITICAL_EXPECTED.items():
        identity = _identity(repo_root / relative_path, relative_path)
        if identity["sha256"] != expected:
            raise ReadinessLockError(
                f"R1 critical artifact changed: {relative_path}; "
                f"expected={expected} actual={identity['sha256']}"
            )
        critical[relative_path] = identity

    identities = {
        name: _identity(repo_root / relative_path, relative_path)
        for name, relative_path in R2_IDENTITY_PATHS.items()
    }
    canonical_validation = _load_json(
        repo_root, R2_IDENTITY_PATHS["canonical_view_validation"]
    )
    rehearsal_validation = _load_json(
        repo_root, R2_IDENTITY_PATHS["rehearsal_validation"]
    )
    runtime_validation = _load_json(
        repo_root, R2_IDENTITY_PATHS["runtime_validation"]
    )
    metadata_audit = _load_json(
        repo_root, R2_IDENTITY_PATHS["holdout_metadata_audit"]
    )
    gate_matrix = _load_json(
        repo_root, R2_IDENTITY_PATHS["candidate_gate_matrix"]
    )
    validator_statuses = {
        "canonical_view": canonical_validation.get("status"),
        "rehearsal": rehearsal_validation.get("status"),
        "runtime": runtime_validation.get("status"),
    }
    if set(validator_statuses.values()) != {"VALID"}:
        raise ReadinessLockError(
            f"validators are not all VALID: {validator_statuses}"
        )
    if metadata_audit.get("status") != "METADATA_ONLY_AVAILABILITY_AUDITED":
        raise ReadinessLockError("holdout metadata audit is not complete")
    if metadata_audit.get("mask_objects_downloaded") != 0:
        raise ReadinessLockError("metadata audit downloaded mask objects")
    if metadata_audit.get("mask_pixels_read") != 0:
        raise ReadinessLockError("metadata audit read mask pixels")
    if metadata_audit.get("candidate_outputs_run") != 0:
        raise ReadinessLockError("metadata audit ran candidate outputs")

    selected = gate_matrix.get("selected_candidate_id")
    qualified = gate_matrix.get("qualified_candidate_count")
    implication = gate_matrix.get("terminal_implication")
    if selected is not None or qualified != 0:
        raise ReadinessLockError(
            "gate matrix unexpectedly selected a candidate; this lock only closes P0 HOLD"
        )
    if implication != "R2_NOT_WORTH_BURNING_FRESH_HOLDOUT":
        raise ReadinessLockError(f"unexpected gate terminal: {implication}")

    return {
        "schema_version":
            "blindassist.dual_loop_segmentation_r2_p0.readiness_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "R2_P0_READINESS_CLOSED",
        "terminal": implication,
        "formal_authority": False,
        "r2_authorized": False,
        "fresh_holdout_selected": False,
        "fresh_mask_truth_accessed": False,
        "candidate_outputs_on_fresh_holdout": 0,
        "selected_candidate_id": None,
        "qualified_candidate_count": 0,
        "validator_statuses": validator_statuses,
        "metadata_eligible_session_count":
            metadata_audit.get("metadata_eligible_session_count"),
        "metadata_error_count": metadata_audit.get("metadata_error_count"),
        "r1_immutable_verification": {
            "critical_artifacts": critical,
            "formal_freeze_identity_count": len(verified_r1_identities),
            "formal_freeze_identities": verified_r1_identities,
        },
        "pre_fresh_access_frozen_identities": identities,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_atomic(output: Path, value: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite readiness lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = build(args.repo_root.resolve())
    write_atomic(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "terminal": result["terminal"],
                "r1_identities_verified":
                    result["r1_immutable_verification"][
                        "formal_freeze_identity_count"
                    ],
            }
        )
    )
