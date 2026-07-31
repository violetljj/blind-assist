"""Independently audit every explicit R2-P0 completion requirement."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import PROTOCOL_ID
from .build_readiness_lock import build as recompute_readiness_lock


TERMINAL = "R2_NOT_WORTH_BURNING_FRESH_HOLDOUT"
CURRENT_STATUSES = {
    "SEGMENTATION_MODEL_SELECTION_R1_BLOCKED",
    "MODEL_SELECTION_NOT_EVALUABLE",
    "R2_NOT_AUTHORIZED",
    "DEVICE_BENCHMARK_NOT_AUTHORIZED",
    "DEFAULT_APP_UNCHANGED",
}
CONSUMED_SESSIONS = {
    "GxMb4zhAvoM5jbF54kfcs8wxTL4fqNnT",
    "972O8sd5HpUbGeEE_UAb1g0z1OZUtfHl",
    "ic_BpoiSOIW-7_mffGenT6yissRNiPzT",
    "eHxtA669WpN381O4ZjVAmG3-3ZUewuXr",
}
RUNTIME_STAGES = {
    "preprocess",
    "tflite_inference",
    "output_dequantize_argmax",
    "component_extraction",
    "fusion_operator",
    "total_increment",
}
SUMMARY_STATS = {"mean", "p50", "p90", "p95", "min", "max"}


class CloseoutValidationError(RuntimeError):
    """Raised when a closeout requirement is not proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CloseoutValidationError(message)


def _load(repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = repo_root / relative_path
    _require(path.is_file(), f"missing JSON: {relative_path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON is not an object: {relative_path}")
    return value


def validate(repo_root: Path) -> dict[str, Any]:
    checks: list[dict[str, str]] = []

    current = (
        repo_root / "docs/research/dual-loop/README.md"
    ).read_text(encoding="utf-8")
    _require(all(status in current for status in CURRENT_STATUSES), "current status mismatch")
    checks.append({"requirement": "1_current_entry", "status": "PASS"})

    recomputed_lock = recompute_readiness_lock(repo_root)
    _require(
        recomputed_lock["r1_immutable_verification"]["formal_freeze_identity_count"]
        == 22,
        "R1 frozen identity count mismatch",
    )
    checks.append({"requirement": "2_r1_immutability", "status": "PASS"})

    amendment = _load(
        repo_root,
        "docs/research/dual-loop/"
        "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_CONSUMED_ROLE_AMENDMENT_2026-08-01.json",
    )
    _require(set(amendment["consumed_sessions"]) == CONSUMED_SESSIONS, "consumed session mismatch")
    _require(
        set(amendment["allowed_roles"])
        == {"regression", "rehearsal", "validator", "canonicalizer_canary"},
        "consumed allowed roles mismatch",
    )
    _require(
        all(
            value == "DOES_NOT_RESTORE_FRESHNESS"
            for key, value in amendment["anti_recovery_invariant"].items()
            if key != "identity_basis"
        ),
        "identity-laundering invariant mismatch",
    )
    checks.append({"requirement": "3_consumed_role", "status": "PASS"})

    protocol = _load(
        repo_root,
        "docs/research/dual-loop/"
        "DUAL_LOOP_SEGMENTATION_R2_P0_PROTOCOL_DRAFT_2026-07-31.json",
    )
    _require(protocol["status"] == "DRAFT_NOT_AUTHORIZED_FOR_FORMAL", "protocol authorized")
    _require(protocol["candidate_strategy"] == "single_candidate_qualification", "not single candidate")
    _require(CURRENT_STATUSES <= set(protocol["current_terminals"]), "protocol terminal mismatch")
    checks.append({"requirement": "4_protocol_draft", "status": "PASS"})

    contract = _load(
        repo_root,
        "configs/dual_loop_segmentation_r2_p0/canonicalization_contract.json",
    )
    mapping = contract["source_native_to_canonical"]
    _require(set(map(int, mapping)) == set(range(31)), "native mapping is incomplete")
    _require(set(mapping.values()) <= {0, 1, 2, 3}, "canonical mapping is invalid")
    _require(
        set(contract["source_decoder"]["accepted_png_modes"]) == {"L", "P", "RGB", "RGBA"},
        "decoder mode contract mismatch",
    )
    _require(contract["unknown_native_id_policy"] == "FAIL_CLOSED_BEFORE_OUTPUT", "unknown ID not closed")
    _require(contract["output"]["resize"] == "PIL_NEAREST", "resize is not nearest")
    lock_names = set(recomputed_lock["pre_fresh_access_frozen_identities"])
    _require(
        {"canonicalizer_code", "canonicalization_contract", "canonical_view_schema"}
        <= lock_names,
        "canonical code/config/schema are not frozen",
    )
    checks.append({"requirement": "5_canonical_contract", "status": "PASS"})

    canonical = _load(
        repo_root,
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "canonical-view-validation.json",
    )
    _require(canonical["status"] == "VALID" and canonical["row_count"] == 924, "canonical view invalid")
    _require(canonical["all_native_ids_0_30_covered"], "native canary coverage incomplete")
    _require(canonical["all_canonical_ids_within_0_3"], "canonical ID leakage")
    checks.append({"requirement": "6_materialized_view", "status": "PASS"})

    rehearsal = _load(
        repo_root,
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "rehearsal-ddrnet-baseline-v2-validation.json",
    )
    _require(rehearsal["status"] == "VALID", "rehearsal validator invalid")
    _require(rehearsal["frame_count"] == 200, "rehearsal frame count mismatch")
    _require(rehearsal["rehearsal_role"] == "r1_consumed_fresh", "rehearsal role mismatch")
    checks.append({"requirement": "7_consumed_synthetic_rehearsal", "status": "PASS"})

    for key in (
        "manifest_identity_valid",
        "canonical_ids_only_0_3",
        "frame_component_counts_valid",
        "source_session_aggregates_recomputed",
        "atomic_publish",
        "interruption_resume_supported",
        "independent_full_recompute",
    ):
        _require(rehearsal[key] is True, f"rehearsal check failed: {key}")
    _require(rehearsal["zero_row_policy"] == "FAIL_CLOSED", "zero rows not fail closed")
    checks.append({"requirement": "8_rehearsal_full_validation", "status": "PASS"})

    runtime = _load(
        repo_root,
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "runtime-ddrnet-baseline-validation.json",
    )
    runtime_schema = _load(
        repo_root,
        "configs/dual_loop_segmentation_r2_p0/runtime_rows.schema.json",
    )
    _require(runtime["status"] == "VALID" and runtime["row_count"] == 200, "runtime invalid")
    _require(
        set(runtime_schema["properties"]["stages_ms"]["required"]) == RUNTIME_STAGES,
        "runtime stage schema mismatch",
    )
    for stage in RUNTIME_STAGES:
        _require(SUMMARY_STATS <= set(runtime["independent_recompute"][stage]), f"stats missing: {stage}")
    checks.append({"requirement": "9_per_frame_runtime", "status": "PASS"})

    _require(
        {
            "dev_frames",
            "dev_components",
            "dev_report",
            "runtime_rows",
            "yolo_trace",
            "checkpoint",
            "tflite",
            "postprocess_config",
            "rehearsal_evaluator",
            "rehearsal_validator",
        }
        <= lock_names,
        "pre-fresh identity freeze incomplete",
    )
    checks.append({"requirement": "10_pre_fresh_hash_freeze", "status": "PASS"})

    gate = _load(
        repo_root,
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "candidate-gate-matrix.json",
    )
    _require(len(gate["candidates"]) == 3, "gate matrix candidate count mismatch")
    ddrnet, segformer, refined = gate["candidates"]
    _require(
        ddrnet["metrics"]["false_activation_components_per_frame"] == 7.885,
        "DDRNet baseline mismatch",
    )
    _require(
        segformer["metrics"]["total_incremental_host_p95_ms"] == 74.139325,
        "SegFormer runtime mismatch",
    )
    _require(
        not ddrnet["all_formal_hard_gates_passed"]
        and not segformer["all_formal_hard_gates_passed"]
        and not refined["all_formal_hard_gates_passed"],
        "gate matrix unexpectedly passed",
    )
    checks.append({"requirement": "11_candidate_gate_matrix", "status": "PASS"})

    refinement = _load(
        repo_root,
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "ddrnet-refinement/report.json",
    )
    _require(refinement["combination_count"] == 36, "refinement search count mismatch")
    _require(refinement["qualified_utility_candidate_count"] == 0, "refinement qualified")
    _require(refinement["selected_candidate_id"] is None, "refinement selected candidate")
    checks.append({"requirement": "12_bounded_refinement", "status": "PASS"})

    _require(gate["qualified_candidate_count"] == 0, "qualified candidate count is nonzero")
    _require(gate["selected_candidate_id"] is None, "candidate selected")
    _require(gate["terminal_implication"] == TERMINAL, "terminal mismatch")
    checks.append({"requirement": "13_terminal_rule", "status": "PASS"})

    metadata = _load(
        repo_root,
        "artifacts.local/evidence/dual-loop-segmentation-r2-p0/"
        "holdout-metadata-audit.json",
    )
    _require(metadata["selection_status"] == "NO_HOLDOUT_SELECTED", "holdout selected")
    _require(metadata["mask_objects_downloaded"] == 0, "mask object downloaded")
    _require(metadata["mask_pixels_read"] == 0, "mask pixels read")
    _require(metadata["candidate_outputs_run"] == 0, "fresh candidate output run")
    checks.append({"requirement": "14_no_new_truth_access", "status": "PASS"})

    return {
        "schema_version":
            "blindassist.dual_loop_segmentation_r2_p0.closeout_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID",
        "terminal": TERMINAL,
        "requirement_count": len(checks),
        "passed_requirement_count": len(checks),
        "checks": checks,
        "independent_current_state_recompute": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_atomic(output: Path, value: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite closeout validation: {output}")
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
    result = validate(args.repo_root.resolve())
    write_atomic(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "terminal": result["terminal"],
                "requirements": result["requirement_count"],
            }
        )
    )
