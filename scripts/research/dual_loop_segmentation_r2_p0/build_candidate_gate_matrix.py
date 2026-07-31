"""Build the R2-P0 candidate gate matrix from frozen Development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import PROTOCOL_ID
from .canonicalizer import sha256_file


FORMAL_GATES = {
    "delta_recall_C_minus_A": (">=", 0.05),
    "delta_false_positive_area_fraction_C_minus_A": ("<=", 0.05),
    "candidate_component_recall": (">=", 0.5),
    "false_activation_components_per_frame": ("<=", 3.0),
    "segmentation_host_p95_ms": ("<=", 25.0),
    "total_incremental_host_p95_ms": ("<=", 30.0),
    "consistent_sessions": (">=", 2),
}
READINESS_GATES = {
    "delta_recall_C_minus_A": (">=", 0.06),
    "delta_false_positive_area_fraction_C_minus_A": ("<=", 0.04),
    "candidate_component_recall": (">=", 0.55),
    "false_activation_components_per_frame": ("<=", 2.5),
    "segmentation_host_p95_ms": ("<=", 22.5),
    "total_incremental_host_p95_ms": ("<=", 27.0),
    "consistent_sessions": (">=", 3),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gate_matrix(metrics: dict[str, Any], gates: dict[str, tuple[str, float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (operator, threshold) in gates.items():
        actual = metrics.get(name)
        if actual is None:
            passed = False
            status = "NOT_EVALUABLE"
        else:
            passed = float(actual) >= threshold if operator == ">=" else float(actual) <= threshold
            status = "PASS" if passed else "FAIL"
        result[name] = {
            "actual": actual,
            "operator": operator,
            "threshold": threshold,
            "status": status,
        }
    return result


def _candidate(
    *,
    candidate_id: str,
    metrics: dict[str, Any],
    evidence_status: str,
    notes: list[str],
) -> dict[str, Any]:
    formal = _gate_matrix(metrics, FORMAL_GATES)
    readiness = _gate_matrix(metrics, READINESS_GATES)
    return {
        "candidate_id": candidate_id,
        "evidence_status": evidence_status,
        "metrics": metrics,
        "formal_hard_gates": formal,
        "readiness_margin_gates": readiness,
        "all_formal_hard_gates_passed": all(
            value["status"] == "PASS" for value in formal.values()
        ),
        "all_readiness_margin_gates_passed": all(
            value["status"] == "PASS" for value in readiness.values()
        ),
        "notes": notes,
    }


def build(
    *,
    ddrnet_report_path: Path,
    segformer_report_path: Path,
    segformer_runtime_path: Path,
    baseline_runtime_validation_path: Path,
    refinement_matrix_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    ddrnet = _read_json(ddrnet_report_path)
    segformer = _read_json(segformer_report_path)
    seg_runtime = _read_json(segformer_runtime_path)
    baseline_runtime = _read_json(baseline_runtime_validation_path)
    refinement_rows = _read_jsonl(refinement_matrix_path)
    if not refinement_rows:
        raise ValueError("refinement matrix is empty")
    best_refined = max(
        refinement_rows,
        key=lambda row: float(row["minimum_normalized_readiness_margin"]),
    )
    ddr_summary = ddrnet["summary"]
    seg_summary = segformer["summary"]
    ddr_metrics = {
        "delta_recall_C_minus_A": ddr_summary["delta_recall_C_minus_A"],
        "delta_false_positive_area_fraction_C_minus_A": ddr_summary[
            "delta_false_positive_area_fraction_C_minus_A"
        ],
        "candidate_component_recall": ddr_summary["candidate_components"]["component_recall"],
        "false_activation_components_per_frame": ddr_summary[
            "false_activation_components_per_frame"
        ],
        "segmentation_host_p95_ms": baseline_runtime["independent_recompute"][
            "tflite_inference"
        ]["p95"],
        "total_incremental_host_p95_ms": baseline_runtime["independent_recompute"][
            "total_increment"
        ]["p95"],
        "consistent_sessions": ddrnet["r1_metrics"]["consistent_sessions"]["count"],
    }
    seg_metrics = {
        "delta_recall_C_minus_A": seg_summary["delta_recall_C_minus_A"],
        "delta_false_positive_area_fraction_C_minus_A": seg_summary[
            "delta_false_positive_area_fraction_C_minus_A"
        ],
        "candidate_component_recall": seg_summary["candidate_components"]["component_recall"],
        "false_activation_components_per_frame": seg_summary[
            "false_activation_components_per_frame"
        ],
        "segmentation_host_p95_ms": seg_runtime["runtime"]["tflite_inference"]["p95"],
        "total_incremental_host_p95_ms": seg_runtime["runtime"]["total_increment"]["p95"],
        "consistent_sessions": segformer["r1_metrics"]["consistent_sessions"]["count"],
    }
    refined_metrics = {
        **best_refined["metrics"],
        "segmentation_host_p95_ms": None,
        "total_incremental_host_p95_ms": None,
    }
    candidates = [
        _candidate(
            candidate_id="DDRNet-23-Slim R1 INT8 baseline",
            metrics=ddr_metrics,
            evidence_status="DEVELOPMENT_WITH_INDEPENDENT_RUNTIME_ROWS",
            notes=[
                "False activation remains 7.885/frame.",
                "Delta FP area remains above both formal and readiness gates.",
            ],
        ),
        _candidate(
            candidate_id="SegFormer-B0 R1 INT8 baseline",
            metrics=seg_metrics,
            evidence_status="DEVELOPMENT_RUNTIME_AGGREGATE_ONLY",
            notes=[
                "Total incremental P95 is 74.139 ms.",
                "R1 runtime receipt lacks immutable per-frame timing rows.",
                "Unchanged SegFormer-B0 is forbidden from R2 formal.",
            ],
        ),
        _candidate(
            candidate_id=str(best_refined["candidate_id"]),
            metrics=refined_metrics,
            evidence_status="BOUNDED_REFINEMENT_NEAR_BEST_NOT_SELECTED",
            notes=[
                "Utility search produced zero readiness-qualified candidates.",
                "Runtime was not run because the candidate already failed delta FP area.",
            ],
        ),
    ]
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_r2_p0.candidate_gate_matrix.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "formal_authority": False,
        "candidates": candidates,
        "selected_candidate_id": None,
        "qualified_candidate_count": 0,
        "terminal_implication": "R2_NOT_WORTH_BURNING_FRESH_HOLDOUT",
        "evidence_sha256": {
            "ddrnet_dev_report": sha256_file(ddrnet_report_path),
            "segformer_dev_report": sha256_file(segformer_report_path),
            "segformer_runtime_receipt": sha256_file(segformer_runtime_path),
            "baseline_runtime_validation": sha256_file(baseline_runtime_validation_path),
            "refinement_matrix": sha256_file(refinement_matrix_path),
        },
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite gate matrix: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddrnet-report", type=Path, required=True)
    parser.add_argument("--segformer-report", type=Path, required=True)
    parser.add_argument("--segformer-runtime", type=Path, required=True)
    parser.add_argument("--baseline-runtime-validation", type=Path, required=True)
    parser.add_argument("--refinement-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    value = build(
        ddrnet_report_path=args.ddrnet_report,
        segformer_report_path=args.segformer_report,
        segformer_runtime_path=args.segformer_runtime,
        baseline_runtime_validation_path=args.baseline_runtime_validation,
        refinement_matrix_path=args.refinement_matrix,
        output_path=args.output,
    )
    print(json.dumps({"status": value["status"], "terminal": value["terminal_implication"]}))
