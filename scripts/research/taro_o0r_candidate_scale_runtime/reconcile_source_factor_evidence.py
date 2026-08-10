#!/usr/bin/env python3
"""Reconcile TARO R1 derived summaries from canonical persisted query records."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale, source_factor
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


ORIGINAL_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-factor-r1"
RECONCILED_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-factor-r1a-reconciliation"
SCALE_ORACLE_PATH = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-candidate-scale-r0/oracle-comparisons.json.gz"
MAXIMUM_EVIDENCE_BYTES = 4 * 1024 * 1024
# One unit at the adapter's 12-decimal canonical precision, plus binary-float
# representation slack for values loaded back from JSON.
ROUNDING_TOLERANCE = 1.1e-12


class ReconciliationError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise ReconciliationError(code, message, **context)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_gzip(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in output, "RECONCILIATION_SEAL_COLLISION", "payload already contains a seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _verify_original_manifest() -> tuple[dict[str, Any], int]:
    manifest_path = ORIGINAL_ROOT / "manifest.json"
    require(manifest_path.is_file(), "ORIGINAL_MANIFEST_MISSING", "original R1 manifest is missing")
    manifest = _load_json(manifest_path)
    require(manifest.get("schema") == "blindassist.taro.o0r.source_anchored_factor_canary_manifest.v1" and isinstance(manifest.get("files"), dict), "ORIGINAL_MANIFEST_INVALID", "original R1 manifest schema/files drift")
    total = 0
    for relative, receipt in manifest["files"].items():
        path = materializer.safe_join(ORIGINAL_ROOT, relative)
        require(
            path.is_file()
            and path.stat().st_size == receipt.get("bytes")
            and materializer.sha256_file(path) == receipt.get("sha256"),
            "ORIGINAL_MANIFEST_FILE_DRIFT",
            "original R1 evidence file differs from manifest",
            relative=relative,
        )
        total += path.stat().st_size
    require(total == manifest["bytes_before_manifest"] and len(manifest["files"]) == manifest["file_count_before_manifest"], "ORIGINAL_MANIFEST_TOTAL_DRIFT", "original manifest totals drift")
    return manifest, total


def _numeric_differences(left: Any, right: Any, path: str = "summary") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left) == set(right), "SUMMARY_STRUCTURE_DRIFT", "summary key structure differs", path=path)
        output: list[dict[str, Any]] = []
        for key in sorted(left):
            if key == "content_sha256":
                continue
            output.extend(_numeric_differences(left[key], right[key], f"{path}.{key}"))
        return output
    if isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), "SUMMARY_STRUCTURE_DRIFT", "summary list length differs", path=path)
        output = []
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            output.extend(_numeric_differences(first, second, f"{path}[{index}]"))
        return output
    numeric = isinstance(left, Real) and not isinstance(left, bool) and isinstance(right, Real) and not isinstance(right, bool)
    if numeric:
        difference = abs(float(left) - float(right))
        return [] if difference == 0.0 else [{"path": path, "original": float(left), "reconciled": float(right), "absolute_difference": difference}]
    require(left == right, "SUMMARY_NONNUMERIC_DRIFT", "summary nonnumeric value differs", path=path, original=left, reconciled=right)
    return []


def _oracle_frame_errors() -> dict[str, float]:
    raw = _load_json_gzip(SCALE_ORACLE_PATH)
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in raw:
        row = apple_scale.validate_oracle_comparison(item)
        if row["evaluable"]:
            grouped[row["physical_frame_id"]].append(float(row["source_abs_log_error"]))
    require(len(raw) == 1494 and len(grouped) == 166, "ORACLE_COMPARISON_COHORT_DRIFT", "R0 oracle comparison cohort drift")
    return {frame_id: float(np.median(np.asarray(values, dtype=np.float64))) for frame_id, values in grouped.items()}


def _diagnostics(records: list[dict[str, Any]], reliability: list[dict[str, Any]]) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    lost_frames: dict[tuple[str, str], int] = Counter()
    for row in records:
        for mode in ("baseline", "source_anchored"):
            if row[mode]["extraction_evaluable"] is False:
                failures[f"{mode}:{'+'.join(row[mode]['reason_codes'])}"] += 1
        if row["effects"]["extraction_lost"]:
            lost_frames[(row["parent_id"], row["physical_frame_id"])] += 1
    effect_rows: dict[str, Any] = {}
    for metric in (
        "support_height_error_reduction_m",
        "support_normal_error_reduction_rad",
        "boundary_jaccard_increase",
        "boundary_xyz_error_reduction_m",
        "query_point_error_reduction_m",
    ):
        values = [float(row["effects"][metric]) for row in records if row["effects"][metric] is not None]
        effect_rows[metric] = {
            "paired_query_count": len(values),
            "improved_query_count": sum(value > 0.0 for value in values),
            "unchanged_query_count": sum(value == 0.0 for value in values),
            "worsened_query_count": sum(value < 0.0 for value in values),
            "query_median": float(np.median(np.asarray(values, dtype=np.float64))) if values else None,
        }
    reliability_by_frame = {row["physical_frame_id"]: row for row in reliability}
    lost_ids = {frame_id for _, frame_id in lost_frames}
    loss_reliability: dict[str, Any] = {}
    for metric in ("log_ratio_mad", "log_ratio_q95_abs_deviation", "tile_median_log_ratio_iqr"):
        lost = [float(reliability_by_frame[frame_id][metric]) for frame_id in lost_ids if reliability_by_frame[frame_id][metric] is not None]
        retained = [float(row[metric]) for row in reliability if row["physical_frame_id"] not in lost_ids and row[metric] is not None]
        loss_reliability[metric] = {
            "lost_frame_count": len(lost),
            "lost_frame_median": float(np.median(np.asarray(lost, dtype=np.float64))) if lost else None,
            "other_frame_count": len(retained),
            "other_frame_median": float(np.median(np.asarray(retained, dtype=np.float64))) if retained else None,
        }
    return _seal(
        {
            "schema": "blindassist.taro.o0r.source_anchored_factor_reconciled_diagnostics.v1",
            "analysis_kind": source_factor.ANALYSIS_KIND,
            "claim_ceiling": source_factor.CLAIM_CEILING,
            "failure_code_counts": dict(sorted(failures.items())),
            "extraction_lost_frames": [
                {"parent_id": parent_id, "physical_frame_id": frame_id, "lost_query_count": count}
                for (parent_id, frame_id), count in sorted(lost_frames.items())
            ],
            "effect_query_counts": effect_rows,
            "reliability_by_extraction_loss": loss_reliability,
            "abstention_threshold_selected": False,
            "threshold_or_pass_fail_decision_applied": False,
        }
    )


def execute() -> dict[str, Any]:
    require(ORIGINAL_ROOT.is_dir() and not RECONCILED_ROOT.exists(), "RECONCILIATION_ROOT_PREFLIGHT_INVALID", "original root must exist and reconciliation root must be absent")
    original_manifest, original_manifest_bytes = _verify_original_manifest()
    records = _load_json_gzip(ORIGINAL_ROOT / "query-records.json.gz")
    reliability = _load_json_gzip(ORIGINAL_ROOT / "reliability-records.json.gz")
    validated_records = [source_factor.validate_query_record(row) for row in records]
    validated_reliability = [source_factor.validate_reliability_record(row) for row in reliability]
    require(len(validated_records) == 1539 and len(validated_reliability) == 171, "RECONCILIATION_COHORT_DRIFT", "canonical R1 aggregate cohort drift")

    frame_paths = sorted((ORIGINAL_ROOT / "frame-canary").rglob("*.json.gz"), key=lambda path: path.as_posix())
    frame_hashes: list[str] = []
    for path in frame_paths:
        frame = _load_json_gzip(path)
        require(frame.get("query_count") == 9 and len(frame.get("query_records", [])) == 9, "FRAME_CANARY_CARDINALITY_DRIFT", "original frame canary query count drift", path=str(path))
        frame_hashes.extend(row["content_sha256"] for row in frame["query_records"])
    aggregate_hashes = [row["content_sha256"] for row in validated_records]
    require(len(frame_paths) == 171 and Counter(frame_hashes) == Counter(aggregate_hashes), "FRAME_AGGREGATE_BINDING_DRIFT", "per-frame and aggregate query records differ")

    original_summary = _load_json(ORIGINAL_ROOT / "summary.json")
    reconciled_summary = source_factor.summarize_source_anchored_canary(validated_records, validated_reliability)
    stable_summary = source_factor.summarize_source_anchored_canary(
        json.loads(adapter.canonical_json_bytes(validated_records).decode("utf-8")),
        json.loads(adapter.canonical_json_bytes(validated_reliability).decode("utf-8")),
    )
    require(reconciled_summary["content_sha256"] == stable_summary["content_sha256"], "RECONCILED_SUMMARY_NOT_STABLE", "reconciled summary is not stable after canonical round trip")
    differences = _numeric_differences(original_summary, reconciled_summary)
    maximum_difference = max((row["absolute_difference"] for row in differences), default=0.0)
    require(maximum_difference <= ROUNDING_TOLERANCE, "ORIGINAL_SUMMARY_MATERIAL_DRIFT", "original summary differs by more than canonical rounding tolerance", maximum_difference=maximum_difference)
    require(original_summary["content_sha256"] != reconciled_summary["content_sha256"], "RECONCILIATION_NOT_REQUIRED", "original summary is already canonical-round-trip stable")

    reliability_association = source_factor.summarize_reliability_association(validated_reliability, _oracle_frame_errors())
    original_reliability = _load_json(ORIGINAL_ROOT / "reliability-association.json")
    require(reliability_association["content_sha256"] == original_reliability["content_sha256"], "RELIABILITY_ASSOCIATION_DRIFT", "R1 reliability association is not reproducible")
    diagnostics = _diagnostics(validated_records, validated_reliability)
    original_result = _load_json(ORIGINAL_ROOT / "result.json")

    writer = FactorEvidenceWriter(RECONCILED_ROOT, MAXIMUM_EVIDENCE_BYTES)
    writer.activate(
        {
            "schema": "blindassist.taro.o0r.source_anchored_factor_r1a_reconciliation_start.v1",
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "original_root": ORIGINAL_ROOT.relative_to(REPO_ROOT).as_posix(),
            "original_manifest_sha256": materializer.sha256_file(ORIGINAL_ROOT / "manifest.json"),
            "original_result_sha256": materializer.sha256_file(ORIGINAL_ROOT / "result.json"),
            "source_factor_code_sha256": materializer.sha256_file(Path(source_factor.__file__).resolve()),
            "reconciliation_code_sha256": materializer.sha256_file(Path(__file__).resolve()),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "geometry_recomputed": False,
            "model_inference": False,
            "training": False,
            "network": False,
        }
    )
    validation = _seal(
        {
            "schema": "blindassist.taro.o0r.source_anchored_factor_r1a_validation.v1",
            "original_manifest_sha256": materializer.sha256_file(ORIGINAL_ROOT / "manifest.json"),
            "original_manifest_file_count": original_manifest["file_count_before_manifest"],
            "original_manifest_bytes_before_manifest": original_manifest_bytes,
            "validated_query_record_count": len(validated_records),
            "validated_reliability_record_count": len(validated_reliability),
            "validated_frame_record_count": len(frame_paths),
            "per_frame_aggregate_hash_multiset_equal": True,
            "original_summary_sha256": original_summary["content_sha256"],
            "reconciled_summary_sha256": reconciled_summary["content_sha256"],
            "numeric_difference_count": len(differences),
            "maximum_absolute_numeric_difference": maximum_difference,
            "rounding_tolerance": ROUNDING_TOLERANCE,
            "all_differences_within_canonical_rounding_tolerance": True,
            "reliability_association_reproduced_exactly": True,
            "original_query_and_frame_evidence_valid": True,
            "original_derived_summary_round_trip_stable": False,
            "original_summary_superseded_for_derived_metrics": True,
            "geometry_recomputed": False,
            "algorithm_outputs_changed": False,
            "differences": differences,
        }
    )
    writer.write_json("reconciled-summary.json", reconciled_summary)
    writer.write_json("reliability-association.json", reliability_association)
    writer.write_json("diagnostics.json", diagnostics)
    writer.write_json("validation.json", validation)
    result = {
        "schema": "blindassist.taro.o0r.source_anchored_factor_canary_r1a_reconciliation_result.v1",
        "terminal": "TARO_O0R_SOURCE_ANCHORED_FACTOR_CANARY_R1A_RECONCILED_COMPLETE",
        "execution_valid": True,
        "scientific_status": original_result["scientific_status"],
        "claim_ceiling": source_factor.CLAIM_CEILING,
        "original_result_terminal": original_result["terminal"],
        "original_query_and_frame_evidence_valid": True,
        "original_summary_superseded": True,
        "reconciled_summary_sha256": reconciled_summary["content_sha256"],
        "validation_sha256": validation["content_sha256"],
        "diagnostics_sha256": diagnostics["content_sha256"],
        "physical_frame_count": 171,
        "query_record_count": 1539,
        "parent_count": 16,
        "geometry_recomputed": False,
        "gpu_inference_count": 0,
        "training_steps": 0,
        "network_requests": 0,
        "formal_o0r_pass_authorized": False,
        "threshold_or_pass_fail_decision_applied": False,
    }
    writer.write_json("result.json", result)
    files = dict(sorted(writer.file_receipts.items()))
    writer.write_json(
        "manifest.json",
        {
            "schema": "blindassist.taro.o0r.source_anchored_factor_r1a_reconciliation_manifest.v1",
            "files": files,
            "file_count_before_manifest": len(files),
            "bytes_before_manifest": sum(int(row["bytes"]) for row in files.values()),
            "one_shot_root_consumed": True,
        },
    )
    return result


def main() -> int:
    try:
        result = execute()
    except Exception as error:
        print(json.dumps({"terminal": "SOURCE_FACTOR_R1A_RECONCILIATION_FAILED", "error_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"terminal": result["terminal"], "summary_sha256": result["reconciled_summary_sha256"], "query_records": result["query_record_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
