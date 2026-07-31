"""Independently recompute all runtime summaries from immutable timing rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import PROTOCOL_ID


STAGES = (
    "preprocess",
    "tflite_inference",
    "output_dequantize_argmax",
    "component_extraction",
    "fusion_operator",
    "total_increment",
)


class RuntimeValidationError(ValueError):
    """Raised when runtime rows cannot reproduce their report."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeValidationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeValidationError(f"blank runtime row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeValidationError(f"expected runtime object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise RuntimeValidationError(f"zero runtime rows: {path}")
    return rows


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0):
        raise RuntimeValidationError("invalid timing values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def validate(
    *,
    runtime_root: Path,
    model_path: Path,
    view_manifest_path: Path,
    trace_path: Path,
    postprocess_path: Path,
    harness_path: Path,
    schema_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    runtime_root = runtime_root.resolve()
    report = _read_json(runtime_root / "report.json")
    rows_path = runtime_root / "runtime_rows.jsonl"
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("status") != "RUNTIME_ROWS_COMPLETE_UNVALIDATED"
        or report.get("formal_authority") is not False
    ):
        raise RuntimeValidationError("runtime report identity/status mismatch")
    identities = {
        "model_sha256": sha256_file(model_path.resolve()),
        "view_manifest_sha256": sha256_file(view_manifest_path.resolve()),
        "trace_sha256": sha256_file(trace_path.resolve()),
        "postprocess_sha256": sha256_file(postprocess_path.resolve()),
        "runtime_harness_sha256": sha256_file(harness_path.resolve()),
        "runtime_row_schema_sha256": sha256_file(schema_path.resolve()),
        "runtime_rows_sha256": sha256_file(rows_path),
    }
    for key, expected in identities.items():
        if report.get(key) != expected:
            raise RuntimeValidationError(f"runtime identity mismatch: {key}")
    contract = report.get("runtime_contract", {})
    measured_frames = int(contract.get("measured_frames", -1))
    if (
        contract.get("threads") != 4
        or contract.get("warmup_frames") != 20
        or measured_frames != 200
        or contract.get("truth_pixels_read") is not False
        or contract.get("timed_stages_ms") != list(STAGES)
    ):
        raise RuntimeValidationError("runtime execution contract mismatch")
    rows = _read_jsonl(rows_path)
    if len(rows) != measured_frames:
        raise RuntimeValidationError("runtime row denominator mismatch")
    identities_seen: set[tuple[str, int, str]] = set()
    indexes: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        if (
            row.get("schema_version")
            != "blindassist.dual_loop_segmentation_r2_p0.runtime_row.v1"
            or row.get("protocol_id") != PROTOCOL_ID
            or row.get("candidate_id") != report.get("candidate_id")
        ):
            raise RuntimeValidationError(f"runtime row {row_number}: identity mismatch")
        key = (
            str(row.get("source_id")),
            int(row.get("frame_id")),
            str(row.get("image_sha256")),
        )
        if key in identities_seen:
            raise RuntimeValidationError(f"runtime row {row_number}: duplicate frame identity")
        identities_seen.add(key)
        indexes.append(int(row.get("measurement_index")))
        stages = row.get("stages_ms")
        if not isinstance(stages, dict) or set(stages) != set(STAGES):
            raise RuntimeValidationError(f"runtime row {row_number}: stage set mismatch")
        if any(
            not math.isfinite(float(stages[name])) or float(stages[name]) < 0
            for name in STAGES
        ):
            raise RuntimeValidationError(f"runtime row {row_number}: invalid timing")
        if float(stages["total_increment"]) < sum(
            float(stages[name]) for name in STAGES if name != "total_increment"
        ):
            raise RuntimeValidationError(
                f"runtime row {row_number}: total increment is smaller than timed stages"
            )
    if indexes != list(range(measured_frames)):
        raise RuntimeValidationError("runtime measurement indexes are not contiguous")
    recomputed = {
        stage: _summary([float(row["stages_ms"][stage]) for row in rows])
        for stage in STAGES
    }
    for stage in STAGES:
        stored = report.get("runtime", {}).get(stage, {})
        for field, expected in recomputed[stage].items():
            actual = stored.get(field)
            if actual is None or not math.isclose(
                float(actual),
                float(expected),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeValidationError(
                    f"runtime {stage}.{field} mismatch: {actual!r} != {expected!r}"
                )
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_r2_p0.runtime_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID",
        "formal_authority": False,
        "candidate_id": report["candidate_id"],
        "row_count": len(rows),
        "frame_identity_count": len(identities_seen),
        "runtime_rows_sha256": identities["runtime_rows_sha256"],
        "independent_recompute": recomputed,
        "truth_pixels_read": False,
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite runtime validation: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output_path.with_suffix(".sha256.json").write_text(
        json.dumps({"sha256": sha256_file(output_path)}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--view-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    value = validate(
        runtime_root=args.runtime_root,
        model_path=args.model,
        view_manifest_path=args.view_manifest,
        trace_path=args.trace,
        postprocess_path=args.postprocess,
        harness_path=args.harness,
        schema_path=args.schema,
        output_path=args.output,
    )
    print(json.dumps({"status": value["status"], "rows": value["row_count"]}))
