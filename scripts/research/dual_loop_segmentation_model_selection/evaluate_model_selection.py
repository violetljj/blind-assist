"""Run the frozen R1 source-native model-selection evaluation.

The pixel, component, temporal, and TFLite primitives remain shared with the
closed R0 utility evaluator.  This adapter binds those primitives to the R1
protocol identity and adds the R1 source/session selection summaries.  It does
not select thresholds, read event labels, or create alert authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..dual_loop_segmentation_candidate_utility import evaluate_candidate_utility as _base


PROTOCOL_ID = "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1"
R1_SCHEMA = "blindassist.dual_loop_segmentation_model_selection_r1"
HAZARD_CLASSES = ("boundary_step_curb", "obstacle")
EXPECTED_GRID = {
    "width": 256,
    "height": 256,
    "projection": "nearest for source-native truth; clipped normalized boxes",
}
TEMPORAL_MATCH_IOU = 0.10


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_r1_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unexpected R1 protocol id: {protocol.get('protocol_id')!r}")
    if protocol.get("status") != "DESIGN_FROZEN":
        raise ValueError(f"R1 protocol is not frozen: {protocol.get('status')!r}")
    evaluation = protocol.get("evaluation_contract")
    if not isinstance(evaluation, dict):
        raise ValueError("R1 evaluation_contract is missing")
    if evaluation.get("analysis_grid") != EXPECTED_GRID:
        raise ValueError("R1 analysis grid differs from the frozen contract")
    if evaluation.get("fusion_operator") != (
        "fixed R0 A/B/C operator: A=YOLO-only, B=segmentation hazard, "
        "C=A union B, candidate=B minus A"
    ):
        raise ValueError("R1 fusion operator differs from the frozen contract")
    model_contract = protocol.get("model_contract")
    if not isinstance(model_contract, dict) or model_contract.get("operator", {}).get("hazard_classes") != [
        "boundary_step_curb",
        "obstacle",
    ]:
        raise ValueError("R1 hazard class contract differs from the frozen contract")
    # The R1 JSON deliberately keeps the temporal implementation detail out of
    # the public gate list.  Supply the fixed implementation constant only to
    # the shared temporal primitive; it is not a post-hoc calibration input.
    normalized = dict(protocol)
    normalized["analysis"] = {"grid": EXPECTED_GRID, "temporal_match_iou": TEMPORAL_MATCH_IOU}
    return normalized


def _percentiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _r1_summaries(frame_rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    source_ids = sorted({str(row["source_id"]) for row in frame_rows})
    fragmentation_rows: list[float] = []
    for row in frame_rows:
        components = int(row["candidate_component_metrics"]["predicted_component_count"])
        pixels = int(row["candidate_hazard_pixels"])
        fragmentation_rows.append(float(components / pixels * 1000.0) if pixels else 0.0)
    temporal = report.get("temporal", {})
    temporal_hazard = [
        source_value.get("candidate_hazard", {})
        for source_value in temporal.values()
        if isinstance(source_value, dict) and isinstance(source_value.get("candidate_hazard"), dict)
    ]
    all_tracks = [
        track
        for item in temporal_hazard
        for track in item.get("component_tracks", [])
        if isinstance(track, dict)
    ]
    flicker_count = sum(int(track.get("duration_frames", 0)) <= 2 for track in all_tracks)
    source_consistency: list[dict[str, Any]] = []
    for item in report.get("session_summary", []):
        delta = item.get("delta_recall")
        source_consistency.append(
            {
                "source_id": item["source_id"],
                "frame_count": item["frame_count"],
                "delta_recall_C_minus_A": delta,
                "candidate_component_recall": item.get("candidate_components", {}).get("component_recall"),
                "passes_consistent_session_floor": delta is not None and float(delta) >= 0.0,
            }
        )
    consistent = sum(int(item["passes_consistent_session_floor"]) for item in source_consistency)
    c_minus_a_negative_frames = sum(
        int(
            row.get("arms", {}).get("C", {}).get("pixel", {}).get("recall") is not None
            and row.get("arms", {}).get("A", {}).get("pixel", {}).get("recall") is not None
            and float(row["arms"]["C"]["pixel"]["recall"]) < float(row["arms"]["A"]["pixel"]["recall"])
        )
        for row in frame_rows
    )
    return {
        "mask_fragmentation": {
            "candidate_components_per_frame": _percentiles(
                [float(row["candidate_component_metrics"]["predicted_component_count"]) for row in frame_rows]
            ),
            "candidate_components_per_1000_candidate_pixels": _percentiles(fragmentation_rows),
        },
        "component_stability": {
            "candidate_hazard_temporal_sources": len(temporal_hazard),
            "raw_adjacent_iou_median_by_source": {
                source_id: temporal.get(source_id, {}).get("candidate_hazard", {}).get("raw_adjacent_iou", {}).get("median")
                for source_id in source_ids
            },
            "candidate_birth_count": int(sum(item.get("candidate_birth_count", 0) for item in temporal_hazard)),
            "candidate_death_count": int(sum(item.get("candidate_death_count", 0) for item in temporal_hazard)),
            "flicker_track_count": int(flicker_count),
            "track_count": len(all_tracks),
            "flicker_track_fraction": _ratio(flicker_count, len(all_tracks)),
        },
        "source_wise_metrics": source_consistency,
        "consistent_sessions": {
            "count": int(consistent),
            "denominator": len(source_consistency),
            "minimum_required": 2,
            "minimum_delta_recall": 0.0,
        },
        "critical_miss_guardrail": {
            "definition": "frame-level C recall must not be lower than YOLO-only A recall; this is a pixel guardrail, not an event claim",
            "frames_with_C_recall_below_A": int(c_minus_a_negative_frames),
            "frames_evaluated": len(frame_rows),
            "passed": c_minus_a_negative_frames == 0,
        },
    }


def _patch_base_globals() -> dict[str, Any]:
    names = ("PROTOCOL_ID", "PRIMARY_HAZARD_CLASSES", "SCHEMA_VERSION", "FRAME_SCHEMA_VERSION", "COMPONENT_SCHEMA_VERSION", "load_protocol")
    saved = {name: getattr(_base, name) for name in names}
    _base.PROTOCOL_ID = PROTOCOL_ID
    _base.PRIMARY_HAZARD_CLASSES = HAZARD_CLASSES
    _base.SCHEMA_VERSION = f"{R1_SCHEMA}.result.v1"
    _base.FRAME_SCHEMA_VERSION = f"{R1_SCHEMA}.frame.v1"
    _base.COMPONENT_SCHEMA_VERSION = f"{R1_SCHEMA}.component.v1"
    _base.load_protocol = _load_r1_protocol
    return saved


def _restore_base_globals(saved: dict[str, Any]) -> None:
    for name, value in saved.items():
        setattr(_base, name, value)


def run_evaluation(
    *,
    repo_root: Path,
    protocol_path: Path,
    manifest_path: Path,
    dataset_root: Path | None,
    trace_path: Path,
    model_path: Path,
    report_path: Path,
    frames_path: Path,
    components_path: Path,
    progress_path: Path,
    phase: str,
    split: str | None,
    motion_trace_path: Path | None,
    threads: int,
    frames_limit: int | None,
    progress_every: int,
    model_config_path: Path | None,
    model_receipt_path: Path | None,
    runtime_receipt_path: Path | None,
) -> dict[str, Any]:
    if phase == "calibration" and split != "dev":
        raise ValueError("R1 calibration is bound to the canonical dev split")
    if phase == "formal" and "fresh_holdout" not in str(manifest_path).lower():
        raise ValueError("R1 formal evaluation requires the frozen fresh_holdout manifest")
    if model_config_path is not None and not model_config_path.is_file():
        raise FileNotFoundError(model_config_path)
    if model_receipt_path is not None and not model_receipt_path.is_file():
        raise FileNotFoundError(model_receipt_path)
    if runtime_receipt_path is not None and not runtime_receipt_path.is_file():
        raise FileNotFoundError(runtime_receipt_path)
    saved = _patch_base_globals()
    try:
        report = _base.run_evaluation(
            repo_root=repo_root,
            protocol_path=protocol_path,
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            trace_path=trace_path,
            model_path=model_path,
            report_path=report_path,
            frames_path=frames_path,
            components_path=components_path,
            progress_path=progress_path,
            phase=phase,
            split=split,
            motion_trace_path=motion_trace_path,
            threads=threads,
            frames_limit=frames_limit,
            progress_every=progress_every,
        )
        frame_rows = _read_jsonl(frames_path)
        report["protocol_id"] = PROTOCOL_ID
        report["evidence_instance"] = PROTOCOL_ID
        report["r1_metrics"] = _r1_summaries(frame_rows, report)
        report["selection_identity"] = {
            "model_config": str(model_config_path.resolve()) if model_config_path else None,
            "model_config_sha256": _base.sha256_file(model_config_path) if model_config_path else None,
            "model_receipt": str(model_receipt_path.resolve()) if model_receipt_path else None,
            "model_receipt_sha256": _base.sha256_file(model_receipt_path) if model_receipt_path else None,
            "model_tflite_sha256": report["model_sha256"],
        }
        if runtime_receipt_path is not None:
            runtime_receipt = _read_json(runtime_receipt_path)
            if runtime_receipt.get("protocol_id") != PROTOCOL_ID or runtime_receipt.get("status") != "RUNTIME_BENCHMARK_COMPLETE":
                raise ValueError("runtime receipt is not a completed R1 benchmark")
            if runtime_receipt.get("model_sha256") != report["model_sha256"]:
                raise ValueError("runtime receipt model SHA256 differs from evaluated TFLite")
            report["runtime_benchmark"] = {
                "receipt_path": str(runtime_receipt_path.resolve()),
                "receipt_sha256": _base.sha256_file(runtime_receipt_path),
                "runtime": runtime_receipt.get("runtime"),
                "contract": runtime_receipt.get("runtime_contract"),
            }
        _base._write_json(report_path, report)
        return report
    finally:
        _restore_base_globals(saved)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--runtime-receipt", type=Path)
    parser.add_argument("--phase", choices=("calibration", "formal"), required=True)
    parser.add_argument("--split")
    parser.add_argument("--motion-trace", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--components", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--frames-limit", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report.resolve()
    report = run_evaluation(
        repo_root=args.repo_root.resolve(),
        protocol_path=args.protocol.resolve(),
        manifest_path=args.manifest.resolve(),
        dataset_root=args.dataset_root.resolve() if args.dataset_root else None,
        trace_path=args.trace.resolve(),
        model_path=args.model.resolve(),
        report_path=report_path,
        frames_path=(args.frames or report_path.with_name("frames.jsonl")).resolve(),
        components_path=(args.components or report_path.with_name("components.jsonl")).resolve(),
        progress_path=(args.progress or report_path.with_name("progress.json")).resolve(),
        phase=args.phase,
        split=args.split,
        motion_trace_path=args.motion_trace.resolve() if args.motion_trace else None,
        threads=args.threads,
        frames_limit=args.frames_limit,
        progress_every=args.progress_every,
        model_config_path=args.model_config.resolve() if args.model_config else None,
        model_receipt_path=args.model_receipt.resolve() if args.model_receipt else None,
        runtime_receipt_path=args.runtime_receipt.resolve() if args.runtime_receipt else None,
    )
    print(json.dumps({"status": report["status"], "phase": report["phase"], "frame_count": report["frame_count"], "report_path": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
