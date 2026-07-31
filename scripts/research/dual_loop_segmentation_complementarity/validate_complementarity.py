"""Independently validate a completed image-space complementarity report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CLASS_NAMES = (
    "walkable",
    "boundary_step_curb",
    "obstacle",
    "unknown_nonwalkable",
)
VALIDATION_SCHEMA = "blindassist.dual_loop_segmentation_complementarity_validation.v1"
FORBIDDEN_FRAME_KEYS = {
    "risk",
    "raw_risk",
    "stable_risk",
    "feedback",
    "feedback_triggered",
    "event",
    "risk_event",
}


class ValidationError(ValueError):
    """Raised when a report or frame artifact violates its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValidationError(f"{path}:{line_number}: blank JSONL line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def _resolve_report_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def validate_report(*, repo_root: Path, report_path: Path, frames_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frames = _read_jsonl(frames_path)
    if report.get("schema_version") != "blindassist.dual_loop_segmentation_complementarity_r1.v1":
        raise ValidationError("unexpected report schema")
    if report.get("status") != "COMPLETE_DEVELOPMENT_DIAGNOSTIC":
        raise ValidationError(f"report status is not complete: {report.get('status')!r}")
    if report.get("risk_feedback_event_fields_read") is not False:
        raise ValidationError("report does not declare risk/feedback/event isolation")
    if report.get("fusion_effect_evaluated") is not False:
        raise ValidationError("report claims fusion effect evaluation")
    if report.get("pairing", {}).get("not_evaluable_frame_count") != 0:
        raise ValidationError("report contains NOT_EVALUABLE frames")
    expected_count = int(report["pairing"]["paired_frame_count"])
    if len(frames) != expected_count or len(frames) != int(report["summary"]["frame_count"]):
        raise ValidationError("frame count differs between report and frames artifact")
    if not frames:
        raise ValidationError("frames artifact is empty")

    grid = report["analysis"]["grid"]
    width, height = int(grid["width"]), int(grid["height"])
    total_pixels = width * height
    if total_pixels <= 0:
        raise ValidationError("invalid analysis grid")
    observed_keys: set[tuple[str, int, str]] = set()
    observed_class_totals = {name: 0 for name in CLASS_NAMES}
    previous_timestamp_by_source: dict[str, int] = {}
    per_source_count: dict[str, int] = {}
    for index, row in enumerate(frames):
        key = (str(row["source_id"]), int(row["frame_id"]), str(row["image_sha256"]).lower())
        if key in observed_keys:
            raise ValidationError(f"duplicate frame identity at row {index + 1}: {key}")
        observed_keys.add(key)
        source_id = key[0]
        timestamp = int(row["source_capture_timestamp_ns"])
        if source_id in previous_timestamp_by_source and timestamp <= previous_timestamp_by_source[source_id]:
            raise ValidationError(f"non-increasing timestamp at row {index + 1}")
        previous_timestamp_by_source[source_id] = timestamp
        per_source_count[source_id] = per_source_count.get(source_id, 0) + 1
        if FORBIDDEN_FRAME_KEYS.intersection(row):
            raise ValidationError(f"forbidden risk/feedback/event field at row {index + 1}")
        if int(row["analysis_width"]) != width or int(row["analysis_height"]) != height:
            raise ValidationError(f"analysis grid drift at row {index + 1}")
        detector = row["detector"]
        detector_pixels = int(detector["covered_pixels"])
        detector_fraction = float(detector["coverage_fraction"])
        if not 0 <= detector_pixels <= total_pixels or not 0 <= detector_fraction <= 1:
            raise ValidationError(f"invalid detector coverage at row {index + 1}")
        if not _close(detector_fraction, detector_pixels / total_pixels):
            raise ValidationError(f"detector fraction arithmetic mismatch at row {index + 1}")
        segmentation = row["segmentation"]
        if set(segmentation) != set(CLASS_NAMES):
            raise ValidationError(f"class set mismatch at row {index + 1}")
        class_pixels_sum = 0
        for class_name in CLASS_NAMES:
            item = segmentation[class_name]
            pixels = int(item["pixels"])
            uncovered_pixels = int(item["uncovered_pixels"])
            fraction = float(item["fraction"])
            uncovered_fraction = float(item["uncovered_fraction"])
            if pixels < 0 or uncovered_pixels < 0 or uncovered_pixels > pixels:
                raise ValidationError(f"invalid {class_name} pixels at row {index + 1}")
            if not _close(fraction, pixels / total_pixels):
                raise ValidationError(f"{class_name} fraction arithmetic mismatch at row {index + 1}")
            if not _close(uncovered_fraction, uncovered_pixels / total_pixels):
                raise ValidationError(f"{class_name} uncovered arithmetic mismatch at row {index + 1}")
            iou = item["temporal_iou"]
            if iou is not None and not 0 <= float(iou) <= 1:
                raise ValidationError(f"{class_name} temporal IoU out of range at row {index + 1}")
            class_pixels_sum += pixels
            observed_class_totals[class_name] += pixels
        if class_pixels_sum != total_pixels:
            raise ValidationError(f"class masks do not partition grid at row {index + 1}")
        fusion = row["fusion_geometry"]
        if int(fusion["all_class_union_pixels"]) != total_pixels:
            raise ValidationError(f"all-class union is not full grid at row {index + 1}")
        if not _close(float(fusion["all_class_union_fraction"]), 1.0):
            raise ValidationError(f"all-class union fraction mismatch at row {index + 1}")
        expected_increment_pixels = total_pixels - detector_pixels
        expected_increment_fraction = expected_increment_pixels / total_pixels
        if int(fusion["union_increment_pixels"]) != expected_increment_pixels:
            raise ValidationError(f"union increment pixel arithmetic mismatch at row {index + 1}")
        if not _close(float(fusion["union_increment_fraction"]), expected_increment_fraction):
            raise ValidationError(f"union increment fraction arithmetic mismatch at row {index + 1}")
        if fusion.get("all_class_union_covers_grid_by_construction") is not True:
            raise ValidationError("missing all-class union construction note")

    report_totals = {name: int(report["class_pixel_totals"][name]) for name in CLASS_NAMES}
    if report_totals != observed_class_totals:
        raise ValidationError("class pixel totals differ between report and frames artifact")
    if not report["stop_checks"]["pairing_pass"] or not report["stop_checks"]["finite_output_pass"]:
        raise ValidationError("report stop checks do not pass")
    if report["stop_checks"]["single_class_collapse"]:
        raise ValidationError("report declares single-class collapse")

    for artifact_name in ("manifest", "trace", "model"):
        artifact = report[artifact_name]
        path = _resolve_report_path(repo_root, artifact["path"])
        if not path.is_file():
            raise ValidationError(f"missing {artifact_name}: {path}")
        if sha256_file(path) != str(artifact["sha256"]).lower():
            raise ValidationError(f"{artifact_name} hash mismatch: {path}")
    progress_path = _resolve_report_path(repo_root, report["artifacts"]["progress"])
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("status") != "COMPLETE" or int(progress.get("completed_frames", -1)) != expected_count:
            raise ValidationError("progress receipt is not complete")

    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "VALID",
        "evidence_instance": report["evidence_instance"],
        "report_sha256": sha256_file(report_path),
        "frames_sha256": sha256_file(frames_path),
        "paired_frame_count": expected_count,
        "source_session_counts": per_source_count,
        "class_pixel_totals": observed_class_totals,
        "risk_feedback_event_fields_present": False,
        "all_class_union_arithmetic_valid": True,
        "timestamp_order_valid": True,
        "input_hashes_valid": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path.cwd().resolve()
    report_path = _resolve_report_path(repo_root, str(args.report))
    frames_path = _resolve_report_path(repo_root, str(args.frames))
    output_path = _resolve_report_path(repo_root, str(args.output))
    artifacts_root = (repo_root / "artifacts.local").resolve()
    try:
        output_path.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValidationError("validation output must stay under artifacts.local") from exc
    result = validate_report(repo_root=repo_root, report_path=report_path, frames_path=frames_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": output_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
