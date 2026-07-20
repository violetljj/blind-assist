#!/usr/bin/env python3
"""Discover static-semantic exit windows in licensed public walking videos.

This is a proposal generator, not an annotator. It samples each registered
video at a fixed interval, applies the frozen prompt-free YOLOE vocabulary, and
reports adjacent samples where a preregistered static-semantic group changes
from present to absent. Candidate windows still require independent visual
review before they may become provisional public-silver episodes.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2

import run_public_silver_frozen_feature_probe as common
import run_public_silver_prompt_free_semantic_probe as semantic
import run_public_silver_risk_lifecycle_mil_head as mil
import public_video_tristate_contract as prospective


SCHEMA = "blindassist_public_video_prompt_free_exit_discovery_v1"
REGISTRY_SCHEMA = "blindassist_public_video_discovery_registry_v1"
REGISTRY_SCHEMA_V2 = "blindassist_public_video_discovery_registry_v2"
REQUIRED_COMMON_SOURCE_FIELDS = {
    "source_id",
    "local_video_path",
    "author",
    "license",
}
REQUIRED_V1_SOURCE_FIELDS = {"commons_title", "commons_page_url"}
REQUIRED_V2_SOURCE_FIELDS = {"source_platform", "source_title", "source_page_url"}
WORKZONE_MARKER_ADDITIONS = prospective.WORKZONE_MARKER_ADDITIONS


def semantic_group_for_class(
    class_name: str, *, include_workzone_markers: bool = False
) -> str | None:
    group = semantic.semantic_group(class_name)
    if group is not None:
        return group
    if include_workzone_markers and class_name.strip().lower() in WORKZONE_MARKER_ADDITIONS:
        return "barrier_structure"
    return None


def validate_registry(registry: dict[str, Any], registry_path: Path) -> list[dict[str, Any]]:
    schema = registry.get("schema")
    if schema not in {REGISTRY_SCHEMA, REGISTRY_SCHEMA_V2}:
        raise ValueError("unexpected source registry schema")
    raw_sources = registry.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("source registry must contain at least one source")
    sources: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for raw in raw_sources:
        required_source_fields = (
            REQUIRED_V1_SOURCE_FIELDS if schema == REGISTRY_SCHEMA else REQUIRED_V2_SOURCE_FIELDS
        )
        if (
            not isinstance(raw, dict)
            or not REQUIRED_COMMON_SOURCE_FIELDS.issubset(raw)
            or not required_source_fields.issubset(raw)
        ):
            raise ValueError("source registry entry is incomplete")
        source_id = raw["source_id"]
        if not isinstance(source_id, str) or not source_id or source_id in source_ids:
            raise ValueError(f"invalid or duplicate source ID: {source_id!r}")
        source_ids.add(source_id)
        source = dict(raw)
        if schema == REGISTRY_SCHEMA:
            source.setdefault("source_platform", "wikimedia_commons")
            source.setdefault("source_title", source["commons_title"])
            source.setdefault("source_page_url", source["commons_page_url"])
        video_path = Path(source["local_video_path"])
        if not video_path.is_absolute():
            video_path = (registry_path.parent / video_path).resolve()
        mil.reject_independent_direction(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        source["local_video_path"] = str(video_path)
        sources.append(source)
    return sources


def nearfield_corridor_geometry(
    xyxy: Sequence[float], orig_shape: Sequence[int]
) -> dict[str, float | bool]:
    if len(xyxy) != 4 or len(orig_shape) < 2:
        raise ValueError("box and image shape are invalid")
    height, width = float(orig_shape[0]), float(orig_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    x1n, x2n = max(0.0, x1 / width), min(1.0, x2 / width)
    y1n, y2n = max(0.0, y1 / height), min(1.0, y2 / height)
    center_x = (x1n + x2n) / 2.0
    bottom_y = y2n
    area_ratio = max(0.0, x2n - x1n) * max(0.0, y2n - y1n)
    corridor_half_width = min(
        0.45,
        0.20 + 0.25 * max(0.0, bottom_y - 0.62) / 0.38,
    )
    accepted = bool(
        bottom_y >= 0.62
        and abs(center_x - 0.5) <= corridor_half_width
        and area_ratio <= 0.45
    )
    return {
        "center_x_norm": center_x,
        "bottom_y_norm": bottom_y,
        "area_ratio": area_ratio,
        "corridor_half_width": corridor_half_width,
        "nearfield_corridor_accepted": accepted,
    }


def semantic_rows(
    model: Any,
    result: Any,
    *,
    require_nearfield_corridor: bool = False,
    include_workzone_markers: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if result.boxes is None:
        return rows
    for score, class_id, xyxy in zip(
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.xyxy.cpu().numpy(),
    ):
        class_name = str(model.names[int(class_id)])
        group = semantic_group_for_class(
            class_name,
            include_workzone_markers=include_workzone_markers,
        )
        if group is not None:
            geometry = nearfield_corridor_geometry(xyxy.tolist(), result.orig_shape)
            if require_nearfield_corridor and not geometry["nearfield_corridor_accepted"]:
                continue
            rows.append({
                "class_name": class_name,
                "semantic_group": group,
                "confidence": float(score),
                "geometry": geometry,
            })
    return rows


def summarize_sample(timestamp_ms: int, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(str(row["class_name"]) for row in rows)
    group_counts = Counter(str(row["semantic_group"]) for row in rows)
    group_max_confidence: dict[str, float] = {}
    for row in rows:
        group = str(row["semantic_group"])
        group_max_confidence[group] = max(
            group_max_confidence.get(group, 0.0),
            float(row["confidence"]),
        )
    return {
        "timestamp_ms": int(timestamp_ms),
        "semantic_detection_count": len(rows),
        "semantic_class_counts": dict(sorted(class_counts.items())),
        "semantic_group_counts": dict(sorted(group_counts.items())),
        "semantic_group_max_confidence": dict(sorted(group_max_confidence.items())),
    }


def source_scan_range(source: dict[str, Any], duration_ms: int) -> tuple[int, int]:
    start_ms = int(source.get("scan_start_ms", 0))
    end_ms = int(source.get("scan_end_ms", duration_ms))
    if start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
        raise ValueError(
            f"invalid source scan range [{start_ms}, {end_ms}) for duration {duration_ms}"
        )
    return start_ms, end_ms


def discover_adjacent_exits(
    source_id: str,
    samples: Sequence[dict[str, Any]],
    *,
    sample_interval_ms: int,
    min_absent_run_samples: int = 1,
) -> list[dict[str, Any]]:
    if min_absent_run_samples <= 0:
        raise ValueError("minimum absent run must be positive")
    candidates: list[dict[str, Any]] = []
    for index, (previous, current) in enumerate(zip(samples, samples[1:])):
        gap = int(current["timestamp_ms"]) - int(previous["timestamp_ms"])
        if gap <= 0 or gap > sample_interval_ms:
            continue
        previous_counts = previous["semantic_group_counts"]
        current_counts = current["semantic_group_counts"]
        for group in semantic.SEMANTIC_GROUPS:
            absent_run = 0
            for future in samples[index + 1:]:
                if int(future["semantic_group_counts"].get(group, 0)) > 0:
                    break
                absent_run += 1
            if (
                int(previous_counts.get(group, 0)) > 0
                and int(current_counts.get(group, 0)) == 0
                and absent_run >= min_absent_run_samples
            ):
                candidates.append({
                    "source_id": source_id,
                    "semantic_group": group,
                    "present_timestamp_ms": int(previous["timestamp_ms"]),
                    "absent_timestamp_ms": int(current["timestamp_ms"]),
                    "gap_ms": gap,
                    "absent_run_sample_count": absent_run,
                    "minimum_absent_run_samples": min_absent_run_samples,
                    "present_class_counts": previous["semantic_class_counts"],
                    "present_group_max_confidence": float(
                        previous["semantic_group_max_confidence"].get(group, 0.0)
                    ),
                })
    return candidates


def scan_source(
    model: Any,
    source: dict[str, Any],
    *,
    sample_interval_ms: int,
    image_size: int,
    confidence: float,
    min_absent_run_samples: int,
    require_nearfield_corridor: bool,
    include_workzone_markers: bool,
) -> dict[str, Any]:
    video_path = Path(source["local_video_path"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"video timing metadata is invalid: {video_path}")
    duration_ms = int(round((frame_count / fps) * 1000.0))
    scan_start_ms, scan_end_ms = source_scan_range(source, duration_ms)
    samples: list[dict[str, Any]] = []
    try:
        for timestamp_ms in range(scan_start_ms, scan_end_ms, sample_interval_ms):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            predictions = model.predict(
                frame,
                imgsz=image_size,
                conf=confidence,
                max_det=100,
                verbose=False,
            )
            if len(predictions) != 1:
                raise RuntimeError("prompt-free detector must return one result per sampled frame")
            samples.append(summarize_sample(
                timestamp_ms,
                semantic_rows(
                    model,
                    predictions[0],
                    require_nearfield_corridor=require_nearfield_corridor,
                    include_workzone_markers=include_workzone_markers,
                ),
            ))
    finally:
        capture.release()
    if len(samples) < 2:
        raise RuntimeError(f"too few decodable samples: {video_path}")
    candidates = discover_adjacent_exits(
        source["source_id"],
        samples,
        sample_interval_ms=sample_interval_ms,
        min_absent_run_samples=min_absent_run_samples,
    )
    return {
        **source,
        "video_sha256": common.sha256_file(video_path),
        "video_fps": fps,
        "video_frame_count": frame_count,
        "video_duration_ms": duration_ms,
        "scan_start_ms": scan_start_ms,
        "scan_end_ms": scan_end_ms,
        "sample_count": len(samples),
        "samples": samples,
        "exit_candidates": candidates,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.source_registry, args.weights, args.cache_dir, args.output]
    if args.contract is not None:
        paths.append(args.contract)
    for path in paths:
        mil.reject_independent_direction(path)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    registry = common.load_json(args.source_registry)
    sources = validate_registry(registry, args.source_registry.resolve())
    contract_attestation: dict[str, str] | None = None
    if args.contract is not None:
        contract, contract_attestation = prospective.load_contract(args.contract)
        prospective.validate_scan_binding(
            contract,
            weights_sha256=common.sha256_file(args.weights),
            sample_interval_ms=args.sample_interval_ms,
            image_size=args.image_size,
            confidence=args.confidence,
            require_nearfield_corridor=args.require_nearfield_corridor,
            include_workzone_markers=args.include_workzone_markers,
        )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.cache_dir.resolve())
    import torch
    import ultralytics
    from ultralytics import YOLOE

    model = YOLOE(str(args.weights))
    results = [
        scan_source(
            model,
            source,
            sample_interval_ms=args.sample_interval_ms,
            image_size=args.image_size,
            confidence=args.confidence,
            min_absent_run_samples=args.min_absent_run_samples,
            require_nearfield_corridor=args.require_nearfield_corridor,
            include_workzone_markers=args.include_workzone_markers,
        )
        for source in sources
    ]
    candidates = [candidate for result in results for candidate in result["exit_candidates"]]
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(args.source_registry.resolve()),
        "source_registry_sha256": common.sha256_file(args.source_registry),
        "prospective_contract": contract_attestation,
        "model": {
            "kind": "YOLOE prompt-free segmentation model with fixed built-in vocabulary",
            "weights": str(args.weights.resolve()),
            "weights_sha256": common.sha256_file(args.weights),
            "image_size": args.image_size,
            "confidence": args.confidence,
            "semantic_class_groups": {
                group: sorted(
                    names | (WORKZONE_MARKER_ADDITIONS if (
                        group == "barrier_structure" and args.include_workzone_markers
                    ) else set())
                )
                for group, names in semantic.SEMANTIC_GROUPS.items()
            },
            "exploratory_workzone_marker_additions_enabled": args.include_workzone_markers,
            "exploratory_workzone_marker_additions": (
                sorted(WORKZONE_MARKER_ADDITIONS) if args.include_workzone_markers else []
            ),
            "text_prompt_used": False,
            "nearfield_corridor_geometry_required": args.require_nearfield_corridor,
            "nearfield_corridor_contract": {
                "minimum_box_bottom_y_norm": 0.62,
                "maximum_box_area_ratio": 0.45,
                "bottom_center_trapezoid_half_width": "0.20 at y=0.62, increasing linearly to 0.45 at y=1.0",
            },
            "runtime": {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "opencv": cv2.__version__,
            },
        },
        "sampling": {
            "sample_interval_ms": args.sample_interval_ms,
            "adjacent_exit_gap_max_ms": args.sample_interval_ms,
            "minimum_consecutive_absent_samples": args.min_absent_run_samples,
        },
        "sources": results,
        "exit_candidates": candidates,
        "summary": {
            "source_count": len(results),
            "sample_count": sum(result["sample_count"] for result in results),
            "exit_candidate_count": len(candidates),
            "exit_candidate_count_by_group": dict(sorted(Counter(
                candidate["semantic_group"] for candidate in candidates
            ).items())),
        },
        "evidence_limit": "Discovery proposals only; no candidate is an event label, training truth, calibration evidence, blind evidence, or production evidence.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("../artifacts.local/cache/ultralytics-public-video-discovery"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--sample-interval-ms", type=int, default=5000)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--min-absent-run-samples", type=int, default=1)
    parser.add_argument("--require-nearfield-corridor", action="store_true")
    parser.add_argument(
        "--include-workzone-markers",
        action="store_true",
        help="Exploratory additive barrier vocabulary; leaves the preregistered baseline unchanged.",
    )
    args = parser.parse_args()
    if (
        args.sample_interval_ms <= 0
        or args.image_size <= 0
        or args.min_absent_run_samples <= 0
        or not 0 < args.confidence < 1
    ):
        parser.error("sample interval, image size, and confidence must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
