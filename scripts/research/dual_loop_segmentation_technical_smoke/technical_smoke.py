#!/usr/bin/env python3
"""Run one non-effect semantic-segmentation interface smoke on fixed RGB slots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


CLASS_NAMES = (
    "walkable",
    "boundary_step_curb",
    "obstacle",
    "unknown_nonwalkable",
)
CLASS_COLORS = (
    (38, 166, 91, 86),
    (255, 193, 7, 120),
    (229, 57, 53, 140),
    (123, 31, 162, 130),
)
REQUIRED_OBSERVATION_FIELDS = ("unit_id", "session_id", "slot_ordinal", "review_image_path")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_input_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo_root / candidate


def ensure_artifact_path(repo_root: Path, value: Path) -> Path:
    artifacts_root = (repo_root / "artifacts.local").resolve()
    resolved = value.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(f"output must stay under artifacts.local: {resolved}") from exc
    return resolved


def load_observations(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixed_units = payload.get("fixed_units")
    if not isinstance(fixed_units, list) or not fixed_units:
        raise ValueError("manifest must contain non-empty fixed_units")

    observations: list[dict[str, Any]] = []
    for unit in fixed_units:
        if not isinstance(unit, dict):
            raise ValueError("fixed_units entries must be objects")
        unit_id = unit.get("unit_id")
        unit_observations = unit.get("observations")
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError("each fixed unit needs a unit_id")
        if not isinstance(unit_observations, list) or not unit_observations:
            raise ValueError(f"{unit_id}: observations must be non-empty")
        for observation in unit_observations:
            if not isinstance(observation, dict):
                raise ValueError(f"{unit_id}: observation must be an object")
            missing = [field for field in REQUIRED_OBSERVATION_FIELDS if field not in observation]
            if missing:
                raise ValueError(f"{unit_id}: missing observation fields {missing}")
            if observation["unit_id"] != unit_id:
                raise ValueError(f"{unit_id}: observation unit_id mismatch")
            if observation.get("candidate_output_visible") is not False:
                raise ValueError(f"{unit_id}: candidate output visibility must be false")
            if observation.get("prior_review_visible") is not False:
                raise ValueError(f"{unit_id}: prior review visibility must be false")
            observations.append(
                {
                    "unit_id": unit_id,
                    "session_id": str(observation["session_id"]),
                    "slot_ordinal": int(observation["slot_ordinal"]),
                    "review_image_path": str(observation["review_image_path"]),
                }
            )
    return observations


def _scalar_quantization(detail: dict[str, Any], label: str) -> tuple[float, int]:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError(f"{label}: positive scalar quantization scale required")
    return float(scale), int(zero_point)


def _prepare_int8_rgb(image: Image.Image, shape: tuple[int, ...], scale: float, zero_point: int) -> np.ndarray:
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
        raise ValueError(f"input tensor must be [1,H,W,3], got {shape}")
    resized = image.convert("RGB").resize((shape[2], shape[1]), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32)
    return np.clip(np.rint(rgb / scale + zero_point), -128, 127).astype(np.int8)[None, ...]


def _dequantize(raw: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(raw.dtype, np.integer):
        scale, zero_point = _scalar_quantization(detail, "output")
        return (raw.astype(np.float32) - zero_point) * scale
    return raw.astype(np.float32)


def class_distribution(class_counts: Iterable[int], total_classes: int) -> list[dict[str, Any]]:
    counts = [int(value) for value in class_counts]
    total = sum(counts)
    rows = []
    for class_id, count in enumerate(counts):
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        rows.append(
            {
                "class_id": class_id,
                "name": name,
                "pixels": count,
                "fraction": (count / total) if total else 0.0,
            }
        )
    if total_classes != len(counts):
        raise ValueError("total_classes must equal class_counts length")
    return rows


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "clip"


def _save_contact_sheets(
    observations: list[dict[str, Any]],
    sample_outputs: list[dict[str, Any]],
    repo_root: Path,
    output_dir: Path,
) -> list[str]:
    by_unit: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for observation, output in zip(observations, sample_outputs):
        by_unit[observation["unit_id"]].append((observation, output))

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for unit_id, items in by_unit.items():
        cell_width, cell_height = 320, 240
        columns = 2
        rows = max(1, math.ceil(len(items) / columns))
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
        for index, (observation, output) in enumerate(sorted(items, key=lambda item: item[0]["slot_ordinal"])):
            source = resolve_input_path(repo_root, observation["review_image_path"])
            image = Image.open(source).convert("RGB").resize((cell_width, cell_height), Image.Resampling.BILINEAR)
            labels = Image.fromarray(output["labels"].astype(np.uint8), mode="L").resize(
                (cell_width, cell_height), Image.Resampling.NEAREST
            )
            labels_array = np.asarray(labels, dtype=np.uint8)
            overlay = Image.new("RGBA", (cell_width, cell_height), (0, 0, 0, 0))
            overlay_array = np.zeros((cell_height, cell_width, 4), dtype=np.uint8)
            for class_id, color in enumerate(CLASS_COLORS):
                overlay_array[labels_array == class_id] = color
            overlay = Image.fromarray(overlay_array, mode="RGBA")
            cell = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            sheet.paste(cell, ((index % columns) * cell_width, (index // columns) * cell_height))
        output_path = output_dir / f"clip-{_safe_filename(unit_id)}.png"
        sheet.save(output_path)
        paths.append(output_path.as_posix())
    return paths


def run_smoke(
    *,
    repo_root: Path,
    manifest_path: Path,
    model_path: Path,
    output_path: Path,
    visualization_dir: Path | None,
    threads: int,
) -> dict[str, Any]:
    if threads <= 0:
        raise ValueError("threads must be positive")
    observations = load_observations(manifest_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    for observation in observations:
        image_path = resolve_input_path(repo_root, observation["review_image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("TensorFlow is required for the TFLite technical smoke") from exc

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_shape = tuple(int(value) for value in input_detail["shape"])
    output_shape = tuple(int(value) for value in output_detail["shape"])
    input_dtype = np.dtype(input_detail["dtype"])
    output_dtype = np.dtype(output_detail["dtype"])
    input_scale, input_zero = _scalar_quantization(input_detail, "input")
    output_scale, output_zero = _scalar_quantization(output_detail, "output")
    interface_checks = {
        "input_shape_is_nhwc_rgb": len(input_shape) == 4 and input_shape[0] == 1 and input_shape[3] == 3,
        "input_dtype_is_int8": input_dtype == np.dtype(np.int8),
        "output_shape_is_nhwc": len(output_shape) == 4 and output_shape[0] == 1,
        "output_dtype_is_int8": output_dtype == np.dtype(np.int8),
        "output_has_four_classes": len(output_shape) == 4 and output_shape[3] == len(CLASS_NAMES),
    }
    if not all(interface_checks.values()):
        raise ValueError(f"interface contract failed: {interface_checks}")

    durations_ms: list[float] = []
    class_counts = np.zeros(output_shape[3], dtype=np.int64)
    finite_values = True
    sample_outputs: list[dict[str, Any]] = []
    for observation in observations:
        source = resolve_input_path(repo_root, observation["review_image_path"])
        with Image.open(source) as image:
            tensor = _prepare_int8_rgb(image, input_shape, input_scale, input_zero)
        interpreter.set_tensor(input_detail["index"], tensor)
        started = time.perf_counter()
        interpreter.invoke()
        durations_ms.append(float((time.perf_counter() - started) * 1000.0))
        raw = interpreter.get_tensor(output_detail["index"])
        values = _dequantize(raw, output_detail)
        finite_values = finite_values and bool(np.isfinite(values).all())
        labels = np.argmax(raw, axis=-1)[0]
        counts = np.bincount(labels.reshape(-1), minlength=output_shape[3])
        class_counts += counts
        sample_outputs.append(
            {
                "labels": labels,
                "dominant_class_id": int(np.argmax(counts)),
                "dominant_fraction": float(np.max(counts) / labels.size),
                "class_pixels": [int(value) for value in counts],
                "inference_ms": durations_ms[-1],
            }
        )

    visualization_paths: list[str] = []
    if visualization_dir is not None:
        visualization_paths = _save_contact_sheets(observations, sample_outputs, repo_root, visualization_dir)

    total_pixels = int(class_counts.sum())
    distribution = class_distribution(class_counts, output_shape[3])
    status = "PASS_INTERFACE_ONLY" if finite_values else "FAIL_NONFINITE_OUTPUT"
    warnings: list[str] = []
    if distribution[0]["fraction"] == 1.0:
        warnings.append("ARGMAX_COLLAPSED_TO_WALKABLE_ON_SMOKE_INPUT")
    if sum(row["pixels"] for row in distribution[1:]) == 0:
        warnings.append("NO_NON_WALKABLE_ARGMAX_OUTPUT_ON_SMOKE_INPUT")

    return {
        "schema_version": 1,
        "evidence_instance": "DUAL_LOOP_SEGMENTATION_TECHNICAL_SMOKE_R0",
        "status": status,
        "authority": "TECHNICAL_SMOKE_ONLY",
        "claim_ceiling": "interface_plausibility_and_output_diagnostic_only",
        "central_obstruction_truth_read": False,
        "d0_a_readiness_contribution": False,
        "d0_b_execution_authorized": False,
        "candidate_selection_performed": False,
        "model_comparison_performed": False,
        "fusion_evaluated": False,
        "device_latency_measured": False,
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
            "role": "excluded_rgb_technical_input_only",
            "observation_count": len(observations),
            "fixed_clip_count": len({row["unit_id"] for row in observations}),
            "source_sessions": sorted({row["session_id"] for row in observations}),
        },
        "model": {
            "path": model_path.as_posix(),
            "sha256": sha256_file(model_path),
            "bytes": model_path.stat().st_size,
            "reference_name": "MobileNetV3Small(alpha=0.75)+LR-ASPP",
        },
        "interface": {
            "checks": interface_checks,
            "input": {
                "shape": list(input_shape),
                "dtype": str(input_dtype),
                "quantization": {"scale": input_scale, "zero_point": input_zero},
            },
            "output": {
                "shape": list(output_shape),
                "dtype": str(output_dtype),
                "quantization": {"scale": output_scale, "zero_point": output_zero},
                "finite_dequantized_values": finite_values,
            },
        },
        "argmax_class_distribution": {
            "total_pixels": total_pixels,
            "classes": distribution,
        },
        "host_runtime_ms": {
            "p50": float(np.percentile(durations_ms, 50)),
            "p95": float(np.percentile(durations_ms, 95)),
            "max": float(max(durations_ms)),
            "threads": threads,
            "note": "host-only TensorFlow Lite timing; not a phone or Snapdragon latency claim",
        },
        "warnings": warnings,
        "visualizations": visualization_paths,
        "sample_summaries": [
            {
                "unit_id": observation["unit_id"],
                "session_id": observation["session_id"],
                "slot_ordinal": observation["slot_ordinal"],
                "dominant_class_id": output["dominant_class_id"],
                "dominant_fraction": output["dominant_fraction"],
                "class_pixels": output["class_pixels"],
                "inference_ms": output["inference_ms"],
            }
            for observation, output in zip(observations, sample_outputs)
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path.cwd().resolve()
    manifest_path = resolve_input_path(repo_root, str(args.manifest)).resolve()
    model_path = resolve_input_path(repo_root, str(args.model)).resolve()
    output_path = ensure_artifact_path(repo_root, resolve_input_path(repo_root, str(args.output)))
    visualization_dir = None
    if args.visualization_dir is not None:
        visualization_dir = ensure_artifact_path(repo_root, resolve_input_path(repo_root, str(args.visualization_dir)))
    report = run_smoke(
        repo_root=repo_root,
        manifest_path=manifest_path,
        model_path=model_path,
        output_path=output_path,
        visualization_dir=visualization_dir,
        threads=args.threads,
    )
    write_report(output_path, report)
    print(json.dumps({"status": report["status"], "report": output_path.as_posix(), "warnings": report["warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
