"""Develop a semantic door-presence gate on source-disjoint historical data.

The fine-tuned semantic model supplies a new per-pixel information source.  It
does not receive named-place identity, portal truth, or PB5 pixels.  Development
selects one connected-component area threshold from six prior glass-door
positives and sixteen separately frozen no-entrance controls.  A later PB6
confirmation must freeze this result before seeing any new model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(spec: dict[str, str]) -> Path:
    path = _path(spec["path"])
    actual = _sha256(path)
    if actual != spec["sha256"]:
        raise ValueError(f"HASH_MISMATCH:{path}:{actual}:{spec['sha256']}")
    return path


def _member(box: Sequence[float], region: Sequence[float]) -> bool:
    center_x = 0.5 * (box[0] + box[2])
    center_y = 0.5 * (box[1] + box[3])
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    coverage = intersection / area if area else 0.0
    return (
        region[0] <= center_x <= region[2]
        and region[1] <= center_y <= region[3]
        and coverage >= 0.5
    )


def _predict(
    model: YOLO, image_path: Path, device: str | int, image_size: int
) -> tuple[np.ndarray, float, int | None, int | None]:
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = model.predict(
        source=str(image_path), device=device, imgsz=image_size, verbose=False
    )[0]
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    semantic = result.semantic_mask
    if semantic is None:
        raise ValueError(f"SEMANTIC_MASK_MISSING:{image_path}")
    mask = semantic.data.detach().to("cpu").numpy().astype(np.uint8)
    peak_allocated = (
        int(torch.cuda.max_memory_allocated()) if device != "cpu" else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved()) if device != "cpu" else None
    )
    return mask, elapsed, peak_allocated, peak_reserved


def _select_backend(
    model_path: Path, representative: Path, image_size: int
) -> tuple[str | int, YOLO, dict[str, Any]]:
    cpu_model = YOLO(str(model_path))
    _predict(cpu_model, representative, "cpu", image_size)
    _, cpu_seconds, _, _ = _predict(cpu_model, representative, "cpu", image_size)
    benchmarks = [
        {
            "backend": "semantic-door-torch-cpu",
            "actual_device_type": "cpu",
            "seconds": cpu_seconds,
        }
    ]
    selected_device: str | int = "cpu"
    selected_model = cpu_model
    selection_reason = "ACCELERATOR_UNAVAILABLE"
    if torch.cuda.is_available():
        gpu_model = YOLO(str(model_path))
        _predict(gpu_model, representative, 0, image_size)
        _, gpu_seconds, peak_allocated, peak_reserved = _predict(
            gpu_model, representative, 0, image_size
        )
        benchmarks.append(
            {
                "backend": "semantic-door-torch-cuda",
                "actual_device_type": "cuda",
                "actual_device_name": torch.cuda.get_device_name(0),
                "seconds": gpu_seconds,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            }
        )
        if gpu_seconds <= cpu_seconds:
            selected_device = 0
            selected_model = gpu_model
            selection_reason = "GPU_FASTER_OR_EQUAL_MEASURED"
        else:
            selection_reason = "CPU_FASTER_MEASURED"
    receipt = {
        "schema": "blindassist-execution-backend-v1",
        "selected_backend": "semantic-door-torch-cuda"
        if selected_device != "cpu"
        else "semantic-door-torch-cpu",
        "selected_device_type": "cuda" if selected_device != "cpu" else "cpu",
        "selected_device_name": torch.cuda.get_device_name(0)
        if selected_device != "cpu"
        else "CPU",
        "selected_framework": f"torch-{torch.__version__}",
        "selection_reason": selection_reason,
        "benchmarks": benchmarks,
        "benchmark_model_calls": 4 if torch.cuda.is_available() else 2,
    }
    return selected_device, selected_model, receipt


def _components(mask: np.ndarray, door_class: int) -> list[dict[str, Any]]:
    binary = (mask == door_class).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    image_area = float(width * height)
    rows = []
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        rows.append(
            {
                "box_xyxy": [x, y, x + component_width, y + component_height],
                "component_pixels": area,
                "component_area_fraction": area / image_area,
                "box_area_fraction": component_width * component_height / image_area,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -row["component_area_fraction"],
            -row["box_area_fraction"],
            row["box_xyxy"],
        ),
    )


def _portal_positive_rows(
    manifest: dict[str, Any], audit: dict[str, Any]
) -> list[dict[str, Any]]:
    sources = {int(row["index"]): row for row in manifest["frames"]}
    truth = audit["portal_set_truth"]["frames"]
    rows = []
    for index in audit["admitted_frames"]:
        source = sources[int(index)]
        rows.append(
            {
                "key": f"portal-fresh-v3:{index}",
                "role": "DEVELOPMENT_POSITIVE_GLASS_PORTAL",
                "image_path": Path(source["local_path"]),
                "image_sha256": source["sha256"],
                "truth_box_xyxy": truth[str(index)]["portal_set_box_xyxy"],
            }
        )
    return rows


def _no_portal_rows(
    library: dict[str, Any], target_protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    target_by_id = {target["id"]: target for target in library["targets"]}
    rows = []
    for target, labels in target_protocol["human_entrance_labels"].items():
        facets = target_by_id[target]["facets"]
        for index_text, visible in labels.items():
            if visible:
                continue
            facet = facets[int(index_text) - 1]
            rows.append(
                {
                    "key": f"target-local-v1:{target}:{index_text}",
                    "role": "DEVELOPMENT_NEGATIVE_NO_PORTAL",
                    "image_path": Path(facet["local_path"]),
                    "image_sha256": facet["sha256"],
                    "truth_box_xyxy": None,
                }
            )
    return rows


def _threshold_metrics(
    threshold: float,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> dict[str, Any]:
    positive_truth = 0
    positive_presence = 0
    for row in positives:
        eligible = [
            candidate
            for candidate in row["components"]
            if candidate["component_area_fraction"] >= threshold
        ]
        positive_presence += bool(eligible)
        positive_truth += any(candidate["truth_member"] for candidate in eligible[:3])
    negative_false = sum(
        any(
            candidate["component_area_fraction"] >= threshold
            for candidate in row["components"]
        )
        for row in negatives
    )
    return {
        "threshold": threshold,
        "positive_truth_retained_top3": positive_truth,
        "positive_presence_frames": positive_presence,
        "negative_false_authorization_frames": negative_false,
        "balanced_accuracy": 0.5
        * (
            positive_truth / len(positives)
            + (len(negatives) - negative_false) / len(negatives)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = _json(protocol_path)
    if _sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("EVALUATOR_HASH_MISMATCH")
    sources = {name: _verify(spec) for name, spec in protocol["sources"].items()}
    model_path = _verify(protocol["model"])
    if ultralytics.__version__ != protocol["runtime"]["ultralytics_version"]:
        raise ValueError("ULTRALYTICS_VERSION_MISMATCH")
    model_probe = YOLO(str(model_path))
    if model_probe.task != "semantic" or model_probe.names != {0: "background", 1: "door"}:
        raise ValueError("SEMANTIC_MODEL_CONTRACT_MISMATCH")

    positives = _portal_positive_rows(
        _json(sources["portal_positive_manifest"]),
        _json(sources["portal_positive_audit"]),
    )
    negatives = _no_portal_rows(
        _json(sources["no_portal_library"]),
        _json(sources["no_portal_protocol"]),
    )
    if len(positives) != 6 or len(negatives) != 16:
        raise ValueError("DEVELOPMENT_COHORT_COUNT_MISMATCH")
    for row in positives + negatives:
        if not row["image_path"].is_file():
            raise ValueError(f"IMAGE_MISSING:{row['key']}:{row['image_path']}")
        if _sha256(row["image_path"]) != row["image_sha256"]:
            raise ValueError(f"IMAGE_HASH_MISMATCH:{row['key']}")

    image_size = int(protocol["runtime"]["image_size"])
    device, model, backend = _select_backend(
        model_path, positives[0]["image_path"], image_size
    )
    rows = []
    peak_allocated = 0
    peak_reserved = 0
    for source in positives + negatives:
        mask, elapsed, allocated, reserved = _predict(
            model, source["image_path"], device, image_size
        )
        peak_allocated = max(peak_allocated, allocated or 0)
        peak_reserved = max(peak_reserved, reserved or 0)
        components = _components(mask, protocol["model"]["door_class"])
        truth = source["truth_box_xyxy"]
        annotated = [
            {
                **component,
                "truth_member": _member(component["box_xyxy"], truth)
                if truth is not None
                else None,
            }
            for component in components[:20]
        ]
        rows.append(
            {
                "key": source["key"],
                "role": source["role"],
                "image_sha256": source["image_sha256"],
                "image_size": [int(mask.shape[1]), int(mask.shape[0])],
                "truth_box_xyxy": truth,
                "latency_seconds": elapsed,
                "door_pixel_fraction": float(np.mean(mask == protocol["model"]["door_class"])),
                "components": annotated,
            }
        )
    positive_results = [row for row in rows if "POSITIVE" in row["role"]]
    negative_results = [row for row in rows if "NEGATIVE" in row["role"]]
    values = sorted(
        {
            float(candidate["component_area_fraction"])
            for row in rows
            for candidate in row["components"]
        }
    )
    thresholds = [0.0, *values]
    if values:
        thresholds.append(float(np.nextafter(values[-1], np.inf)))
    sweep = [
        _threshold_metrics(threshold, positive_results, negative_results)
        for threshold in thresholds
    ]
    gate = protocol["development_gate"]
    eligible = [
        row
        for row in sweep
        if row["positive_truth_retained_top3"]
        >= gate["minimum_positive_truth_retained_top3"]
        and row["negative_false_authorization_frames"]
        <= gate["maximum_negative_false_authorization_frames"]
    ]
    selected = (
        sorted(
            eligible,
            key=lambda row: (
                row["negative_false_authorization_frames"],
                -row["positive_truth_retained_top3"],
                -row["threshold"],
            ),
        )[0]
        if eligible
        else None
    )
    decision = (
        "L10_PB6_SEMANTIC_PORTAL_PRESENCE_DEVELOPMENT_GATE_MET"
        if selected is not None
        else "L10_PB6_SEMANTIC_PORTAL_PRESENCE_DEVELOPMENT_GATE_NOT_MET"
    )
    backend["peak_selected_run_allocated_bytes"] = peak_allocated or None
    backend["peak_selected_run_reserved_bytes"] = peak_reserved or None
    result = {
        "schema": "l10-named-poi-semantic-portal-presence-development-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "model": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "task": model_probe.task,
            "classes": model_probe.names,
        },
        "backend": backend,
        "development": {
            "positive_frames": len(positive_results),
            "negative_frames": len(negative_results),
            "selection_rule": protocol["threshold_selection"],
            "selected": selected,
            "threshold_candidates": len(sweep),
            "sweep": sweep,
        },
        "decision": decision,
        "rows": rows,
        "next_step": protocol["next_step"],
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": decision,
                "selected": selected,
                "backend": backend["selected_backend"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
