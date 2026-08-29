"""Develop an ADE20K door-component portal source on the consumed PB6 cohort."""

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


def _components(mask: np.ndarray, classes: set[int]) -> list[dict[str, Any]]:
    binary = np.isin(mask, list(classes)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    image_area = float(width * height)
    rows = []
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        class_pixels = mask[labels == label]
        class_counts = {
            str(class_id): int(np.sum(class_pixels == class_id)) for class_id in classes
        }
        rows.append(
            {
                "box_xyxy": [x, y, x + component_width, y + component_height],
                "component_pixels": area,
                "component_area_fraction": area / image_area,
                "box_bottom_fraction": (y + component_height) / float(height),
                "class_pixels": class_counts,
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["component_area_fraction"], row["box_xyxy"]),
    )


def _metrics(
    threshold: float,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    bottom_floor: float,
) -> dict[str, Any]:
    positive_truth = 0
    positive_presence = 0
    for row in positives:
        eligible = [
            component
            for component in row["components"]
            if component["component_area_fraction"] >= threshold
            and component["box_bottom_fraction"] >= bottom_floor
        ]
        positive_presence += bool(eligible)
        positive_truth += any(component["truth_member"] for component in eligible[:3])
    negative_false = sum(
        any(
            component["component_area_fraction"] >= threshold
            and component["box_bottom_fraction"] >= bottom_floor
            for component in row["components"]
        )
        for row in negatives
    )
    return {
        "threshold": threshold,
        "positive_truth_retained_top3": positive_truth,
        "positive_presence_frames": positive_presence,
        "negative_false_presence_frames": negative_false,
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
    output_path = args.output.resolve()
    if output_path.exists():
        raise ValueError(f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = _json(protocol_path)
    if _sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("EVALUATOR_HASH_MISMATCH")
    sources = {name: _verify(spec) for name, spec in protocol["sources"].items()}
    model_path = _verify(protocol["model"])
    if ultralytics.__version__ != protocol["runtime"]["ultralytics_version"]:
        raise ValueError("ULTRALYTICS_VERSION_MISMATCH")
    if not torch.cuda.is_available():
        raise ValueError("CUDA_UNAVAILABLE")
    model = YOLO(str(model_path))
    expected = {14: "door", 58: "screen door"}
    if model.task != "semantic" or any(model.names[key] != value for key, value in expected.items()):
        raise ValueError("ADE20K_DOOR_CLASS_CONTRACT_MISMATCH")

    source = _json(sources["source_spec"])
    manifest = _json(sources["source_manifest"])
    audit = _json(sources["source_audit"])
    if any(
        audit["checks"][key] != 0
        for key in (
            "ocr_calls_before_freeze",
            "semantic_calls_before_freeze",
            "geometry_calls_before_freeze",
        )
    ):
        raise ValueError("SOURCE_WAS_NOT_MODEL_BLIND_AT_FREEZE")
    source_by_index = {int(row["index"]): row for row in source["cohort"]}
    manifest_by_index = {int(row["index"]): row for row in manifest["frames"]}
    audit_by_index = {int(row["index"]): row for row in audit["frames"]}
    if set(source_by_index) != set(range(1, 9)):
        raise ValueError("COHORT_INDEX_MISMATCH")

    rows = []
    torch.cuda.reset_peak_memory_stats()
    for index in sorted(source_by_index):
        source_row = source_by_index[index]
        frame = manifest_by_index[index]
        truth = audit_by_index[index]
        image_path = Path(frame["local_path"])
        if _sha256(image_path) != frame["sha256"] or frame["sha256"] != truth["image_sha256"]:
            raise ValueError(f"IMAGE_HASH_MISMATCH:{index}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or [image.shape[1], image.shape[0]] != truth["local_image_size"]:
            raise ValueError(f"IMAGE_DECODE_OR_SIZE_MISMATCH:{index}")
        started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            device=0,
            imgsz=int(protocol["runtime"]["image_size"]),
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if result.semantic_mask is None:
            raise ValueError(f"SEMANTIC_MASK_MISSING:{index}")
        mask = result.semantic_mask.data.detach().to("cpu").numpy().astype(np.uint8)
        truth_box = truth["portal_set_box_xyxy"]
        components = [
            {
                **component,
                "truth_member": _member(component["box_xyxy"], truth_box)
                if truth_box is not None
                else None,
            }
            for component in _components(mask, set(expected))[:30]
        ]
        rows.append(
            {
                "index": index,
                "id": source_row["id"],
                "role": truth["status"],
                "image_sha256": frame["sha256"],
                "latency_seconds": elapsed,
                "mask_size": [int(mask.shape[1]), int(mask.shape[0])],
                "door_or_screen_door_pixel_fraction": float(np.mean(np.isin(mask, list(expected)))),
                "truth_box_xyxy": truth_box,
                "components": components,
            }
        )
    actual_device = str(next(model.model.parameters()).device)
    if not actual_device.startswith("cuda"):
        raise ValueError(f"SILENT_DEVICE_FALLBACK:{actual_device}")

    positives = [row for row in rows if row["role"] == "IN_SCOPE_DOOR_LEVEL_POSITIVE"]
    negatives = [row for row in rows if row["role"] == "NO_PORTAL_NEGATIVE"]
    ood = [row for row in rows if row["role"] == "LARGE_OPEN_ENTRANCE_OOD"]
    if (len(positives), len(negatives), len(ood)) != (4, 2, 2):
        raise ValueError("ROLE_COUNT_MISMATCH")
    values = sorted(
        {
            float(component["component_area_fraction"])
            for row in positives + negatives
            for component in row["components"]
            if component["box_bottom_fraction"]
            >= protocol["component_rule"]["minimum_box_bottom_fraction"]
        }
    )
    thresholds = [0.0, *values]
    if values:
        thresholds.append(float(np.nextafter(values[-1], np.inf)))
    sweep = [
        _metrics(
            threshold,
            positives,
            negatives,
            float(protocol["component_rule"]["minimum_box_bottom_fraction"]),
        )
        for threshold in thresholds
    ]
    gate = protocol["development_gate"]
    eligible = [
        row
        for row in sweep
        if row["positive_truth_retained_top3"]
        >= gate["minimum_positive_truth_retained_top3"]
        and row["negative_false_presence_frames"]
        <= gate["maximum_negative_false_presence_frames"]
    ]
    selected = (
        sorted(
            eligible,
            key=lambda row: (
                row["negative_false_presence_frames"],
                -row["positive_truth_retained_top3"],
                -row["threshold"],
            ),
        )[0]
        if eligible
        else None
    )
    bottom_floor = float(protocol["component_rule"]["minimum_box_bottom_fraction"])
    ood_presence = sum(
        any(
            component["component_area_fraction"] >= selected["threshold"]
            and component["box_bottom_fraction"] >= bottom_floor
            for component in row["components"]
        )
        for row in ood
    ) if selected else 0
    decision = (
        "L10_PB7_ADE20K_DOOR_COMPONENT_DEVELOPMENT_GATE_MET"
        if selected is not None
        else "L10_PB7_ADE20K_DOOR_COMPONENT_DEVELOPMENT_GATE_NOT_MET"
    )
    result = {
        "schema": "l10-named-poi-ade20k-portal-development-result-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "decision": decision,
        "model": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "task": model.task,
            "classes": expected,
        },
        "backend": {
            "selected": "ade20k-semantic-torch-cuda",
            "actual_device": actual_device,
            "actual_device_name": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "ultralytics": ultralytics.__version__,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "development": {
            "positive_frames": len(positives),
            "negative_frames": len(negatives),
            "ood_challenge_frames": len(ood),
            "selected": selected,
            "ood_presence_at_selected_threshold": ood_presence,
            "threshold_candidates": len(sweep),
            "selection_rule": protocol["threshold_selection"],
            "sweep": sweep,
        },
        "rows": rows,
        "next_step": protocol["next_step"],
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "decision": decision,
                "selected": selected,
                "ood_presence_at_selected_threshold": ood_presence,
                "backend": "ade20k-semantic-torch-cuda",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
