"""Develop a generic semantic entrance proposal source on historical cohorts.

YOLOE sees only the current RGB frame and the fixed class prompt
``building entrance``.  It never receives the requested place identity, SCIL
evidence, PB5 geometry, or portal truth.  Development selects one raw proposal
confidence threshold on six frozen portal-set positives and sixteen separately
frozen no-portal controls.  Any later PB6 confirmation must freeze the selected
provider before new-model output is observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import ultralytics
from ultralytics import YOLOE


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


def _set_prompt(model: YOLOE, text_encoder_path: Path, prompt: str) -> None:
    previous = Path.cwd()
    os.chdir(text_encoder_path.parent)
    try:
        model.set_classes([prompt])
    finally:
        os.chdir(previous)


def _make_model(model_path: Path, text_encoder_path: Path, prompt: str) -> YOLOE:
    model = YOLOE(str(model_path))
    _set_prompt(model, text_encoder_path, prompt)
    return model


def _predict(
    model: YOLOE,
    image_path: Path,
    device: str | int,
    image_size: int,
    confidence_floor: float,
    provider_max_det: int,
    bounded_pool_size: int,
) -> tuple[list[dict[str, Any]], int, float, int | None, int | None]:
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = model.predict(
        source=str(image_path),
        device=device,
        imgsz=image_size,
        conf=confidence_floor,
        max_det=provider_max_det,
        verbose=False,
    )[0]
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    ranked = []
    if result.boxes is not None:
        ranked = sorted(
            zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )
    candidates = [
        {
            "rank": rank,
            "box_xyxy": [float(value) for value in box],
            "proposal_score": float(score),
        }
        for rank, (score, box) in enumerate(ranked[:bounded_pool_size], start=1)
    ]
    peak_allocated = (
        int(torch.cuda.max_memory_allocated()) if device != "cpu" else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved()) if device != "cpu" else None
    )
    return candidates, len(ranked), elapsed, peak_allocated, peak_reserved


def _select_backend(
    model_path: Path,
    text_encoder_path: Path,
    prompt: str,
    representative: Path,
    runtime: dict[str, Any],
) -> tuple[str | int, YOLOE, dict[str, Any]]:
    predict_args = (
        int(runtime["image_size"]),
        float(runtime["confidence_floor"]),
        int(runtime["provider_max_det"]),
        int(runtime["bounded_pool_size"]),
    )
    cpu_model = _make_model(model_path, text_encoder_path, prompt)
    _predict(cpu_model, representative, "cpu", *predict_args)
    *_, cpu_seconds, _, _ = _predict(
        cpu_model, representative, "cpu", *predict_args
    )
    benchmarks = [
        {
            "backend": "yoloe-torch-cpu",
            "actual_device_type": "cpu",
            "seconds": cpu_seconds,
        }
    ]
    selected_device: str | int = "cpu"
    selected_model = cpu_model
    selection_reason = "ACCELERATOR_UNAVAILABLE"
    if torch.cuda.is_available():
        gpu_model = _make_model(model_path, text_encoder_path, prompt)
        _predict(gpu_model, representative, 0, *predict_args)
        *_, gpu_seconds, peak_allocated, peak_reserved = _predict(
            gpu_model, representative, 0, *predict_args
        )
        benchmarks.append(
            {
                "backend": "yoloe-torch-cuda",
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
        "selected_backend": "yoloe-torch-cuda"
        if selected_device != "cpu"
        else "yoloe-torch-cpu",
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
            for candidate in row["candidates"]
            if candidate["proposal_score"] >= threshold
        ]
        positive_presence += bool(eligible)
        positive_truth += any(candidate["truth_member"] for candidate in eligible[:3])
    negative_false = sum(
        any(
            candidate["proposal_score"] >= threshold
            for candidate in row["candidates"]
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
    output_path = args.output.resolve()
    if output_path.exists():
        raise ValueError(f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = _json(protocol_path)
    if _sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("EVALUATOR_HASH_MISMATCH")
    sources = {name: _verify(spec) for name, spec in protocol["sources"].items()}
    model_path = _verify(protocol["model"])
    text_encoder_path = _verify(protocol["text_encoder"])
    if ultralytics.__version__ != protocol["runtime"]["ultralytics_version"]:
        raise ValueError("ULTRALYTICS_VERSION_MISMATCH")
    if text_encoder_path.name != "mobileclip2_b.ts":
        raise ValueError("TEXT_ENCODER_CONTRACT_MISMATCH")

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

    runtime = protocol["runtime"]
    prompt = protocol["representation"]["fixed_prompt"]
    device, model, backend = _select_backend(
        model_path,
        text_encoder_path,
        prompt,
        positives[0]["image_path"],
        runtime,
    )
    predict_args = (
        int(runtime["image_size"]),
        float(runtime["confidence_floor"]),
        int(runtime["provider_max_det"]),
        int(runtime["bounded_pool_size"]),
    )
    rows = []
    peak_allocated = 0
    peak_reserved = 0
    for source in positives + negatives:
        candidates, total, elapsed, allocated, reserved = _predict(
            model, source["image_path"], device, *predict_args
        )
        peak_allocated = max(peak_allocated, allocated or 0)
        peak_reserved = max(peak_reserved, reserved or 0)
        truth = source["truth_box_xyxy"]
        annotated = [
            {
                **candidate,
                "truth_member": _member(candidate["box_xyxy"], truth)
                if truth is not None
                else None,
            }
            for candidate in candidates
        ]
        rows.append(
            {
                "key": source["key"],
                "role": source["role"],
                "image_sha256": source["image_sha256"],
                "truth_box_xyxy": truth,
                "latency_seconds": elapsed,
                "provider_postprocessed_candidate_count": total,
                "candidates": annotated,
            }
        )
    positive_results = [row for row in rows if "POSITIVE" in row["role"]]
    negative_results = [row for row in rows if "NEGATIVE" in row["role"]]
    values = sorted(
        {
            float(candidate["proposal_score"])
            for row in rows
            for candidate in row["candidates"]
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
        "L10_PB6_YOLOE_PORTAL_PRESENCE_DEVELOPMENT_GATE_MET"
        if selected is not None
        else "L10_PB6_YOLOE_PORTAL_PRESENCE_DEVELOPMENT_GATE_NOT_MET"
    )
    backend["peak_selected_run_allocated_bytes"] = peak_allocated or None
    backend["peak_selected_run_reserved_bytes"] = peak_reserved or None
    result = {
        "schema": "l10-named-poi-yoloe-portal-presence-development-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "provider": {
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "text_encoder_path": str(text_encoder_path),
            "text_encoder_sha256": _sha256(text_encoder_path),
            "fixed_prompt": prompt,
            "ultralytics_version": ultralytics.__version__,
            "image_size": runtime["image_size"],
            "confidence_floor": runtime["confidence_floor"],
            "provider_max_det": runtime["provider_max_det"],
            "bounded_pool_size": runtime["bounded_pool_size"],
            "identity_input": "FORBIDDEN",
        },
        "backend": backend,
        "development": {
            "positive_frames": len(positive_results),
            "negative_frames": len(negative_results),
            "membership_rule": protocol["candidate_membership"],
            "selection_rule": protocol["threshold_selection"],
            "selected": selected,
            "threshold_candidates": len(sweep),
            "sweep": sweep,
        },
        "decision": decision,
        "provider_selection": protocol["provider_selection"],
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
                "backend": backend["selected_backend"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
