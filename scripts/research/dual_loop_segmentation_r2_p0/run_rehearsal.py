"""Run a consumed-only, canonical-view formal rehearsal with atomic resume."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from . import PROTOCOL_ID
from .canonicalizer import sha256_file
from .postprocess import filter_candidate_by_class, load_postprocess
from ..dual_loop_segmentation_candidate_utility import evaluate_candidate_utility as base
from ..dual_loop_segmentation_candidate_utility.component_metrics import (
    aggregate_confusion,
    component_metrics,
    component_records,
    pixel_metrics,
)


FRAME_SCHEMA = "blindassist.dual_loop_segmentation_r2_p0.rehearsal_frame.v1"
COMPONENT_SCHEMA = "blindassist.dual_loop_segmentation_r2_p0.rehearsal_component.v1"
REPORT_SCHEMA = "blindassist.dual_loop_segmentation_r2_p0.rehearsal_report.v1"
CLASS_TO_ID = {
    "walkable": 0,
    "boundary_step_curb": 1,
    "obstacle": 2,
    "unknown_nonwalkable": 3,
}


class RehearsalError(ValueError):
    """Raised when consumed-only rehearsal identity or output is invalid."""


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _write_atomic(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(value, pretty=pretty))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RehearsalError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RehearsalError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RehearsalError(f"expected object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise RehearsalError(f"zero-row input is invalid: {path}")
    return rows


def _pack(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(np.uint8), axis=None).tobytes()).decode("ascii")


def _aggregate_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = sum(int(row["candidate_component_metrics"]["predicted_component_count"]) for row in rows)
    truth = sum(int(row["candidate_component_metrics"]["truth_component_count"]) for row in rows)
    hit_predicted = sum(int(row["candidate_component_metrics"]["hit_predicted_component_count"]) for row in rows)
    hit_truth = sum(int(row["candidate_component_metrics"]["hit_truth_component_count"]) for row in rows)
    false_count = sum(int(row["candidate_component_metrics"]["false_activation_component_count"]) for row in rows)
    return {
        "predicted_component_count": predicted,
        "truth_component_count": truth,
        "hit_predicted_component_count": hit_predicted,
        "hit_truth_component_count": hit_truth,
        "component_precision": float(hit_predicted / predicted) if predicted else (1.0 if truth == 0 else None),
        "component_recall": float(hit_truth / truth) if truth else (1.0 if predicted == 0 else None),
        "false_activation_component_count": false_count,
        "false_activation_components_per_frame": float(false_count / len(rows)),
    }


def _input_identity(
    *,
    view_manifest: Path,
    trace_path: Path,
    model_path: Path,
    postprocess_path: Path,
) -> dict[str, str]:
    return {
        "view_manifest_sha256": sha256_file(view_manifest),
        "trace_sha256": sha256_file(trace_path),
        "model_sha256": sha256_file(model_path),
        "postprocess_sha256": sha256_file(postprocess_path),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }


def _identity_digest(identity: dict[str, str], role: str) -> str:
    payload = json.dumps(
        {"identity": identity, "role": role},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_aggregates(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_source[str(row["source_id"])].append(row)
    result: list[dict[str, Any]] = []
    for source_id, rows in sorted(by_source.items()):
        arm_a = aggregate_confusion(row["arms"]["A"]["pixel"] for row in rows)
        arm_c = aggregate_confusion(row["arms"]["C"]["pixel"] for row in rows)
        components = _aggregate_components(rows)
        result.append(
            {
                "source_id": source_id,
                "frame_count": len(rows),
                "delta_recall_C_minus_A": float(arm_c["recall"] - arm_a["recall"]),
                "delta_false_positive_area_fraction_C_minus_A": float(
                    arm_c["false_positive_area_fraction"]
                    - arm_a["false_positive_area_fraction"]
                ),
                "candidate_component_recall": components["component_recall"],
                "false_activation_components_per_frame": components[
                    "false_activation_components_per_frame"
                ],
            }
        )
    return result


def run(
    *,
    repo_root: Path,
    view_root: Path,
    role: str,
    trace_path: Path,
    model_path: Path,
    postprocess_path: Path,
    output_root: Path,
    threads: int,
    stop_after_frames: int | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    view_root = view_root.resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to((repo_root / "artifacts.local").resolve())
    except ValueError as exc:
        raise RehearsalError("rehearsal output must stay under artifacts.local") from exc
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite rehearsal: {output_root}")
    view_receipt = _read_json(view_root / "receipt.json")
    if view_receipt.get("status") != "CANONICAL_VIEW_MATERIALIZED":
        raise RehearsalError("canonical view is not materialized")
    view_manifest = view_root / str(view_receipt["manifest"])
    if sha256_file(view_manifest) != view_receipt.get("manifest_sha256"):
        raise RehearsalError("canonical view manifest SHA256 mismatch")
    all_rows = _read_jsonl(view_manifest)
    selected = [row for row in all_rows if row.get("role") == role]
    if not selected:
        raise RehearsalError(f"zero rows selected for rehearsal role {role!r}")
    if role not in {"dev", "consumed_old_blind", "r1_consumed_fresh"}:
        raise RehearsalError(f"role {role!r} is not permitted for model rehearsal")
    traces = base.load_trace(trace_path.resolve())
    postprocess = load_postprocess(postprocess_path.resolve())
    identity = _input_identity(
        view_manifest=view_manifest,
        trace_path=trace_path.resolve(),
        model_path=model_path.resolve(),
        postprocess_path=postprocess_path.resolve(),
    )
    identity_digest = _identity_digest(identity, role)
    staging = output_root.with_name(output_root.name + ".staging")
    state_path = staging / "state.json"
    if staging.exists():
        state = _read_json(state_path)
        if state.get("input_identity_sha256") != identity_digest:
            raise RehearsalError("rehearsal staging identity mismatch")
    else:
        staging.mkdir(parents=True)
        _write_atomic(
            state_path,
            {
                "schema_version": "blindassist.dual_loop_segmentation_r2_p0.rehearsal_state.v1",
                "protocol_id": PROTOCOL_ID,
                "status": "RUNNING",
                "input_identity_sha256": identity_digest,
                "completed_frames": 0,
            },
        )
    segmenter = base.TFLiteSegmenter(model_path.resolve(), threads=threads)
    frame_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        frame_part = staging / "frame_parts" / f"{index:06d}.json"
        component_part = staging / "component_parts" / f"{index:06d}.json"
        if frame_part.is_file() and component_part.is_file():
            frame_row = _read_json(frame_part)
            local_components = _read_json(component_part).get("rows")
            if not isinstance(local_components, list):
                raise RehearsalError(f"invalid resumed component part: {component_part}")
            frame_rows.append(frame_row)
            component_rows.extend(local_components)
            continue
        image_path = (repo_root / str(row["image_repo_relative_path"])).resolve()
        canonical_path = (view_root / str(row["canonical_mask_path"])).resolve()
        if sha256_file(image_path) != row.get("image_sha256"):
            raise RehearsalError(f"{row['id']}: image SHA256 mismatch")
        if sha256_file(canonical_path) != row.get("canonical_mask_sha256"):
            raise RehearsalError(f"{row['id']}: canonical mask SHA256 mismatch")
        with Image.open(image_path) as image:
            source_width, source_height = image.size
            ids, confidence, margin, segmentation_timing = segmenter.infer(image)
        with Image.open(canonical_path) as image:
            if image.mode != "L" or image.size != (256, 256):
                raise RehearsalError(f"{row['id']}: evaluator accepts only canonical L/256x256")
            truth_ids = np.asarray(image, dtype=np.uint8)
        if np.any(truth_ids > 3):
            raise RehearsalError(f"{row['id']}: canonical truth contains IDs outside 0..3")
        key = (str(row["source_id"]), int(row["frame_id"]), str(row["image_sha256"]))
        trace = traces.get(key)
        if trace is None:
            raise RehearsalError(f"{row['id']}: missing frozen YOLO trace")
        detector_mask = base.box_union_mask(
            trace["detections"],
            source_width=source_width,
            source_height=source_height,
        )
        postprocess_start = time.perf_counter()
        candidate_by_class = filter_candidate_by_class(
            ids=ids,
            confidence=confidence,
            margin=margin,
            detector_mask=detector_mask,
            class_to_id=CLASS_TO_ID,
            config=postprocess,
        )
        candidate_hazard = np.zeros((256, 256), dtype=bool)
        for class_mask in candidate_by_class.values():
            candidate_hazard |= class_mask
        segmentation_hazard = np.isin(ids, [1, 2])
        truth_hazard = np.isin(truth_ids, [1, 2])
        candidate_truth = truth_hazard & ~detector_mask
        arm_masks = {
            "A": detector_mask,
            "B": candidate_hazard,
            "C": detector_mask | candidate_hazard,
        }
        arm_metrics = {
            name: {"pixel": pixel_metrics(mask, truth_hazard)}
            for name, mask in arm_masks.items()
        }
        candidate_component_metrics = component_metrics(candidate_hazard, candidate_truth)
        local_components: list[dict[str, Any]] = []
        for class_name, class_mask in candidate_by_class.items():
            truth_class = (truth_ids == CLASS_TO_ID[class_name]) & ~detector_mask
            local_components.extend(
                component_records(
                    {class_name: class_mask},
                    truth_class,
                    detector_mask,
                    confidence,
                    margin,
                    source_id=str(row["source_id"]),
                    frame_id=int(row["frame_id"]),
                )
            )
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        for component in local_components:
            component["schema_version"] = COMPONENT_SCHEMA
            component["protocol_id"] = PROTOCOL_ID
            component["rehearsal_role"] = role
        frame_row = {
            "schema_version": FRAME_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "formal_authority": False,
            "rehearsal_role": role,
            "view_row_id": row["id"],
            "source_id": row["source_id"],
            "session_id": row["session_id"],
            "frame_id": int(row["frame_id"]),
            "image_sha256": row["image_sha256"],
            "canonical_mask_sha256": row["canonical_mask_sha256"],
            "candidate_id": postprocess["candidate_id"],
            "arms": arm_metrics,
            "candidate_pixel_metrics": pixel_metrics(candidate_hazard, candidate_truth),
            "candidate_component_metrics": candidate_component_metrics,
            "runtime_observation_only": {
                **segmentation_timing,
                "postprocess_and_metrics_ms": float(postprocess_ms),
            },
            "packed_masks": {
                "shape": [256, 256],
                "A": _pack(arm_masks["A"]),
                "B": _pack(arm_masks["B"]),
                "C": _pack(arm_masks["C"]),
                "candidate": _pack(candidate_hazard),
                "candidate_boundary_step_curb": _pack(
                    candidate_by_class["boundary_step_curb"]
                ),
                "candidate_obstacle": _pack(candidate_by_class["obstacle"]),
            },
        }
        _write_atomic(frame_part, frame_row)
        _write_atomic(component_part, {"rows": local_components})
        frame_rows.append(frame_row)
        component_rows.extend(local_components)
        _write_atomic(
            state_path,
            {
                "schema_version": "blindassist.dual_loop_segmentation_r2_p0.rehearsal_state.v1",
                "protocol_id": PROTOCOL_ID,
                "status": "RUNNING",
                "input_identity_sha256": identity_digest,
                "completed_frames": len(frame_rows),
                "last_view_row_id": row["id"],
            },
        )
        if stop_after_frames is not None and len(frame_rows) >= stop_after_frames:
            raise RuntimeError("intentional rehearsal interruption")
    if not frame_rows:
        raise RehearsalError("rehearsal produced zero frame rows")
    frame_rows.sort(key=lambda item: (str(item["source_id"]), int(item["frame_id"])))
    component_rows.sort(key=lambda item: str(item["component_id"]))
    frames_path = staging / "frames.jsonl"
    components_path = staging / "components.jsonl"
    frames_path.write_bytes(b"".join(_json_bytes(row) for row in frame_rows))
    components_path.write_bytes(b"".join(_json_bytes(row) for row in component_rows))
    arm_aggregate = {
        arm: aggregate_confusion(row["arms"][arm]["pixel"] for row in frame_rows)
        for arm in ("A", "B", "C")
    }
    candidate_components = _aggregate_components(frame_rows)
    source_aggregates = _source_aggregates(frame_rows)
    report = {
        "schema_version": REPORT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "REHEARSAL_COMPLETE_UNVALIDATED",
        "formal_authority": False,
        "rehearsal_role": role,
        "candidate_id": postprocess["candidate_id"],
        "input_identity": identity,
        "input_identity_sha256": identity_digest,
        "frame_count": len(frame_rows),
        "component_row_count": len(component_rows),
        "frames_sha256": sha256_file(frames_path),
        "components_sha256": sha256_file(components_path),
        "summary": {
            "arm_pixel_metrics": arm_aggregate,
            "candidate_components": candidate_components,
            "delta_recall_C_minus_A": float(
                arm_aggregate["C"]["recall"] - arm_aggregate["A"]["recall"]
            ),
            "delta_false_positive_area_fraction_C_minus_A": float(
                arm_aggregate["C"]["false_positive_area_fraction"]
                - arm_aggregate["A"]["false_positive_area_fraction"]
            ),
            "false_activation_components_per_frame": candidate_components[
                "false_activation_components_per_frame"
            ],
        },
        "source_aggregates": source_aggregates,
        "session_aggregates": source_aggregates,
        "atomic_publish": True,
        "interruption_resume_supported": True,
        "forbidden_inputs_consumed": [],
    }
    _write_atomic(staging / "report.json", report)
    _write_atomic(
        state_path,
        {
            "schema_version": "blindassist.dual_loop_segmentation_r2_p0.rehearsal_state.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "input_identity_sha256": identity_digest,
            "completed_frames": len(frame_rows),
            "report_sha256": sha256_file(staging / "report.json"),
        },
    )
    staging.replace(output_root)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stop-after-frames", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    value = run(
        repo_root=args.repo_root,
        view_root=args.view_root,
        role=args.role,
        trace_path=args.trace,
        model_path=args.model,
        postprocess_path=args.postprocess,
        output_root=args.output_root,
        threads=args.threads,
        stop_after_frames=args.stop_after_frames,
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "frames": value["frame_count"],
                "components": value["component_row_count"],
            }
        )
    )
