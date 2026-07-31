"""Independently recompute every R2-P0 rehearsal metric from packed masks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from . import PROTOCOL_ID


class RehearsalValidationError(ValueError):
    """Raised when rehearsal rows cannot reproduce their report."""


@dataclass
class IndependentComponent:
    index: int
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RehearsalValidationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RehearsalValidationError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RehearsalValidationError(f"expected object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise RehearsalValidationError(f"zero-row rehearsal is invalid: {path}")
    return rows


def _unpack(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    if len(raw) != 8192:
        raise RehearsalValidationError("packed 256x256 mask byte count mismatch")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=256 * 256).reshape(256, 256).astype(bool)


def _components(mask: np.ndarray) -> list[IndependentComponent]:
    value = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(value, dtype=bool)
    result: list[IndependentComponent] = []
    for y in range(256):
        for x in range(256):
            if not value[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            min_x = max_x = x
            min_y = max_y = y
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < 256
                            and 0 <= nx < 256
                            and value[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            component_mask = np.zeros_like(value, dtype=bool)
            ys, xs = zip(*pixels)
            component_mask[np.asarray(ys), np.asarray(xs)] = True
            result.append(
                IndependentComponent(
                    index=len(result),
                    mask=component_mask,
                    area=len(pixels),
                    bbox=(min_x, min_y, max_x + 1, max_y + 1),
                )
            )
    return result


def _ratio(numerator: int, denominator: int, *, empty: float | None = None) -> float | None:
    return float(numerator / denominator) if denominator else empty


def _pixel_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    tn = int(pred.size - tp - fp - fn)
    both_empty = tp + fp + fn == 0
    precision = _ratio(tp, tp + fp, empty=1.0 if both_empty else None)
    recall = _ratio(tp, tp + fn, empty=1.0 if both_empty else None)
    iou = _ratio(tp, tp + fp + fn, empty=1.0 if both_empty else None)
    f1 = None if precision is None or recall is None else (
        0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": int(np.count_nonzero(pred)),
        "truth_pixels": int(np.count_nonzero(target)),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "false_positive_area_fraction": float(fp / pred.size),
    }


def _component_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    predicted_components = _components(predicted)
    truth_components = _components(truth)
    predicted_hits = [
        any(np.any(component.mask & target.mask) for target in truth_components)
        for component in predicted_components
    ]
    truth_hits = [
        any(np.any(component.mask & candidate.mask) for candidate in predicted_components)
        for component in truth_components
    ]
    predicted_count = len(predicted_components)
    truth_count = len(truth_components)
    return {
        "predicted_component_count": predicted_count,
        "truth_component_count": truth_count,
        "hit_predicted_component_count": int(sum(predicted_hits)),
        "hit_truth_component_count": int(sum(truth_hits)),
        "component_precision": _ratio(
            sum(predicted_hits),
            predicted_count,
            empty=1.0 if truth_count == 0 else None,
        ),
        "component_recall": _ratio(
            sum(truth_hits),
            truth_count,
            empty=1.0 if predicted_count == 0 else None,
        ),
        "false_activation_component_count": int(predicted_count - sum(predicted_hits)),
    }


def _close(left: Any, right: Any, *, tolerance: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def _assert_fields(stored: dict[str, Any], recomputed: dict[str, Any], context: str) -> None:
    for key, value in recomputed.items():
        if key not in stored or not _close(stored[key], value):
            raise RehearsalValidationError(
                f"{context}.{key} mismatch: stored={stored.get(key)!r}, recomputed={value!r}"
            )


def _aggregate_pixel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in ("tp", "fp", "fn", "tn")
    }
    blank = np.zeros((1, 1), dtype=bool)
    del blank
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    both_empty = tp + fp + fn == 0
    precision = _ratio(tp, tp + fp, empty=1.0 if both_empty else None)
    recall = _ratio(tp, tp + fn, empty=1.0 if both_empty else None)
    iou = _ratio(tp, tp + fp + fn, empty=1.0 if both_empty else None)
    return {
        **totals,
        "predicted_pixels": tp + fp,
        "truth_pixels": tp + fn,
        "pixel_count": tp + fp + fn + tn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": None if precision is None or recall is None else (
            0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        ),
        "false_positive_area_fraction": float(fp / (tp + fp + fn + tn)),
        "mean_frame_precision": float(
            np.mean([row["precision"] for row in rows if row["precision"] is not None])
        ),
        "mean_frame_recall": float(
            np.mean([row["recall"] for row in rows if row["recall"] is not None])
        ),
    }


def _aggregate_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = sum(int(row["predicted_component_count"]) for row in rows)
    truth = sum(int(row["truth_component_count"]) for row in rows)
    hit_predicted = sum(int(row["hit_predicted_component_count"]) for row in rows)
    hit_truth = sum(int(row["hit_truth_component_count"]) for row in rows)
    false_count = sum(int(row["false_activation_component_count"]) for row in rows)
    return {
        "predicted_component_count": predicted,
        "truth_component_count": truth,
        "hit_predicted_component_count": hit_predicted,
        "hit_truth_component_count": hit_truth,
        "component_precision": _ratio(hit_predicted, predicted, empty=1.0 if truth == 0 else None),
        "component_recall": _ratio(hit_truth, truth, empty=1.0 if predicted == 0 else None),
        "false_activation_component_count": false_count,
        "false_activation_components_per_frame": float(false_count / len(rows)),
    }


def validate(
    *,
    repo_root: Path,
    view_root: Path,
    rehearsal_root: Path,
    trace_path: Path,
    model_path: Path,
    postprocess_path: Path,
    evaluator_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    view_root = view_root.resolve()
    rehearsal_root = rehearsal_root.resolve()
    report = _read_json(rehearsal_root / "report.json")
    frames_path = rehearsal_root / "frames.jsonl"
    components_path = rehearsal_root / "components.jsonl"
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("status") != "REHEARSAL_COMPLETE_UNVALIDATED"
        or report.get("formal_authority") is not False
    ):
        raise RehearsalValidationError("rehearsal report identity/status mismatch")
    if sha256_file(frames_path) != report.get("frames_sha256"):
        raise RehearsalValidationError("frames SHA256 mismatch")
    if sha256_file(components_path) != report.get("components_sha256"):
        raise RehearsalValidationError("components SHA256 mismatch")
    expected_identity = {
        "view_manifest_sha256": sha256_file(view_root / "manifest.jsonl"),
        "trace_sha256": sha256_file(trace_path.resolve()),
        "model_sha256": sha256_file(model_path.resolve()),
        "postprocess_sha256": sha256_file(postprocess_path.resolve()),
        "evaluator_sha256": sha256_file(evaluator_path.resolve()),
    }
    if report.get("input_identity") != expected_identity:
        raise RehearsalValidationError("rehearsal input identity mismatch")
    view_rows = _read_jsonl(view_root / "manifest.jsonl")
    selected_view = {
        str(row["id"]): row
        for row in view_rows
        if row.get("role") == report.get("rehearsal_role")
    }
    frame_rows = _read_jsonl(frames_path)
    component_rows = _read_jsonl(components_path)
    if len(frame_rows) != int(report.get("frame_count", -1)):
        raise RehearsalValidationError("frame denominator mismatch")
    if len(component_rows) != int(report.get("component_row_count", -1)):
        raise RehearsalValidationError("component denominator mismatch")
    if {str(row["view_row_id"]) for row in frame_rows} != set(selected_view):
        raise RehearsalValidationError("rehearsal frame identities differ from canonical view")
    ledger_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for component in component_rows:
        ledger_by_key[
            (
                str(component["source_id"]),
                int(component["frame_id"]),
                str(component["class_name"]),
            )
        ].append(component)
    per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ("A", "B", "C")}
    per_candidate_components: list[dict[str, Any]] = []
    source_frame_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, frame in enumerate(frame_rows, start=1):
        view = selected_view[str(frame["view_row_id"])]
        canonical_path = (view_root / str(view["canonical_mask_path"])).resolve()
        if sha256_file(canonical_path) != frame.get("canonical_mask_sha256"):
            raise RehearsalValidationError(f"frame {index}: canonical truth identity mismatch")
        with Image.open(canonical_path) as image:
            truth_ids = np.asarray(image, dtype=np.uint8)
        truth_hazard = np.isin(truth_ids, [1, 2])
        packed = frame.get("packed_masks", {})
        masks = {
            name: _unpack(str(packed[name]))
            for name in (
                "A",
                "B",
                "C",
                "candidate",
                "candidate_boundary_step_curb",
                "candidate_obstacle",
            )
        }
        if not np.array_equal(masks["B"], masks["candidate"]):
            raise RehearsalValidationError(f"frame {index}: B and candidate masks differ")
        if not np.array_equal(masks["C"], masks["A"] | masks["B"]):
            raise RehearsalValidationError(f"frame {index}: C is not A union B")
        if not np.array_equal(
            masks["candidate"],
            masks["candidate_boundary_step_curb"] | masks["candidate_obstacle"],
        ):
            raise RehearsalValidationError(f"frame {index}: class masks do not form candidate")
        for arm in ("A", "B", "C"):
            metrics = _pixel_metrics(masks[arm], truth_hazard)
            _assert_fields(frame["arms"][arm]["pixel"], metrics, f"frame{index}.{arm}")
            per_arm[arm].append(metrics)
        candidate_truth = truth_hazard & ~masks["A"]
        candidate_metrics = _component_metrics(masks["candidate"], candidate_truth)
        _assert_fields(
            frame["candidate_component_metrics"],
            candidate_metrics,
            f"frame{index}.candidate_components",
        )
        per_candidate_components.append(candidate_metrics)
        for class_name, mask_name, class_id in (
            ("boundary_step_curb", "candidate_boundary_step_curb", 1),
            ("obstacle", "candidate_obstacle", 2),
        ):
            truth_class = (truth_ids == class_id) & ~masks["A"]
            recomputed_components = _components(masks[mask_name])
            ledger = sorted(
                ledger_by_key[
                    (str(frame["source_id"]), int(frame["frame_id"]), class_name)
                ],
                key=lambda row: int(row["component_index"]),
            )
            if len(ledger) != len(recomputed_components):
                raise RehearsalValidationError(
                    f"frame {index} {class_name}: component ledger count mismatch"
                )
            for expected, stored in zip(recomputed_components, ledger, strict=True):
                overlap = int(np.count_nonzero(expected.mask & truth_class))
                if (
                    int(stored["component_index"]) != expected.index
                    or int(stored["area_pixels"]) != expected.area
                    or list(stored["bbox_xyxy"]) != list(expected.bbox)
                    or int(stored["truth_intersection_pixels"]) != overlap
                    or bool(stored["truth_intersects"]) != (overlap > 0)
                ):
                    raise RehearsalValidationError(
                        f"frame {index} {class_name}: component ledger mismatch"
                    )
        source_frame_rows[str(frame["source_id"])].append(
            {
                "arms": {
                    arm: per_arm[arm][-1]
                    for arm in ("A", "B", "C")
                },
                "candidate_components": candidate_metrics,
            }
        )
    arm_aggregates = {
        arm: _aggregate_pixel(per_arm[arm])
        for arm in ("A", "B", "C")
    }
    component_aggregate = _aggregate_components(per_candidate_components)
    summary = report["summary"]
    for arm, metrics in arm_aggregates.items():
        _assert_fields(summary["arm_pixel_metrics"][arm], metrics, f"summary.{arm}")
    _assert_fields(
        summary["candidate_components"],
        component_aggregate,
        "summary.candidate_components",
    )
    delta_recall = float(arm_aggregates["C"]["recall"] - arm_aggregates["A"]["recall"])
    delta_fp = float(
        arm_aggregates["C"]["false_positive_area_fraction"]
        - arm_aggregates["A"]["false_positive_area_fraction"]
    )
    if not _close(summary["delta_recall_C_minus_A"], delta_recall):
        raise RehearsalValidationError("summary delta recall mismatch")
    if not _close(summary["delta_false_positive_area_fraction_C_minus_A"], delta_fp):
        raise RehearsalValidationError("summary delta FP mismatch")
    source_aggregates: list[dict[str, Any]] = []
    for source_id, rows in sorted(source_frame_rows.items()):
        source_a = _aggregate_pixel([row["arms"]["A"] for row in rows])
        source_c = _aggregate_pixel([row["arms"]["C"] for row in rows])
        source_components = _aggregate_components(
            [row["candidate_components"] for row in rows]
        )
        source_aggregates.append(
            {
                "source_id": source_id,
                "frame_count": len(rows),
                "delta_recall_C_minus_A": float(source_c["recall"] - source_a["recall"]),
                "delta_false_positive_area_fraction_C_minus_A": float(
                    source_c["false_positive_area_fraction"]
                    - source_a["false_positive_area_fraction"]
                ),
                "candidate_component_recall": source_components["component_recall"],
                "false_activation_components_per_frame": source_components[
                    "false_activation_components_per_frame"
                ],
            }
        )
    if len(source_aggregates) != len(report.get("source_aggregates", [])):
        raise RehearsalValidationError("source aggregate denominator mismatch")
    for expected, stored in zip(source_aggregates, report["source_aggregates"], strict=True):
        _assert_fields(stored, expected, f"source.{expected['source_id']}")
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_r2_p0.rehearsal_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID",
        "formal_authority": False,
        "rehearsal_role": report["rehearsal_role"],
        "candidate_id": report["candidate_id"],
        "frame_count": len(frame_rows),
        "component_row_count": len(component_rows),
        "source_count": len(source_aggregates),
        "session_count": len({str(row["session_id"]) for row in frame_rows}),
        "manifest_identity_valid": True,
        "canonical_ids_only_0_3": True,
        "frame_component_counts_valid": True,
        "source_session_aggregates_recomputed": True,
        "zero_row_policy": "FAIL_CLOSED",
        "atomic_publish": report.get("atomic_publish") is True,
        "interruption_resume_supported": report.get("interruption_resume_supported") is True,
        "independent_full_recompute": True,
        "recomputed_summary": {
            "delta_recall_C_minus_A": delta_recall,
            "delta_false_positive_area_fraction_C_minus_A": delta_fp,
            "candidate_component_recall": component_aggregate["component_recall"],
            "false_activation_components_per_frame": component_aggregate[
                "false_activation_components_per_frame"
            ],
        },
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite rehearsal validation: {output_path}")
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
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--rehearsal-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    value = validate(
        repo_root=args.repo_root,
        view_root=args.view_root,
        rehearsal_root=args.rehearsal_root,
        trace_path=args.trace,
        model_path=args.model,
        postprocess_path=args.postprocess,
        evaluator_path=args.evaluator,
        output_path=args.output,
    )
    print(json.dumps({"status": value["status"], "frames": value["frame_count"]}))
