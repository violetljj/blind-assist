"""Execute the one frozen 36-point DDRNet Development refinement search."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from . import PROTOCOL_ID
from .canonicalizer import sha256_file
from .postprocess import filter_candidate_by_class
from ..dual_loop_segmentation_candidate_utility import evaluate_candidate_utility as base
from ..dual_loop_segmentation_candidate_utility.component_metrics import (
    aggregate_confusion,
    component_metrics,
    pixel_metrics,
)


CLASS_TO_ID = {
    "walkable": 0,
    "boundary_step_curb": 1,
    "obstacle": 2,
    "unknown_nonwalkable": 3,
}
READINESS = {
    "delta_recall_C_minus_A": (">=", 0.06),
    "delta_false_positive_area_fraction_C_minus_A": ("<=", 0.04),
    "candidate_component_recall": (">=", 0.55),
    "false_activation_components_per_frame": ("<=", 2.5),
    "consistent_sessions": (">=", 3),
}


class RefinementError(ValueError):
    """Raised when the frozen refinement identity or result is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RefinementError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RefinementError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RefinementError(f"expected object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise RefinementError(f"zero rows: {path}")
    return rows


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
        "component_precision": float(hit_predicted / predicted) if predicted else (1.0 if truth == 0 else None),
        "component_recall": float(hit_truth / truth) if truth else (1.0 if predicted == 0 else None),
        "false_activation_component_count": false_count,
        "false_activation_components_per_frame": float(false_count / len(rows)),
    }


def _candidate_id(parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"DDRNET23_SLIM_R2P0_POSTPROCESS_{digest}"


def _gate(actual: float | int, operator: str, threshold: float | int) -> dict[str, Any]:
    passed = float(actual) >= float(threshold) if operator == ">=" else float(actual) <= float(threshold)
    return {
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _normalized_margin(name: str, actual: float | int) -> float:
    operator, threshold = READINESS[name]
    if operator == ">=":
        return float((float(actual) - float(threshold)) / max(abs(float(threshold)), 1.0))
    return float((float(threshold) - float(actual)) / max(abs(float(threshold)), 1.0))


def run(
    *,
    repo_root: Path,
    search_config_path: Path,
    view_root: Path,
    trace_path: Path,
    model_path: Path,
    output_root: Path,
    threads: int,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to((repo_root / "artifacts.local").resolve())
    except ValueError as exc:
        raise RefinementError("refinement output must stay under artifacts.local") from exc
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite refinement output: {output_root}")
    search = _read_json(search_config_path.resolve())
    if (
        search.get("protocol_id") != PROTOCOL_ID
        or search.get("status") != "SEARCH_SPACE_FROZEN_BEFORE_EXECUTION"
        or search.get("one_search_only") is not True
    ):
        raise RefinementError("search config identity/status mismatch")
    if sha256_file(model_path.resolve()) != search.get("base_model_sha256"):
        raise RefinementError("DDRNet base model SHA256 mismatch")
    names = (
        "minimum_component_area_pixels",
        "minimum_component_confidence_median",
        "minimum_component_margin_median",
        "minimum_component_bottom_fraction",
    )
    values = [search["parameters"][name] for name in names]
    combinations = [dict(zip(names, items, strict=True)) for items in itertools.product(*values)]
    if len(combinations) != int(search.get("combination_count", -1)) or len(combinations) != 36:
        raise RefinementError("frozen refinement space must contain exactly 36 candidates")
    view_root = view_root.resolve()
    view_receipt = _read_json(view_root / "receipt.json")
    view_manifest = view_root / str(view_receipt["manifest"])
    if sha256_file(view_manifest) != view_receipt.get("manifest_sha256"):
        raise RefinementError("canonical view manifest identity mismatch")
    dev_rows = [row for row in _read_jsonl(view_manifest) if row.get("role") == "dev"]
    if len(dev_rows) != 200:
        raise RefinementError(f"refinement requires 200 canonical dev rows, got {len(dev_rows)}")
    traces = base.load_trace(trace_path.resolve())
    segmenter = base.TFLiteSegmenter(model_path.resolve(), threads=threads)
    prepared: list[dict[str, Any]] = []
    for row in dev_rows:
        image_path = (repo_root / str(row["image_repo_relative_path"])).resolve()
        canonical_path = (view_root / str(row["canonical_mask_path"])).resolve()
        if sha256_file(image_path) != row.get("image_sha256"):
            raise RefinementError(f"{row['id']}: image identity mismatch")
        if sha256_file(canonical_path) != row.get("canonical_mask_sha256"):
            raise RefinementError(f"{row['id']}: truth identity mismatch")
        with Image.open(image_path) as image:
            width, height = image.size
            ids, confidence, margin, _ = segmenter.infer(image)
        with Image.open(canonical_path) as image:
            truth_ids = np.asarray(image, dtype=np.uint8)
        key = (str(row["source_id"]), int(row["frame_id"]), str(row["image_sha256"]))
        trace = traces.get(key)
        if trace is None:
            raise RefinementError(f"{row['id']}: missing frozen YOLO trace")
        detector_mask = base.box_union_mask(
            trace["detections"],
            source_width=width,
            source_height=height,
        )
        truth_hazard = np.isin(truth_ids, [1, 2])
        prepared.append(
            {
                "row": row,
                "ids": ids,
                "confidence": confidence,
                "margin": margin,
                "truth_hazard": truth_hazard,
                "detector_mask": detector_mask,
                "arm_a": pixel_metrics(detector_mask, truth_hazard),
            }
        )
    candidate_results: list[dict[str, Any]] = []
    for parameters in combinations:
        config = {
            "schema_version": "blindassist.dual_loop_segmentation_r2_p0.postprocess.v1",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": _candidate_id(parameters),
            **parameters,
            **search["fixed_rules"],
            "authority": "DEVELOPMENT_SELECTION_ONLY",
        }
        arm_a_rows: list[dict[str, Any]] = []
        arm_c_rows: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        by_source: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for item in prepared:
            candidate_by_class = filter_candidate_by_class(
                ids=item["ids"],
                confidence=item["confidence"],
                margin=item["margin"],
                detector_mask=item["detector_mask"],
                class_to_id=CLASS_TO_ID,
                config=config,
            )
            candidate = (
                candidate_by_class["boundary_step_curb"]
                | candidate_by_class["obstacle"]
            )
            arm_c = pixel_metrics(
                item["detector_mask"] | candidate,
                item["truth_hazard"],
            )
            candidate_truth = item["truth_hazard"] & ~item["detector_mask"]
            components = component_metrics(candidate, candidate_truth)
            arm_a_rows.append(item["arm_a"])
            arm_c_rows.append(arm_c)
            component_rows.append(components)
            by_source[str(item["row"]["source_id"])].append((item["arm_a"], arm_c))
        arm_a = aggregate_confusion(arm_a_rows)
        arm_c = aggregate_confusion(arm_c_rows)
        components = _aggregate_components(component_rows)
        consistent_sessions = 0
        source_metrics: list[dict[str, Any]] = []
        for source_id, rows in sorted(by_source.items()):
            source_a = aggregate_confusion([row[0] for row in rows])
            source_c = aggregate_confusion([row[1] for row in rows])
            delta = float(source_c["recall"] - source_a["recall"])
            consistent = delta >= 0.0
            consistent_sessions += int(consistent)
            source_metrics.append(
                {
                    "source_id": source_id,
                    "frame_count": len(rows),
                    "delta_recall_C_minus_A": delta,
                    "consistent": consistent,
                }
            )
        metrics = {
            "delta_recall_C_minus_A": float(arm_c["recall"] - arm_a["recall"]),
            "delta_false_positive_area_fraction_C_minus_A": float(
                arm_c["false_positive_area_fraction"]
                - arm_a["false_positive_area_fraction"]
            ),
            "candidate_component_recall": components["component_recall"],
            "false_activation_components_per_frame": components[
                "false_activation_components_per_frame"
            ],
            "consistent_sessions": consistent_sessions,
        }
        gates = {
            name: _gate(metrics[name], operator, threshold)
            for name, (operator, threshold) in READINESS.items()
        }
        utility_qualified = all(value["passed"] for value in gates.values())
        minimum_margin = min(_normalized_margin(name, metrics[name]) for name in READINESS)
        candidate_results.append(
            {
                "schema_version": "blindassist.dual_loop_segmentation_r2_p0.refinement_candidate.v1",
                "protocol_id": PROTOCOL_ID,
                "candidate_id": config["candidate_id"],
                "parameters": parameters,
                "metrics": metrics,
                "readiness_utility_gates": gates,
                "all_readiness_utility_gates_passed": utility_qualified,
                "minimum_normalized_readiness_margin": minimum_margin,
                "source_metrics": source_metrics,
            }
        )
    qualified = [
        row for row in candidate_results
        if row["all_readiness_utility_gates_passed"]
    ]
    qualified.sort(
        key=lambda row: (
            -float(row["minimum_normalized_readiness_margin"]),
            float(row["metrics"]["false_activation_components_per_frame"]),
            -float(row["metrics"]["delta_recall_C_minus_A"]),
            str(row["candidate_id"]),
        )
    )
    selected = qualified[0] if qualified else None
    output_root.mkdir(parents=True)
    matrix_path = output_root / "candidate_matrix.jsonl"
    matrix_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in candidate_results
        ),
        encoding="utf-8",
        newline="\n",
    )
    selected_config: dict[str, Any] | None = None
    if selected is not None:
        selected_config = {
            "schema_version": "blindassist.dual_loop_segmentation_r2_p0.postprocess.v1",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": selected["candidate_id"],
            **selected["parameters"],
            **search["fixed_rules"],
            "authority": "DEVELOPMENT_SELECTED_REHEARSAL_REQUIRED",
            "selection_matrix_sha256": sha256_file(matrix_path),
        }
        (output_root / "selected_postprocess.json").write_text(
            json.dumps(selected_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    report = {
        "schema_version": "blindassist.dual_loop_segmentation_r2_p0.refinement_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "BOUNDED_REFINEMENT_COMPLETE",
        "formal_authority": False,
        "search_config_sha256": sha256_file(search_config_path.resolve()),
        "canonical_view_manifest_sha256": sha256_file(view_manifest),
        "dev_yolo_trace_sha256": sha256_file(trace_path.resolve()),
        "base_model_sha256": sha256_file(model_path.resolve()),
        "refinement_code_sha256": sha256_file(Path(__file__).resolve()),
        "combination_count": len(candidate_results),
        "qualified_utility_candidate_count": len(qualified),
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_candidate_metrics": selected["metrics"] if selected else None,
        "selected_postprocess_sha256": (
            sha256_file(output_root / "selected_postprocess.json")
            if selected_config is not None
            else None
        ),
        "candidate_matrix_sha256": sha256_file(matrix_path),
        "next_required_check": (
            "selected_candidate_runtime_and_consumed_rehearsal"
            if selected is not None
            else "R2_NOT_WORTH_BURNING_FRESH_HOLDOUT"
        ),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    value = run(
        repo_root=args.repo_root,
        search_config_path=args.search_config,
        view_root=args.view_root,
        trace_path=args.trace,
        model_path=args.model,
        output_root=args.output_root,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "qualified": value["qualified_utility_candidate_count"],
                "selected": value["selected_candidate_id"],
            }
        )
    )
