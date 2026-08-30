#!/usr/bin/env python3
"""Evaluate multi-scale address OCR plus mask-level portal topology on SEVN.

This is a representation-changing successor to l10_sevn_pixel_replay.py.  It
keeps the frozen 24 episodes, action-selected observations, renderer, models,
and truth boundary, while replacing full-frame OCR plus nearest bounding box
with overlapping magnified OCR tiles and an upper-mask credential topology.

The official ZIP is streamed in memory.  The evaluator writes only one durable
JSON result and no panorama, crop, tile, mask image, overlay, or OCR cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from rapidocr import RapidOCR
from ultralytics import YOLO

from l10_panolab import require, sha256_file, utc_now
from l10_sevn_panolab import validate_public, validate_truth
import l10_sevn_pixel_replay as v1


def ocr_pass_specs(image: np.ndarray, contract: dict[str, Any]) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    specs = [{
        "pass_id": "FULL",
        "x": 0,
        "y": 0,
        "width": width,
        "height": height,
        "output_width": width,
        "output_height": height,
        "magnification": 1.0,
    }]
    tile = contract["tile_pass"]
    tile_width = round(width * float(tile["width_fraction"]))
    tile_height = round(height * float(tile["height_fraction"]))
    x_positions = np.linspace(0, width - tile_width, int(tile["columns"])).round().astype(int)
    y_positions = np.linspace(0, height - tile_height, int(tile["rows"])).round().astype(int)
    for row, y in enumerate(y_positions):
        for column, x in enumerate(x_positions):
            specs.append({
                "pass_id": f"TILE_R{row + 1}_C{column + 1}",
                "x": int(x),
                "y": int(y),
                "width": tile_width,
                "height": tile_height,
                "output_width": width,
                "output_height": height,
                "magnification": round(width / tile_width, 6),
            })
    return specs


def map_polygon(points: Any, spec: dict[str, Any]) -> np.ndarray:
    array = np.asarray(points, dtype=float).copy()
    array[:, 0] = spec["x"] + array[:, 0] * spec["width"] / spec["output_width"]
    array[:, 1] = spec["y"] + array[:, 1] * spec["height"] / spec["output_height"]
    return array


def ocr_words_for_pass(engine: RapidOCR, image: np.ndarray, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    tile = image[
        spec["y"]:spec["y"] + spec["height"],
        spec["x"]:spec["x"] + spec["width"],
    ]
    if tile.shape[1] != spec["output_width"] or tile.shape[0] != spec["output_height"]:
        observed = cv2.resize(
            tile,
            (spec["output_width"], spec["output_height"]),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        observed = tile
    started = time.perf_counter()
    output = engine(observed, return_word_box=True)
    seconds = time.perf_counter() - started
    boxes = output.boxes if output.boxes is not None else ()
    texts = output.txts if output.txts is not None else ()
    scores = output.scores if output.scores is not None else ()
    word_lines = output.word_results if output.word_results is not None else ()
    rows: list[dict[str, Any]] = []
    for index, (line_box, line_text, line_score) in enumerate(zip(boxes, texts, scores)):
        words = word_lines[index] if index < len(word_lines) else ()
        if words:
            source_rows = words
        else:
            source_rows = ((line_text, line_score, line_box),)
        for text, score, box in source_rows:
            mapped = map_polygon(box, spec)
            rows.append({
                "text": str(text),
                "canonical": v1.canonical_token(text),
                "score": float(score),
                "box_xyxy": v1.polygon_box(mapped),
                "source_pass": spec["pass_id"],
                "source_magnification": spec["magnification"],
            })
    return rows, seconds


def normalized_center_distance(a: list[float], b: list[float], width: int, height: int) -> float:
    ax, ay = v1.box_center(a)
    bx, by = v1.box_center(b)
    return math.hypot(ax - bx, ay - by) / math.hypot(width, height)


def deduplicate_ocr(rows: list[dict[str, Any]], width: int, height: int, contract: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = sorted(
        (row for row in rows if row["canonical"]),
        key=lambda row: (-row["score"], row["source_pass"], tuple(row["box_xyxy"])),
    )
    kept: list[dict[str, Any]] = []
    for source_row in ordered:
        row = {**source_row, "observed_passes": [source_row["source_pass"]]}
        duplicate = None
        for prior in kept:
            if (
                row["canonical"] == prior["canonical"]
                and (
                    v1.box_iou(row["box_xyxy"], prior["box_xyxy"]) >= float(contract["minimum_box_iou"])
                    or normalized_center_distance(row["box_xyxy"], prior["box_xyxy"], width, height)
                    <= float(contract["maximum_normalized_center_distance"])
                )
            ):
                duplicate = prior
                break
        if duplicate is None:
            kept.append(row)
        else:
            duplicate["observed_passes"] = sorted(set(duplicate["observed_passes"] + row["observed_passes"]))
    kept.sort(key=lambda row: (row["canonical"], -row["score"], row["source_pass"], tuple(row["box_xyxy"])))
    return kept


def multiscale_ocr(engine: RapidOCR, image: np.ndarray, contract: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pass_receipts = []
    for spec in ocr_pass_specs(image, contract):
        pass_rows, seconds = ocr_words_for_pass(engine, image, spec)
        rows.extend(pass_rows)
        pass_receipts.append({
            **spec,
            "raw_word_count": len(pass_rows),
            "seconds": round(seconds, 6),
        })
    height, width = image.shape[:2]
    deduplicated = deduplicate_ocr(rows, width, height, contract["deduplication"])
    return deduplicated, {
        "passes": pass_receipts,
        "raw_word_count": len(rows),
        "deduplicated_word_count": len(deduplicated),
        "seconds": round(sum(row["seconds"] for row in pass_receipts), 6),
    }


def mask_digest(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(mask.reshape(-1)).tobytes())
    return digest.hexdigest()


def mask_bounds(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    require(len(xs) > 0, "empty portal mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(a, b).sum())
    area_a = int(a.sum())
    area_b = int(b.sum())
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    containment = intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0
    return iou, containment


def portal_masks(
    model: YOLO,
    postprocessor: Any,
    image: np.ndarray,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    result = model.predict(
        source=image,
        device=int(contract["cuda_device"]),
        conf=float(contract["predict_confidence"]),
        imgsz=int(contract["image_size"]),
        verbose=False,
    )[0]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if result.boxes is None or result.masks is None:
        return [], {"raw_detection_count": 0, "author_portal_count": 0, "deduplicated_portal_count": 0, "seconds": round(seconds, 6)}

    raw = [
        {
            "cls": int(class_id),
            "box": [float(value) for value in box],
            "conf": float(confidence),
            "idx": index,
        }
        for index, (box, class_id, confidence) in enumerate(zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ))
    ]
    final = postprocessor.finalize(raw, out_conf=float(contract["output_confidence"]))
    candidates = []
    for row in final:
        if int(row["cls"]) not in {0, 1}:
            continue
        polygon = np.asarray(result.masks.xy[int(row["idx"])], dtype=float)
        if len(polygon) < 3:
            continue
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        if not mask.any():
            continue
        bounds = mask_bounds(mask)
        summary = {
            "candidate_id": f"P{int(row['idx']):03d}",
            "class_id": int(row["cls"]),
            "class_name": "doorway" if int(row["cls"]) == 0 else "door",
            "confidence": float(row["conf"]),
            "detection_box_xyxy": [float(value) for value in row["box"]],
            "mask_box_xyxy": bounds,
            "mask_area_pixels": int(mask.sum()),
            "mask_sha256": mask_digest(mask),
        }
        candidates.append({"summary": summary, "mask": mask})

    candidates.sort(key=lambda row: (
        -row["summary"]["confidence"],
        row["summary"]["class_id"],
        tuple(row["summary"]["mask_box_xyxy"]),
    ))
    kept: list[dict[str, Any]] = []
    removed = []
    for candidate in candidates:
        duplicate_of = None
        for prior in kept:
            iou, containment = mask_overlap(candidate["mask"], prior["mask"])
            if iou >= float(contract["deduplication"]["minimum_mask_iou"]) or containment >= float(
                contract["deduplication"]["minimum_mask_containment"]
            ):
                duplicate_of = prior["summary"]["candidate_id"]
                removed.append({
                    "candidate_id": candidate["summary"]["candidate_id"],
                    "duplicate_of": duplicate_of,
                    "mask_iou": round(iou, 8),
                    "mask_containment": round(containment, 8),
                })
                break
        if duplicate_of is None:
            kept.append(candidate)
    return kept, {
        "raw_detection_count": len(raw),
        "author_portal_count": len(candidates),
        "deduplicated_portal_count": len(kept),
        "mask_duplicates_removed": removed,
        "seconds": round(seconds, 6),
    }


def text_box_slice(box: list[float], width: int, height: int) -> tuple[slice, slice]:
    x1 = max(0, min(width - 1, math.floor(box[0])))
    y1 = max(0, min(height - 1, math.floor(box[1])))
    x2 = max(x1 + 1, min(width, math.ceil(box[2])))
    y2 = max(y1 + 1, min(height, math.ceil(box[3])))
    return slice(y1, y2), slice(x1, x2)


def portal_topology(candidate: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    mask = candidate["mask"]
    x1, y1, x2, y2 = candidate["summary"]["mask_box_xyxy"]
    portal_width = max(1, x2 - x1)
    portal_height = max(1, y2 - y1)
    cutoff = min(mask.shape[0], round(y1 + float(contract["upper_mask_fraction"]) * portal_height))
    upper = mask.copy()
    upper[cutoff:, :] = 0
    if not upper.any():
        upper = mask
        cutoff = y2
    inverse = (upper == 0).astype(np.uint8)
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(upper)
    return {
        "distance": distance,
        "upper_centroid_x": float(xs.mean()),
        "upper_centroid_y": float(ys.mean()),
        "upper_cutoff_y": cutoff,
        "horizontal_radius": max(
            int(contract["minimum_radius_pixels"]),
            min(int(contract["maximum_radius_pixels"]), round(float(contract["horizontal_radius_fraction"]) * portal_width)),
        ),
        "vertical_radius": max(
            int(contract["minimum_radius_pixels"]),
            min(int(contract["maximum_radius_pixels"]), round(float(contract["vertical_radius_fraction"]) * portal_height)),
        ),
    }


def pair_topology(
    text: dict[str, Any],
    candidate: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, Any]:
    height, width = candidate["mask"].shape
    x1, y1, x2, y2 = candidate["summary"]["mask_box_xyxy"]
    tx, ty = v1.box_center(text["box_xyxy"])
    y_slice, x_slice = text_box_slice(text["box_xyxy"], width, height)
    distance_pixels = float(topology["distance"][y_slice, x_slice].min())
    radius = math.hypot(topology["horizontal_radius"], topology["vertical_radius"])
    distance_ratio = distance_pixels / radius if radius else float("inf")
    expanded_x = x1 - topology["horizontal_radius"] <= tx <= x2 + topology["horizontal_radius"]
    expanded_y = y1 - topology["vertical_radius"] <= ty <= topology["upper_cutoff_y"] + topology["vertical_radius"]
    admissible = bool(expanded_x and expanded_y and distance_ratio <= 1.0)
    lateral_offset = abs(tx - topology["upper_centroid_x"]) / max(1, x2 - x1)
    vertical_offset = abs(ty - y1) / max(1, y2 - y1)
    if tx < x1:
        relation = "UPPER_LEFT"
    elif tx > x2:
        relation = "UPPER_RIGHT"
    elif ty < y1:
        relation = "ABOVE"
    else:
        relation = "OVER_UPPER_MASK"
    return {
        "admissible": admissible,
        "relation": relation,
        "upper_mask_distance_pixels": round(distance_pixels, 6),
        "upper_mask_distance_ratio": round(distance_ratio, 8),
        "lateral_offset_portal_widths": round(lateral_offset, 8),
        "vertical_offset_portal_heights": round(vertical_offset, 8),
        "horizontal_radius_pixels": topology["horizontal_radius"],
        "vertical_radius_pixels": topology["vertical_radius"],
        "upper_mask_cutoff_y": topology["upper_cutoff_y"],
    }


def infer_binding(
    image: np.ndarray,
    mission: dict[str, Any],
    ocr_engine: RapidOCR,
    portal_model: YOLO,
    postprocessor: Any,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ocr_rows, ocr_receipt = multiscale_ocr(ocr_engine, image, contract["ocr"])
    portals, portal_receipt = portal_masks(portal_model, postprocessor, image, contract["portal_model"])
    target = v1.canonical_token(mission["house_number"])
    exact = [
        row for row in ocr_rows
        if row["canonical"] == target and row["score"] >= float(contract["ocr"]["minimum_score"])
    ]
    topology_by_id = {
        row["summary"]["candidate_id"]: portal_topology(row, contract["association"])
        for row in portals
    }
    pairs = []
    pair_audit = []
    for text in exact:
        for portal in portals:
            candidate_id = portal["summary"]["candidate_id"]
            topology = pair_topology(text, portal, topology_by_id[candidate_id])
            audit = {
                "text": text,
                "portal_id": candidate_id,
                "portal_confidence": portal["summary"]["confidence"],
                "topology": topology,
            }
            pair_audit.append(audit)
            if topology["admissible"]:
                key = (
                    topology["upper_mask_distance_ratio"],
                    topology["lateral_offset_portal_widths"],
                    topology["vertical_offset_portal_heights"],
                    -portal["summary"]["confidence"],
                    -text["score"],
                    candidate_id,
                )
                pairs.append((key, text, portal, topology))
    pairs.sort(key=lambda row: row[0])
    selected = None
    if not exact:
        state = "UNKNOWN_TARGET_TEXT"
    elif not portals:
        state = "UNKNOWN_PORTAL_MASK"
    elif not pairs:
        state = "UNKNOWN_NO_ADMISSIBLE_MASK_TOPOLOGY"
    else:
        _, text, portal, topology = pairs[0]
        state = "PIXEL_BOUND_MASK_PORTAL_PROPOSAL"
        selected = {
            "text": text,
            "portal": portal["summary"],
            "topology": topology,
        }
    output = {
        "state": state,
        "target_house_number_canonical": target,
        "exact_target_text_candidates": exact,
        "exact_target_text_full_pass": any("FULL" in row["observed_passes"] for row in exact),
        "exact_target_text_tile_pass": any(
            any(source_pass != "FULL" for source_pass in row["observed_passes"])
            for row in exact
        ),
        "deduplicated_ocr_words": ocr_rows,
        "portal_mask_candidates": [row["summary"] for row in portals],
        "topology_pair_audit": pair_audit,
        "selected_binding": selected,
        "runtime_receipt": {
            "ocr": ocr_receipt,
            "portal": portal_receipt,
        },
    }
    return output, {row["summary"]["candidate_id"]: row["mask"] for row in portals}


def truth_mask_metrics(mask: np.ndarray, truth_box: list[float], contract: dict[str, Any]) -> dict[str, Any]:
    height, width = mask.shape
    x1 = max(0, min(width - 1, math.floor(truth_box[0])))
    y1 = max(0, min(height - 1, math.floor(truth_box[1])))
    x2 = max(x1 + 1, min(width, math.ceil(truth_box[2])))
    y2 = max(y1 + 1, min(height, math.ceil(truth_box[3])))
    intersection = int(mask[y1:y2, x1:x2].sum())
    mask_area = int(mask.sum())
    truth_area = (x2 - x1) * (y2 - y1)
    mask_precision = intersection / mask_area if mask_area else 0.0
    truth_coverage = intersection / truth_area if truth_area else 0.0
    ys, xs = np.nonzero(mask)
    centroid_x = float(xs.mean()) if len(xs) else -1.0
    centroid_y = float(ys.mean()) if len(ys) else -1.0
    centroid_in_truth = x1 <= centroid_x <= x2 and y1 <= centroid_y <= y2
    overlap_authorized = (
        mask_precision >= float(contract["minimum_selected_mask_precision"])
        and truth_coverage >= float(contract["minimum_target_box_coverage"])
    )
    correct = bool(centroid_in_truth or overlap_authorized)
    return {
        "intersection_pixels": intersection,
        "selected_mask_area_pixels": mask_area,
        "target_box_area_pixels": truth_area,
        "selected_mask_precision": round(mask_precision, 8),
        "target_box_coverage": round(truth_coverage, 8),
        "selected_mask_centroid_xy": [round(centroid_x, 6), round(centroid_y, 6)],
        "selected_mask_centroid_in_target_box": centroid_in_truth,
        "overlap_rule_authorized": overlap_authorized,
        "correct": correct,
    }


def summarize(rows: list[dict[str, Any]], gate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    total = len(rows)
    truth_visible = sum(row["evaluation"]["target_house_number_visible"] for row in rows)
    exact = sum(row["evaluation"]["exact_target_text"] for row in rows)
    exact_visible = sum(
        row["evaluation"]["exact_target_text"] and row["evaluation"]["target_house_number_visible"]
        for row in rows
    )
    full = sum(row["runtime_output"]["exact_target_text_full_pass"] for row in rows)
    tile = sum(row["runtime_output"]["exact_target_text_tile_pass"] for row in rows)
    tile_only = sum(
        row["runtime_output"]["exact_target_text_tile_pass"]
        and not row["runtime_output"]["exact_target_text_full_pass"]
        for row in rows
    )
    joined = sum(row["runtime_output"]["selected_binding"] is not None for row in rows)
    correct = sum(row["evaluation"]["outcome"] == "CORRECT_TARGET_DOOR" for row in rows)
    wrong = sum(row["evaluation"]["outcome"] == "WRONG_DOOR" for row in rows)
    unknown = total - correct - wrong
    v1_correct = [row for row in rows if row["baseline_v1_outcome"] == "CORRECT_TARGET_DOOR"]
    v1_noncorrect = [row for row in rows if row["baseline_v1_outcome"] != "CORRECT_TARGET_DOOR"]
    retained = sum(row["evaluation"]["outcome"] == "CORRECT_TARGET_DOOR" for row in v1_correct)
    recovered = sum(row["evaluation"]["outcome"] == "CORRECT_TARGET_DOOR" for row in v1_noncorrect)
    new_wrong = sum(
        row["evaluation"]["outcome"] == "WRONG_DOOR" and row["baseline_v1_outcome"] != "WRONG_DOOR"
        for row in rows
    )
    repaired_wrong = sum(
        row["baseline_v1_outcome"] == "WRONG_DOOR" and row["evaluation"]["outcome"] == "CORRECT_TARGET_DOOR"
        for row in rows
    )
    metrics = {
        "episode_count": total,
        "truth_visible_house_number_opportunities": truth_visible,
        "exact_target_text": exact,
        "exact_target_text_when_truth_visible": exact_visible,
        "exact_target_text_when_truth_visible_rate": round(exact_visible / truth_visible, 6),
        "exact_target_text_full_pass": full,
        "exact_target_text_tile_pass": tile,
        "exact_target_text_tile_only_gain": tile_only,
        "joined_mask_topology_proposal": joined,
        "correct_target_door_bindings": correct,
        "correct_target_door_binding_rate": round(correct / total, 6),
        "wrong_door_bindings": wrong,
        "unknown": unknown,
        "v1_correct_baseline": len(v1_correct),
        "v1_correct_retained": retained,
        "v1_noncorrect_to_correct": recovered,
        "v1_wrong_repaired_to_correct": repaired_wrong,
        "new_wrong_from_v1_nonwrong": new_wrong,
    }
    gates = {
        "all_episodes_rendered": total == int(gate["episode_count"]),
        "minimum_visible_text_exact_ocr": exact_visible >= int(gate["minimum_visible_text_exact_ocr"]),
        "minimum_correct_target_door_bindings": correct >= int(gate["minimum_correct_target_door_bindings"]),
        "maximum_wrong_door_bindings": wrong <= int(gate["maximum_wrong_door_bindings"]),
        "minimum_v1_correct_retained": retained >= int(gate["minimum_v1_correct_retained"]),
        "minimum_v1_noncorrect_to_correct": recovered >= int(gate["minimum_v1_noncorrect_to_correct"]),
    }
    return metrics, gates


def scenario_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for scenario in sorted({row["scenario_class"] for row in rows}):
        subset = [row for row in rows if row["scenario_class"] == scenario]
        counts = Counter(row["evaluation"]["outcome"] for row in subset)
        result[scenario] = {
            "episodes": len(subset),
            "correct_target_door": counts["CORRECT_TARGET_DOOR"],
            "wrong_door": counts["WRONG_DOOR"],
            "unknown": counts["UNKNOWN"],
            "exact_target_text": sum(row["evaluation"]["exact_target_text"] for row in subset),
            "tile_only_ocr_gain": sum(
                row["runtime_output"]["exact_target_text_tile_pass"]
                and not row["runtime_output"]["exact_target_text_full_pass"]
                for row in subset
            ),
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    source_path = args.source.resolve()
    truth_path = args.truth.resolve()
    archive_path = args.highres_archive.resolve()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"output already exists: {output_path}")
    protocol = v1.read_json(protocol_path)
    require(protocol["schema"] == "blindassist-l10-sevn-multiscale-mask-topology-protocol-v2", "protocol schema mismatch")
    require(sha256_file(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "evaluator hash mismatch")
    require(source_path == v1.verify_sha256(protocol["frozen_inputs"]["public_source"]), "source path mismatch")
    require(truth_path == v1.verify_sha256(protocol["frozen_inputs"]["evaluator_truth"]), "truth path mismatch")
    baseline_path = v1.verify_sha256(protocol["frozen_inputs"]["v1_result"])
    v1.verify_sha256(protocol["frozen_inputs"]["v1_runtime_dependency"])
    model_path = v1.verify_sha256(protocol["models"]["portal_model"]["weights"])
    postprocessor_path = v1.verify_sha256(protocol["models"]["portal_model"]["postprocessor"])
    v1.verify_sha256(protocol["models"]["portal_model"]["model_card"])
    ocr_model_root = v1.resolve(protocol["models"]["ocr"]["model_root"])
    for filename, expected in protocol["models"]["ocr"]["sha256"].items():
        require(sha256_file(ocr_model_root / filename) == expected, f"OCR model hash mismatch: {filename}")

    archive = protocol["high_resolution_archive"]
    require(archive_path.is_file(), f"missing archive: {archive_path}")
    require(archive_path.stat().st_size == int(archive["bytes"]), "archive byte size mismatch")
    observed_archive_md5 = v1.md5_file(archive_path)
    require(observed_archive_md5 == archive["md5"], "archive Zenodo MD5 mismatch")
    print(json.dumps({"archive_md5_verified": observed_archive_md5}), flush=True)

    versions = {
        "torch": importlib.metadata.version("torch"),
        "ultralytics": ultralytics.__version__,
        "rapidocr": importlib.metadata.version("rapidocr"),
        "onnxruntime": importlib.metadata.version("onnxruntime"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "numpy": importlib.metadata.version("numpy"),
    }
    require(versions == protocol["runtime"]["versions"], f"runtime version mismatch: {versions}")
    require(torch.cuda.is_available(), "CUDA unavailable")

    source = v1.read_json(source_path)
    truth = v1.read_json(truth_path)
    baseline = v1.read_json(baseline_path)
    baseline_by_episode = {row["episode_id"]: row for row in baseline["episode_results"]}
    per_scenario = int(protocol["cohort"]["episodes_per_scenario"])
    validate_public(source, per_scenario)
    validate_truth(truth, source, per_scenario)
    require(len(source["episodes"]) == int(protocol["gate"]["episode_count"]), "episode count mismatch")
    require(set(baseline_by_episode) == {row["episode_id"] for row in source["episodes"]}, "V1 baseline episode mismatch")

    postprocessor = v1.load_module(postprocessor_path)
    portal_model = YOLO(str(model_path))
    expected_names = {0: "doorway", 1: "door", 2: "people", 3: "window", 4: "mirror"}
    require(portal_model.task == "segment" and portal_model.names == expected_names, "portal model ontology mismatch")
    ocr_engine = RapidOCR(params={
        "Global.model_root_dir": str(ocr_model_root),
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.intra_op_num_threads": int(protocol["pixel_contract"]["ocr"]["intra_op_threads"]),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })

    rows = []
    started = time.perf_counter()
    with zipfile.ZipFile(archive_path, "r") as archive_zip:
        for sequence, episode in enumerate(source["episodes"], start=1):
            episode_id = episode["episode_id"]
            episode_truth = truth["episodes"][episode_id]
            scenario = episode_truth["scenario_class"]
            action = protocol["action_control"]["scenario_to_action"][scenario]
            start_id = episode["start_observation_id"]
            transition = episode["transitions"][start_id][action]
            require(transition["action_executed"], f"{episode_id}: frozen action unavailable")
            observation_id = transition["to_observation_id"]
            observation = source["observations"][observation_id]
            headings = observation["viewport_headings_degrees"]
            require(len(headings) == 1, f"{episode_id}: expected one viewport")
            frame_id = int(observation["frame_id"])
            member_name = archive["member_pattern"].format(frame_id=frame_id)
            info = archive_zip.getinfo(member_name)
            encoded = archive_zip.read(info)
            panorama = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            require(panorama is not None, f"{episode_id}: panorama decode failed")
            require(list(panorama.shape) == protocol["render_contract"]["panorama_shape_hwc"], "panorama shape mismatch")
            viewport, render_receipt = v1.render_viewport(
                panorama,
                float(observation["camera_pose"]["panorama_angle_degrees"]),
                float(headings[0]),
                float(observation["horizontal_fov_degrees"]),
            )
            render_receipt.update({
                "archive_member": member_name,
                "archive_member_crc32": f"{info.CRC:08x}",
                "archive_member_bytes": info.file_size,
                "archive_member_sha256": hashlib.sha256(encoded).hexdigest(),
            })

            runtime_output, candidate_masks = infer_binding(
                viewport,
                episode["mission"],
                ocr_engine,
                portal_model,
                postprocessor,
                protocol["pixel_contract"],
            )

            observation_truth = episode_truth["observations"][observation_id]
            target_box = v1.annotation_to_viewport(
                episode_truth["target_door_annotation"],
                render_receipt,
                float(protocol["render_contract"]["annotation_canvas_width"]),
                float(protocol["render_contract"]["annotation_canvas_height"]),
            )
            selected = runtime_output["selected_binding"]
            if selected is None:
                mask_metrics = None
                outcome = "UNKNOWN"
            else:
                selected_mask = candidate_masks[selected["portal"]["candidate_id"]]
                mask_metrics = truth_mask_metrics(selected_mask, target_box, protocol["pixel_contract"]["evaluation"])
                outcome = "CORRECT_TARGET_DOOR" if mask_metrics["correct"] else "WRONG_DOOR"
            baseline_outcome = baseline_by_episode[episode_id]["evaluation"]["outcome"]
            evaluation = {
                "truth_binding_state": observation_truth["binding_state"],
                "target_house_number_visible": bool(observation_truth["target_house_number_visible"]),
                "exact_target_text": bool(runtime_output["exact_target_text_candidates"]),
                "target_door_box_xyxy": [round(value, 6) for value in target_box],
                "selected_mask_truth_metrics": mask_metrics,
                "outcome": outcome,
            }
            rows.append({
                "episode_id": episode_id,
                "scenario_class": scenario,
                "frozen_action": action,
                "observation_id": observation_id,
                "mission": episode["mission"],
                "baseline_v1_outcome": baseline_outcome,
                "render_receipt": render_receipt,
                "runtime_output": runtime_output,
                "evaluation": evaluation,
            })
            print(json.dumps({
                "episode": episode_id,
                "progress": f"{sequence}/{len(source['episodes'])}",
                "action": action,
                "v1": baseline_outcome,
                "state": runtime_output["state"],
                "outcome": outcome,
                "tile_only_ocr": runtime_output["exact_target_text_tile_pass"] and not runtime_output["exact_target_text_full_pass"],
            }, ensure_ascii=False), flush=True)

    actual_device = str(next(portal_model.model.parameters()).device)
    require(actual_device.startswith("cuda"), f"silent model fallback: {actual_device}")
    metrics, gates = summarize(rows, protocol["gate"])
    if all(gates.values()):
        decision = "L10_SEVN_MULTISCALE_MASK_TOPOLOGY_DEVELOPMENT_GATE_MET"
    elif (
        metrics["correct_target_door_bindings"] >= int(protocol["signal_floor"]["minimum_correct_bindings"])
        and metrics["wrong_door_bindings"] <= int(protocol["signal_floor"]["maximum_wrong_bindings"])
    ):
        decision = "L10_SEVN_MULTISCALE_MASK_TOPOLOGY_SIGNAL_ONLY_GATE_NOT_MET"
    else:
        decision = "L10_SEVN_MULTISCALE_MASK_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET"
    result = {
        "schema": "blindassist-l10-sevn-multiscale-mask-topology-result-v2",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "claim_scope": "CURATED_SEVN_MULTISCALE_OCR_AND_MASK_TOPOLOGY_DEVELOPMENT",
        "question": protocol["question"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "evaluator_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "v1_result": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
            "high_resolution_archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size, "md5": observed_archive_md5},
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "runtime": {
            "versions": versions,
            "torch_cuda": torch.version.cuda,
            "device": actual_device,
            "device_name": torch.cuda.get_device_name(int(protocol["pixel_contract"]["portal_model"]["cuda_device"])),
        },
        "action_boundary": protocol["action_control"]["claim_boundary"],
        "metrics": metrics,
        "scenario_metrics": scenario_metrics(rows),
        "gates": gates,
        "gate_met": all(gates.values()),
        "wall_seconds": round(time.perf_counter() - started, 4),
        "episode_results": rows,
        "retention": {
            "panorama_crop_tile_mask_or_overlay_files_written": 0,
            "reconstructible_intermediate_images_retained": 0,
            "durable_result": str(output_path),
            "source_archive_retained": str(archive_path),
        },
        "non_claims": protocol["non_claims"],
    }
    output_path.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--highres-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "decision": result["decision"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "wall_seconds": result["wall_seconds"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
