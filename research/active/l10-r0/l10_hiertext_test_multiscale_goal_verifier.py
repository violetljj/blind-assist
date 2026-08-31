#!/usr/bin/env python3
"""Evaluate a goal-conditioned OCR verifier on exhaustive HierText test truth."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr import RapidOCR


PROTOCOL_SCHEMA = "blindassist-l10-hiertext-test-multiscale-goal-verifier-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-hiertext-test-multiscale-goal-verifier-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return "".join(character.lower() for character in text if character.isascii() and character.isalnum())


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, character in enumerate(left, 1):
        current = [index]
        for other_index, other_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[other_index] + 1,
                    previous[other_index - 1] + (character != other_character),
                )
            )
        previous = current
    return previous[-1]


def bbox(vertices: list[list[float]]) -> list[float]:
    points = np.asarray(vertices, dtype=np.float32)
    return [
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    ]


def overlap_over_truth(candidate: list[list[float]], truth: list[float]) -> float:
    points = np.asarray(candidate, dtype=np.float32)
    cx0, cy0 = float(points[:, 0].min()), float(points[:, 1].min())
    cx1, cy1 = float(points[:, 0].max()), float(points[:, 1].max())
    tx0, ty0, tx1, ty1 = truth
    intersection = max(0.0, min(cx1, tx1) - max(cx0, tx0)) * max(0.0, min(cy1, ty1) - max(cy0, ty0))
    return intersection / max(1.0, (tx1 - tx0) * (ty1 - ty0))


def all_words(image: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for paragraph in image.get("paragraphs", []):
        for line in paragraph.get("lines", []):
            if line.get("legible") is not True or line.get("handwritten") is True:
                continue
            for word in line.get("words", []):
                token = normalize(str(word.get("text", "")))
                if word.get("legible") is True and word.get("handwritten") is not True and token:
                    rows.append({"token": token, "text": word["text"], "vertices": word["vertices"]})
    return rows


def information_bits(token: str, document_frequency: dict[str, int], documents: int) -> float:
    return math.log2((documents + 1) / (document_frequency.get(token, 0) + 1))


def select_cohort(
    annotations: list[dict[str, Any]],
    document_frequency: dict[str, int],
    documents: int,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    cohort = []
    for image in annotations:
        width, height = int(image["image_width"]), int(image["image_height"])
        words = all_words(image)
        counts = Counter(row["token"] for row in words)
        eligible = []
        for row in words:
            token = row["token"]
            x0, y0, x1, y1 = bbox(row["vertices"])
            if not (
                contract["minimum_token_length"] <= len(token) <= contract["maximum_token_length"]
                and token.isalpha()
                and counts[token] == 1
                and (x1 - x0) / width >= contract["minimum_normalized_width"]
                and (y1 - y0) / height >= contract["minimum_normalized_height"]
                and information_bits(token, document_frequency, documents) >= contract["minimum_background_information_bits"]
            ):
                continue
            eligible.append(
                {
                    "token": token,
                    "display_text": row["text"],
                    "truth_bbox": [x0, y0, x1, y1],
                    "normalized_area": ((x1 - x0) * (y1 - y0)) / (width * height),
                }
            )
        if eligible:
            target = max(eligible, key=lambda row: (row["normalized_area"], row["token"]))
            cohort.append(
                {
                    "image_id": image["image_id"],
                    "width": width,
                    "height": height,
                    "truth_tokens": sorted({row["token"] for row in words}),
                    **target,
                }
            )
        if len(cohort) == contract["cohort_size"]:
            break
    require(len(cohort) == contract["cohort_size"], "INSUFFICIENT_GT_ONLY_COHORT")
    for index, row in enumerate(cohort):
        absent = cohort[(index + 1) % len(cohort)]["token"]
        require(absent not in row["truth_tokens"], f"CYCLIC_QUERY_PRESENT:{row['image_id']}:{absent}")
        row["absent_query"] = absent
    return cohort


def image_path(images: Path, image_id: str) -> Path:
    matches = list(images.rglob(f"{image_id}.*"))
    require(len(matches) == 1, f"IMAGE_NOT_UNIQUE:{image_id}:{len(matches)}")
    return matches[0]


def detections(output: Any, offset_x: float = 0.0, offset_y: float = 0.0, scale: float = 1.0, source: str = "full") -> list[dict[str, Any]]:
    rows = []
    boxes = output.boxes if output.boxes is not None else ()
    texts = output.txts if output.txts is not None else ()
    scores = output.scores if output.scores is not None else ()
    for box_points, text, score in zip(boxes, texts, scores):
        box_array = np.asarray(box_points, dtype=np.float32)
        box_array[:, 0] = box_array[:, 0] / scale + offset_x
        box_array[:, 1] = box_array[:, 1] / scale + offset_y
        rows.append(
            {
                "box": box_array.astype(float).tolist(),
                "text": str(text),
                "token": normalize(str(text)),
                "confidence": float(score),
                "source": source,
            }
        )
    return rows


def build_cache(cohort: list[dict[str, Any]], images: Path, models: Path, cache: Path, contract: dict[str, Any]) -> dict[str, Any]:
    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(models),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    payload: dict[str, Any] = {
        "schema": "blindassist-l10-hiertext-test-rapidocr-cache-v1",
        "backend": "RapidOCR 3.9.2 / PP-OCRv6 small / ONNX Runtime CPU",
        "images": {},
    }
    started = time.perf_counter()
    for index, row in enumerate(cohort, 1):
        path = image_path(images, row["image_id"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        require(image is not None, f"IMAGE_DECODE_FAILED:{path}")
        full = detections(engine(image), source="full")
        height, width = image.shape[:2]
        fraction = float(contract["tile_fraction"])
        tile_width, tile_height = round(width * fraction), round(height * fraction)
        scale = float(contract["tile_scale"])
        tiled = []
        for name, x0, y0 in (
            ("top_left", 0, 0),
            ("top_right", width - tile_width, 0),
            ("bottom_left", 0, height - tile_height),
            ("bottom_right", width - tile_width, height - tile_height),
        ):
            crop = image[y0 : y0 + tile_height, x0 : x0 + tile_width]
            enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            tiled.extend(detections(engine(enlarged), x0, y0, scale, name))
        payload["images"][row["image_id"]] = {
            "path": str(path.resolve()),
            "shape": list(image.shape),
            "full": full,
            "tiled": tiled,
        }
        print(json.dumps({"ocr_image": index, "total": len(cohort), "image_id": row["image_id"]}), flush=True)
    payload["ocr_wall_s"] = round(time.perf_counter() - started, 3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def best_match(query: str, candidates: list[dict[str, Any]], contract: dict[str, Any], allow_edit_one: bool) -> dict[str, Any] | None:
    ranked = []
    for index, candidate in enumerate(candidates):
        token = candidate["token"]
        if candidate["confidence"] < contract["minimum_ocr_confidence"] or not token:
            continue
        distance = edit_distance(query, token)
        admissible = distance == 0 or (allow_edit_one and len(query) >= contract["minimum_edit_one_length"] and len(token) >= contract["minimum_edit_one_length"] and distance == 1)
        if admissible:
            ranked.append((distance, -candidate["confidence"], candidate["source"], index, candidate))
    if not ranked:
        return None
    distance, _, _, _, candidate = min(ranked)
    return {**candidate, "edit_distance": distance}


def positive_state(match: dict[str, Any] | None, truth_bbox: list[float], overlap_threshold: float) -> str:
    if match is None:
        return "UNKNOWN"
    return "CORRECT" if overlap_over_truth(match["box"], truth_bbox) >= overlap_threshold else "WRONG_CARRIER"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol_path, output_path = args.protocol.resolve(), args.output.resolve()
    require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    inputs = {key: Path(value).resolve() for key, value in protocol["inputs"].items()}
    require(sha256(inputs["ground_truth"]) == protocol["input_sha256"]["ground_truth"], "GROUND_TRUTH_HASH_MISMATCH")
    require(sha256(inputs["image_archive"]) == protocol["input_sha256"]["image_archive"], "IMAGE_ARCHIVE_HASH_MISMATCH")
    require(sha256(inputs["training_token_df"]) == protocol["input_sha256"]["training_token_df"], "TOKEN_DF_HASH_MISMATCH")
    for name, expected in protocol["model_sha256"].items():
        require(sha256(inputs["models"] / name) == expected, f"MODEL_HASH_MISMATCH:{name}")
    with gzip.open(inputs["ground_truth"], "rt", encoding="utf-8") as stream:
        annotations = json.load(stream)["annotations"]
    df_payload = json.loads(inputs["training_token_df"].read_text(encoding="utf-8"))
    document_frequency = {key: int(value) for key, value in df_payload["document_frequency"].items()}
    documents = int(df_payload["train_videos"])
    cohort = select_cohort(annotations, document_frequency, documents, protocol["selection_contract"])
    cache_path = inputs["ocr_cache"]
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else build_cache(
        cohort, inputs["images"], inputs["models"], cache_path, protocol["algorithm_contract"]
    )
    require(set(cache["images"]) == {row["image_id"] for row in cohort}, "CACHE_COHORT_MISMATCH")
    counts = {"baseline": Counter(), "successor": Counter()}
    false_accepts = {"baseline": [], "successor": []}
    rows = []
    for row in cohort:
        observed = cache["images"][row["image_id"]]
        baseline = best_match(row["token"], observed["full"], protocol["algorithm_contract"], False)
        successor = best_match(row["token"], observed["full"] + observed["tiled"], protocol["algorithm_contract"], True)
        baseline_state = positive_state(baseline, row["truth_bbox"], protocol["algorithm_contract"]["minimum_truth_overlap"])
        successor_state = positive_state(successor, row["truth_bbox"], protocol["algorithm_contract"]["minimum_truth_overlap"])
        counts["baseline"][baseline_state] += 1
        counts["successor"][successor_state] += 1
        baseline_absent = best_match(row["absent_query"], observed["full"], protocol["algorithm_contract"], False)
        successor_absent = best_match(row["absent_query"], observed["full"] + observed["tiled"], protocol["algorithm_contract"], True)
        if baseline_absent:
            false_accepts["baseline"].append({"image_id": row["image_id"], "query": row["absent_query"], "match": baseline_absent})
        if successor_absent:
            false_accepts["successor"].append({"image_id": row["image_id"], "query": row["absent_query"], "match": successor_absent})
        rows.append(
            {
                "image_id": row["image_id"],
                "target": row["token"],
                "target_display": row["display_text"],
                "target_background_information_bits": round(information_bits(row["token"], document_frequency, documents), 6),
                "truth_bbox": row["truth_bbox"],
                "absent_query": row["absent_query"],
                "baseline": {"state": baseline_state, "match": baseline},
                "successor": {"state": successor_state, "match": successor},
            }
        )
    baseline_correct = counts["baseline"]["CORRECT"]
    successor_correct = counts["successor"]["CORRECT"]
    gate = {
        "thirty_gt_only_selected_test_images": len(rows) == 30,
        "minimum_three_correct_carrier_gain": successor_correct - baseline_correct >= 3,
        "no_wrong_carrier_regression": counts["successor"]["WRONG_CARRIER"] <= counts["baseline"]["WRONG_CARRIER"],
        "zero_exhaustive_truth_absent_accepts": len(false_accepts["successor"]) == 0,
        "zero_identity_or_portal_bindings": True,
    }
    gate["passed"] = all(gate.values())
    metrics = {
        "images": len(rows),
        "positive_queries": len(rows),
        "truth_absent_queries": len(rows),
        "baseline_correct_wrong_unknown": [counts["baseline"]["CORRECT"], counts["baseline"]["WRONG_CARRIER"], counts["baseline"]["UNKNOWN"]],
        "successor_correct_wrong_unknown": [counts["successor"]["CORRECT"], counts["successor"]["WRONG_CARRIER"], counts["successor"]["UNKNOWN"]],
        "correct_carrier_gain": successor_correct - baseline_correct,
        "baseline_truth_absent_accepts": len(false_accepts["baseline"]),
        "successor_truth_absent_accepts": len(false_accepts["successor"]),
        "identity_bindings_emitted": 0,
        "portal_bindings_emitted": 0,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "PREDECLARED_HIERTEXT_TEST_PIXEL_AND_EXHAUSTIVE_WORD_TRUTH_DEVELOPMENT_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "ocr_cache": str(cache_path),
        "ocr_cache_sha256": sha256(cache_path),
        "metrics": metrics,
        "gate": gate,
        "truth_absent_accepts": false_accepts,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"decision": result["decision"], "metrics": metrics, "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
