#!/usr/bin/env python3
"""Provider-disjoint Mapillary transfer of the frozen distinctive-token branch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import PaddleOCR

import l10_panolab_distinctive_edit_token_posthoc as successor
import l10_panolab_track_lexical_ledger as ledger
import l10_panolab_track_token_bank_fresh as lexical


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-mapillary-distinctive-token-provider-transfer-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-mapillary-distinctive-token-provider-transfer-result-v1"


def crop_facade(image: np.ndarray, interval: list[int], contract: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    lexical.require(width == 2 * height, "IMAGE_NOT_EQUIRECTANGULAR_2_TO_1")
    pixels_per_degree = width / 360.0
    center = int(round((int(interval[0]) + int(interval[1])) / 2.0))
    half_width = int(round(float(contract["horizontal_half_fov_degrees"]) * pixels_per_degree))
    top = int(round(height / 2 - float(contract["above_horizon_degrees"]) * pixels_per_degree))
    bottom = int(round(height / 2 + float(contract["below_horizon_degrees"]) * pixels_per_degree))
    columns = np.arange(center - half_width, center + half_width) % width
    crop = np.ascontiguousarray(image[top:bottom, columns, :])
    lexical.require(crop.size > 0, "EMPTY_FACADE_CROP")
    return crop, {
        "human_interval_x": interval,
        "center_x": center,
        "top_y": top,
        "bottom_y_exclusive": bottom,
        "width": int(crop.shape[1]),
        "height": int(crop.shape[0]),
        "pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    lexical.require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = lexical.load(protocol_path)
    lexical.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    lexical.require(lexical.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    source = lexical.load(lexical.verify(protocol["inputs"]["mapillary_source"]))
    successor_protocol = lexical.load(lexical.verify(protocol["inputs"]["successor_protocol"]))
    lexical.verify(protocol["inputs"]["fresh_panolab_result"])
    lexical_protocol = lexical.load(lexical.verify(protocol["inputs"]["frozen_ocr_protocol"]))
    lexical.require(source["summary"]["prior_governed_ocr_calls"] == 0, "MAPILLARY_OCR_PREEXPOSED")
    lexical.require(protocol["name_token_contract"] == successor_protocol["name_token_contract"], "NAME_TOKEN_CONTRACT_CHANGED")
    stable_keys = {
        "minimum_ocr_row_score", "minimum_target_token_length", "maximum_edit_distance",
        "apostrophe_or_token_fragment_rule", "long_token_length_for_two_units",
        "minimum_distinctive_evidence_units", "evidence_weighting", "authority",
    }
    lexical.require(
        {key: protocol["distinctive_edit_token_contract"][key] for key in stable_keys}
        == {key: successor_protocol["distinctive_edit_token_contract"][key] for key in stable_keys},
        "SUCCESSOR_MATCHING_CONTRACT_CHANGED",
    )
    runtime = {
        "python": sys.version.split()[0], "numpy": importlib.metadata.version("numpy"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "paddleocr": importlib.metadata.version("paddleocr"), "paddlex": importlib.metadata.version("paddlex"),
        "onnxruntime-gpu": importlib.metadata.version("onnxruntime-gpu"),
    }
    lexical.require(runtime == protocol["runtime"]["versions"], f"RUNTIME_MISMATCH:{runtime}")
    detection_root = ledger.verify_model(lexical_protocol["models"]["medium_detection"])
    recognition_root = ledger.verify_model(lexical_protocol["models"]["medium_recognition"])
    pipeline = PaddleOCR(
        text_detection_model_dir=str(detection_root), text_recognition_model_dir=str(recognition_root),
        engine="onnxruntime", device="cpu", use_doc_orientation_classify=False,
        use_doc_unwarping=False, use_textline_orientation=False,
    )

    targets = protocol["target_roster"]
    target_ids = [row["episode_id"] for row in targets]
    lexical.require(len(target_ids) == len(set(target_ids)) == 17, "TARGET_ROSTER_NOT_SEVENTEEN")
    source_index = {row["episode_id"]: row for row in source["episodes"]}
    rows = []
    correct = wrong = unknown = ambiguous = 0
    wrong_target_matches = 0
    started = time.perf_counter()
    for episode_id in protocol["evaluated_episodes"]:
        episode = source_index[episode_id]
        image_spec = episode["image"]
        image_path = ledger.resolve(image_spec["path"])
        lexical.require(lexical.sha256(image_path) == image_spec["sha256"], f"IMAGE_HASH_MISMATCH:{episode_id}")
        lexical.require(image_path.stat().st_size == int(image_spec["bytes"]), f"IMAGE_BYTES_MISMATCH:{episode_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        lexical.require(image is not None and [image.shape[1], image.shape[0]] == image_spec["size"], f"IMAGE_DECODE_MISMATCH:{episode_id}")
        crop, crop_receipt = crop_facade(image, episode["frozen_facade_interval_x"], source["crop_contract"])
        ocr_rows, ocr_receipt = ledger.run_ocr(pipeline, crop)
        frame = {
            "frame_key": f"{episode_id}_mapillary",
            "panorama_item_id": episode["mapillary_image_id"],
            "ocr_rows": ocr_rows,
        }
        matches = {
            target["episode_id"]: successor.match_target(
                [frame], target["entity_name"], protocol["name_token_contract"],
                protocol["distinctive_edit_token_contract"],
            )
            for target in targets
        }
        matched_targets = [target_id for target_id in target_ids if matches[target_id]["matched"]]
        wrong_matches = [target_id for target_id in matched_targets if target_id != episode_id]
        wrong_target_matches += len(wrong_matches)
        if matched_targets == [episode_id]:
            candidate = episode_id
            state = "UNIQUE_OWN_TARGET_MATCH"
            correct += 1
        elif len(matched_targets) == 1:
            candidate = matched_targets[0]
            state = "UNIQUE_WRONG_TARGET_MATCH"
            wrong += 1
        elif not matched_targets:
            candidate = None
            state = "NO_MATCH"
            unknown += 1
        else:
            candidate = None
            state = "AMBIGUOUS_MULTIPLE_MATCHES"
            ambiguous += 1
        rows.append(
            {
                "episode_id": episode_id, "city": episode["city"],
                "target_name": episode["target_name"], "target_way_id": episode["target_way_id"],
                "human_entity_truth": episode["human_entity_truth"],
                "portal_truth_boundary": episode["portal_truth_boundary"],
                "crop_receipt": crop_receipt, "ocr": ocr_receipt, "ocr_rows": ocr_rows,
                "target_matches": matches,
                "lexical": {
                    "state": state, "matched_targets": matched_targets, "candidate": candidate,
                    "wrong_target_matches": wrong_matches,
                    "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                    "portal_ownership_binding": None,
                },
            }
        )

    metrics = {
        "provider_disjoint_preexisting_pixel_first_ocr_episodes": len(rows),
        "cities": len({row["city"] for row in rows}), "mapillary_sequences": len(rows),
        "target_roster": len(target_ids),
        "correct_wrong_unknown_ambiguous": [correct, wrong, unknown, ambiguous],
        "positive_coverage": correct / len(rows),
        "wrong_target_trials": len(rows) * (len(target_ids) - 1),
        "wrong_target_matches": wrong_target_matches,
        "portal_ownership_bindings_emitted": 0,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    gate = {
        "four_mapillary_sequences": len(rows) == 4, "two_cities": metrics["cities"] == 2,
        "minimum_two_correct": correct >= 2, "zero_wrong": wrong == 0,
        "zero_ambiguous": ambiguous == 0, "zero_wrong_target_matches": wrong_target_matches == 0,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "PROVIDER_AND_CITY_DISJOINT_PREEXISTING_PIXEL_HUMAN_WINDOW_FIRST_OCR_DEVELOPMENT_ONLY",
        "protocol": str(protocol_path), "protocol_sha256": lexical.sha256(protocol_path),
        "evaluator_sha256": lexical.sha256(Path(__file__).resolve()), "runtime": runtime,
        "metrics": metrics, "gate": gate, "rows": rows, "claim_boundary": protocol["claim_boundary"],
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
