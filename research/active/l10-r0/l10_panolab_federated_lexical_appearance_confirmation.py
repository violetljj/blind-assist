#!/usr/bin/env python3
"""Confirm the frozen lexical-then-appearance router on new-city target sequences."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from paddleocr import PaddleOCR

import l10_panolab_track_lexical_ledger as ledger
import l10_panolab_track_token_bank_fresh as lexical


PROTOCOL_SCHEMA = "blindassist-l10-panolab-federated-lexical-appearance-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-federated-lexical-appearance-confirmation-result-v1"


def build_frame(
    episode: dict[str, Any], member: dict[str, Any], image_receipt: dict[str, Any]
) -> dict[str, Any]:
    entrance = episode["main_entrance_node"]
    return {
        "key": f"{episode['episode_id']}_{member['relation_to_anchor']}",
        "target": {
            "entrance_node": {
                "id": entrance["id"],
                "lon_lat": [entrance["lon"], entrance["lat"]],
            }
        },
        "panorama": {
            "local_path": image_receipt["path"],
            "image_sha256": image_receipt["sha256"],
            "image_bytes": image_receipt["bytes"],
            "image_size": image_receipt["image_size"],
            "provider_item": member["provider_item"],
        },
    }


def evaluate_frame(
    pipeline: PaddleOCR,
    frame: dict[str, Any],
    projection_protocol: dict[str, Any],
    lexical_protocol: dict[str, Any],
) -> dict[str, Any]:
    panorama = frame["panorama"]
    path = ledger.resolve(panorama["local_path"])
    lexical.require(lexical.sha256(path) == panorama["image_sha256"], f"IMAGE_HASH:{frame['key']}")
    lexical.require(path.stat().st_size == int(panorama["image_bytes"]), f"IMAGE_BYTES:{frame['key']}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    lexical.require(image is not None, f"IMAGE_DECODE:{path}")
    lexical.require([image.shape[1], image.shape[0]] == panorama["image_size"], f"IMAGE_SIZE:{frame['key']}")
    ray = ledger.strict_ray(frame, projection_protocol)
    crop, window = ledger.entrance_window(image, ray, lexical_protocol["observation_window"])
    rows, ocr = ledger.run_ocr(pipeline, crop)
    return {
        "frame_key": frame["key"],
        "relation": frame["key"].rsplit("_", 1)[-1],
        "panorama_item_id": panorama["provider_item"]["id"],
        "strict_entrance_ray": ray,
        "window": window,
        "ocr": ocr,
        "ocr_rows": rows,
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
    lexical.require(
        lexical.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"],
        "EVALUATOR_HASH_MISMATCH",
    )

    inputs = protocol["inputs"]
    lexical_protocol = lexical.load(lexical.verify(inputs["frozen_lexical_protocol"]))
    selection_path = lexical.verify(inputs["confirmation_selection"])
    selection = lexical.load(selection_path)
    manifest_path = lexical.verify(inputs["confirmation_materialization"])
    manifest = lexical.load(manifest_path)
    truth = lexical.load(lexical.verify(inputs["confirmation_truth"]))
    appearance_result = lexical.load(lexical.verify(inputs["confirmation_appearance_result"]))
    projection_protocol = lexical.load(lexical.verify(inputs["orientation_projection_protocol"]))
    lexical.require(
        manifest["selection_sha256"] == lexical.sha256(selection_path),
        "CONFIRMATION_SELECTION_LINK_MISMATCH",
    )
    lexical.require(
        truth["inputs"]["selection"]["sha256"] == lexical.sha256(selection_path)
        and truth["inputs"]["materialization"]["sha256"] == lexical.sha256(manifest_path),
        "CONFIRMATION_TRUTH_LINK_MISMATCH",
    )
    lexical.require(
        appearance_result["decision"]
        == "L10_PANOLAB_FEDERATED_NEW_CITY_TEMPORAL_APPEARANCE_ROUTER_DEVELOPMENT_GATE_MET",
        "EXPECTED_CONFIRMATION_APPEARANCE_RESULT_NOT_FROZEN",
    )
    lexical.require(protocol["ocr_calls_on_selected_images_before_protocol_freeze"] == 0, "OCR_PREEXPOSED")

    runtime = {
        "python": sys.version.split()[0],
        "numpy": importlib.metadata.version("numpy"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "paddleocr": importlib.metadata.version("paddleocr"),
        "paddlex": importlib.metadata.version("paddlex"),
        "onnxruntime-gpu": importlib.metadata.version("onnxruntime-gpu"),
    }
    lexical.require(runtime == lexical_protocol["runtime"]["versions"], f"RUNTIME_MISMATCH:{runtime}")
    detection_root = ledger.verify_model(lexical_protocol["models"]["medium_detection"])
    recognition_root = ledger.verify_model(lexical_protocol["models"]["medium_recognition"])
    pipeline = PaddleOCR(
        text_detection_model_dir=str(detection_root),
        text_recognition_model_dir=str(recognition_root),
        engine="onnxruntime",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    targets = protocol["target_roster"]
    target_ids = [row["episode_id"] for row in targets]
    lexical.require(len(target_ids) == len(set(target_ids)) == 8, "TARGET_ROSTER_NOT_EIGHT")
    evaluated_ids = protocol["evaluated_episodes"]
    lexical.require(target_ids[-2:] == evaluated_ids, "EVALUATED_TARGET_ORDER_MISMATCH")
    episode_index = {row["episode_id"]: row for row in selection["episodes"]}
    image_index = {row["item_id"]: row for row in manifest["images"]}
    truth_index = {row["episode_id"]: row for row in truth["episodes"]}
    appearance_index = {row["episode_id"]: row for row in appearance_result["rows"]}
    lexical.require(
        set(evaluated_ids) == set(episode_index) == set(truth_index) == set(appearance_index),
        "EVALUATED_EPISODE_MISMATCH",
    )

    result_rows = []
    lexical_correct = 0
    lexical_wrong = 0
    lexical_no_match = 0
    lexical_ambiguous = 0
    combined_correct = 0
    combined_wrong = 0
    combined_unknown = 0
    wrong_target_lexical_matches = 0
    started = time.perf_counter()
    for episode_id in evaluated_ids:
        episode = episode_index[episode_id]
        members = sorted(episode["queries"], key=lambda row: int(row["sequence_index"]))
        lexical.require(
            [row["relation_to_anchor"] for row in members] == ["prev", "anchor", "next"],
            f"QUERY_SEQUENCE_INVALID:{episode_id}",
        )
        frames = [
            evaluate_frame(
                pipeline,
                build_frame(episode, member, image_index[member["item_id"]]),
                projection_protocol,
                lexical_protocol,
            )
            for member in members
        ]
        token_banks = {
            target["episode_id"]: lexical.token_bank_match(
                frames,
                target["entity_name"],
                lexical_protocol["lexical_match"],
                lexical_protocol["long_short_token_bank"],
            )
            for target in targets
        }
        matched_targets = [target_id for target_id in target_ids if token_banks[target_id]["matched"]]
        wrong_matches = [target_id for target_id in matched_targets if target_id != episode_id]
        wrong_target_lexical_matches += len(wrong_matches)
        if matched_targets == [episode_id]:
            lexical_candidate = episode_id
            lexical_state = "UNIQUE_OWN_TARGET_MATCH"
            lexical_correct += 1
        elif len(matched_targets) == 1:
            lexical_candidate = matched_targets[0]
            lexical_state = "UNIQUE_WRONG_TARGET_MATCH"
            lexical_wrong += 1
        elif not matched_targets:
            lexical_candidate = None
            lexical_state = "NO_MATCH"
            lexical_no_match += 1
        else:
            lexical_candidate = None
            lexical_state = "AMBIGUOUS_MULTIPLE_MATCHES"
            lexical_ambiguous += 1

        appearance_candidate = appearance_index[episode_id]["acceptance"]["candidate"]
        if lexical_state == "UNIQUE_OWN_TARGET_MATCH":
            candidate = lexical_candidate
            route = "TEMPORAL_EXACT_TOKEN_SEARCH_PRIORITY_CANDIDATE"
        elif lexical_state == "NO_MATCH":
            candidate = appearance_candidate
            route = (
                "TEMPORAL_APPEARANCE_SEARCH_PRIORITY_CANDIDATE"
                if candidate is not None
                else "UNKNOWN_KEEP_SEARCHING"
            )
        else:
            candidate = None
            route = "UNKNOWN_LEXICAL_CONFLICT_KEEP_SEARCHING"
        if candidate == episode_id:
            combined_correct += 1
        elif candidate is None:
            combined_unknown += 1
        else:
            combined_wrong += 1
        result_rows.append(
            {
                "episode_id": episode_id,
                "target_name": episode["target_name"],
                "target_way_id": episode["target_way_id"],
                "source_city": episode["source_city"],
                "frames": frames,
                "token_banks": token_banks,
                "lexical": {
                    "state": lexical_state,
                    "matched_targets": matched_targets,
                    "candidate": lexical_candidate,
                    "wrong_target_matches": wrong_matches,
                },
                "appearance_fallback": {
                    "prior_route": appearance_index[episode_id]["acceptance"]["route"],
                    "prior_candidate": appearance_candidate,
                },
                "combined_router": {
                    "route": route,
                    "candidate": candidate,
                    "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                    "portal_ownership_binding": None,
                },
            }
        )

    appearance_correct = sum(
        appearance_index[episode_id]["acceptance"]["candidate"] == episode_id
        for episode_id in evaluated_ids
    )
    metrics = {
        "confirmation_episodes": len(evaluated_ids),
        "query_frames": sum(len(row["frames"]) for row in result_rows),
        "target_roster": len(target_ids),
        "lexical_correct_wrong_no_match_ambiguous": [
            lexical_correct,
            lexical_wrong,
            lexical_no_match,
            lexical_ambiguous,
        ],
        "appearance_fallback_correct_wrong_unknown": [
            appearance_correct,
            0,
            len(evaluated_ids) - appearance_correct,
        ],
        "combined_correct_wrong_unknown": [combined_correct, combined_wrong, combined_unknown],
        "combined_correct_delta_from_appearance": combined_correct - appearance_correct,
        "wrong_target_lexical_trials": len(evaluated_ids) * (len(target_ids) - 1),
        "wrong_target_lexical_matches": wrong_target_lexical_matches,
        "portal_ownership_bindings_emitted": 0,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    gate = {
        "two_confirmation_sequences_evaluated": len(result_rows) == 2,
        "six_query_frames": metrics["query_frames"] == 6,
        "minimum_one_unique_own_lexical_match": lexical_correct >= 1,
        "two_of_two_combined_correct": combined_correct == 2,
        "zero_combined_wrong": combined_wrong == 0,
        "zero_combined_unknown": combined_unknown == 0,
        "zero_wrong_target_lexical_matches": wrong_target_lexical_matches == 0,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "NEW_CITY_TARGET_WAY_ITEM_DISJOINT_OCR_UNSEEN_SAME_PROVIDER_DEVELOPMENT_ROUTING_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": lexical.sha256(protocol_path),
        "evaluator_sha256": lexical.sha256(Path(__file__).resolve()),
        "runtime": runtime,
        "metrics": metrics,
        "gate": gate,
        "rows": result_rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"result": str(output_path), "decision": result["decision"], "metrics": metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
