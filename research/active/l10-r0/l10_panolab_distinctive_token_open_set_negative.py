#!/usr/bin/env python3
"""Attack the fixed distinctive-token branch with four exact-roster-absent images."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from paddleocr import PaddleOCR

import l10_panolab_distinctive_edit_token_posthoc as successor
import l10_panolab_federated_lexical_appearance_confirmation as frame_eval
import l10_panolab_track_lexical_ledger as ledger
import l10_panolab_track_token_bank_fresh as lexical


PROTOCOL_SCHEMA = "blindassist-l10-panolab-distinctive-token-open-set-negative-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-distinctive-token-open-set-negative-result-v1"


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
    inputs = protocol["inputs"]
    selection_path = lexical.verify(inputs["negative_selection"])
    selection = lexical.load(selection_path)
    manifest_path = lexical.verify(inputs["negative_materialization"])
    manifest = lexical.load(manifest_path)
    truth = lexical.load(lexical.verify(inputs["negative_truth"]))
    posthoc_protocol = lexical.load(lexical.verify(inputs["successor_protocol"]))
    lexical.verify(inputs["fresh_positive_result"])
    lexical_protocol = lexical.load(lexical.verify(inputs["frozen_ocr_protocol"]))
    projection_protocol = lexical.load(lexical.verify(inputs["orientation_projection_protocol"]))
    lexical.require(manifest["selection_sha256"] == lexical.sha256(selection_path), "SELECTION_LINK_MISMATCH")
    lexical.require(truth["summary"]["exact_positive_roster_entity_absent"] == 4, "NEGATIVE_TRUTH_NOT_FOUR")
    lexical.require(all(not row["exact_positive_roster_entity_present"] for row in truth["episodes"]), "POSITIVE_PRESENT_IN_NEGATIVE")
    lexical.require(protocol["name_token_contract"] == posthoc_protocol["name_token_contract"], "NAME_TOKEN_CONTRACT_CHANGED")
    stable_keys = {
        "minimum_ocr_row_score", "minimum_target_token_length", "maximum_edit_distance",
        "apostrophe_or_token_fragment_rule", "long_token_length_for_two_units",
        "minimum_distinctive_evidence_units", "evidence_weighting", "authority",
    }
    lexical.require(
        {key: protocol["distinctive_edit_token_contract"][key] for key in stable_keys}
        == {key: posthoc_protocol["distinctive_edit_token_contract"][key] for key in stable_keys},
        "SUCCESSOR_MATCHING_CONTRACT_CHANGED",
    )

    runtime = {
        "python": sys.version.split()[0],
        "numpy": importlib.metadata.version("numpy"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "paddleocr": importlib.metadata.version("paddleocr"),
        "paddlex": importlib.metadata.version("paddlex"),
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
    lexical.require(len(target_ids) == len(set(target_ids)) == 13, "TARGET_ROSTER_NOT_THIRTEEN")
    image_index = {row["episode_id"]: row for row in manifest["images"]}
    truth_index = {row["episode_id"]: row for row in truth["episodes"]}
    rows = []
    false_accepts = 0
    matched_target_count = 0
    started = time.perf_counter()
    for episode in selection["episodes"]:
        episode_id = episode["episode_id"]
        receipt = image_index[episode_id]
        frame = {
            "key": f"{episode_id}_single",
            "target": {
                "entrance_node": {
                    "id": episode["main_entrance_node"]["id"],
                    "lon_lat": [episode["main_entrance_node"]["lon"], episode["main_entrance_node"]["lat"]],
                }
            },
            "panorama": {
                "local_path": receipt["path"], "image_sha256": receipt["sha256"],
                "image_bytes": receipt["bytes"], "image_size": receipt["image_size"],
                "provider_item": episode["provider_item"],
            },
        }
        evaluated = frame_eval.evaluate_frame(pipeline, frame, projection_protocol, lexical_protocol)
        matches = {
            target["episode_id"]: successor.match_target(
                [evaluated], target["entity_name"], protocol["name_token_contract"],
                protocol["distinctive_edit_token_contract"],
            )
            for target in targets
        }
        matched_targets = [target_id for target_id in target_ids if matches[target_id]["matched"]]
        matched_target_count += len(matched_targets)
        candidate = matched_targets[0] if len(matched_targets) == 1 else None
        false_accepts += int(candidate is not None)
        rows.append(
            {
                "episode_id": episode_id,
                "source_city": episode["source_city"],
                "negative_target_name": truth_index[episode_id]["negative_target_name"],
                "truth": truth_index[episode_id]["truth"],
                "frame": evaluated,
                "target_matches": matches,
                "lexical": {
                    "matched_targets": matched_targets,
                    "candidate": candidate,
                    "route": "WRONG_SEARCH_PRIORITY_CANDIDATE" if candidate else "UNKNOWN_KEEP_SEARCHING",
                    "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                    "portal_ownership_binding": None,
                },
            }
        )

    metrics = {
        "preexisting_pixel_but_governed_ocr_unseen_negative_queries": len(rows),
        "negative_source_cities": len({row["source_city"] for row in rows}),
        "positive_roster": len(target_ids),
        "wrong_target_lexical_trials": len(rows) * len(target_ids),
        "matched_target_rows": matched_target_count,
        "negative_false_accepts": false_accepts,
        "negative_rejected_unknown": len(rows) - false_accepts,
        "negative_rejection_rate": (len(rows) - false_accepts) / len(rows),
        "portal_ownership_bindings_emitted": 0,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    gate = {
        "four_exact_roster_absent_queries": len(rows) == 4,
        "three_source_cities": metrics["negative_source_cities"] == 3,
        "fifty_two_wrong_target_trials": metrics["wrong_target_lexical_trials"] == 52,
        "zero_matched_target_rows": matched_target_count == 0,
        "zero_negative_false_accepts": false_accepts == 0,
        "four_of_four_unknown": metrics["negative_rejected_unknown"] == 4,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "PREEXISTING_PIXEL_FIRST_GOVERNED_OCR_OPEN_SET_NEGATIVE_DEVELOPMENT_ONLY",
        "protocol": str(protocol_path), "protocol_sha256": lexical.sha256(protocol_path),
        "evaluator_sha256": lexical.sha256(Path(__file__).resolve()), "runtime": runtime,
        "metrics": metrics, "gate": gate, "rows": rows, "claim_boundary": protocol["claim_boundary"],
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
