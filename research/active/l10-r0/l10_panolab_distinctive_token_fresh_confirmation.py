#!/usr/bin/env python3
"""Fresh confirmation of the fixed distinctive exact-or-one-edit token branch."""

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


PROTOCOL_SCHEMA = "blindassist-l10-panolab-distinctive-token-fresh-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-distinctive-token-fresh-confirmation-result-v1"


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
    source_protocol = lexical.load(lexical.verify(inputs["fresh_source_protocol"]))
    selection_path = lexical.verify(inputs["fresh_selection"])
    selection = lexical.load(selection_path)
    manifest_path = lexical.verify(inputs["fresh_materialization"])
    manifest = lexical.load(manifest_path)
    truth = lexical.load(lexical.verify(inputs["fresh_truth"]))
    posthoc_protocol = lexical.load(lexical.verify(inputs["posthoc_successor_protocol"]))
    posthoc_result = lexical.load(lexical.verify(inputs["posthoc_successor_result"]))
    lexical_protocol = lexical.load(lexical.verify(inputs["frozen_ocr_protocol"]))
    projection_protocol = lexical.load(lexical.verify(inputs["orientation_projection_protocol"]))
    lexical.require(posthoc_result["gate"]["passed"], "POSTHOC_SUCCESSOR_GATE_NOT_MET")
    stable_contract_keys = {
        "minimum_ocr_row_score",
        "minimum_target_token_length",
        "maximum_edit_distance",
        "apostrophe_or_token_fragment_rule",
        "long_token_length_for_two_units",
        "minimum_distinctive_evidence_units",
        "evidence_weighting",
        "association",
        "authority",
    }
    lexical.require(
        {key: protocol["distinctive_edit_token_contract"][key] for key in stable_contract_keys}
        == {key: posthoc_protocol["distinctive_edit_token_contract"][key] for key in stable_contract_keys},
        "SUCCESSOR_MATCHING_CONTRACT_CHANGED",
    )
    lexical.require(protocol["name_token_contract"] == posthoc_protocol["name_token_contract"], "NAME_TOKEN_CONTRACT_CHANGED")
    lexical.require(manifest["selection_sha256"] == lexical.sha256(selection_path), "SELECTION_LINK_MISMATCH")
    lexical.require(
        truth["inputs"]["selection"]["sha256"] == lexical.sha256(selection_path)
        and truth["inputs"]["materialization"]["sha256"] == lexical.sha256(manifest_path),
        "TRUTH_LINK_MISMATCH",
    )
    lexical.require(
        selection["selected_ocr_calls_before_freeze"] == 0
        and truth["ocr_calls_on_selected_images_before_truth_freeze"] == 0
        and protocol["ocr_calls_on_selected_images_before_protocol_freeze"] == 0,
        "SELECTED_OCR_PREEXPOSED",
    )
    lexical.require(all(row["valid_for_fixed_replay"] for row in truth["episodes"]), "INVALID_HUMAN_TRUTH")
    lexical.require(source_protocol["selection"]["panel_size"] == 2, "SOURCE_PANEL_SIZE_CHANGED")

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
    evaluated_ids = protocol["evaluated_episodes"]
    lexical.require(len(target_ids) == len(set(target_ids)) == 13, "TARGET_ROSTER_NOT_THIRTEEN")
    lexical.require(target_ids[-2:] == evaluated_ids, "FRESH_TARGET_ORDER_MISMATCH")
    episode_index = {row["episode_id"]: row for row in selection["episodes"]}
    image_index = {row["item_id"]: row for row in manifest["images"]}
    truth_index = {row["episode_id"]: row for row in truth["episodes"]}
    lexical.require(set(evaluated_ids) == set(episode_index) == set(truth_index), "FRESH_EPISODE_MISMATCH")

    rows = []
    correct = wrong = no_match = ambiguous = 0
    wrong_target_matches = 0
    started = time.perf_counter()
    for episode_id in evaluated_ids:
        episode = episode_index[episode_id]
        members = sorted(episode["queries"], key=lambda row: int(row["sequence_index"]))
        lexical.require(
            [row["relation_to_anchor"] for row in members] == ["prev", "anchor", "next"],
            f"QUERY_SEQUENCE_INVALID:{episode_id}",
        )
        frames = [
            frame_eval.evaluate_frame(
                pipeline,
                frame_eval.build_frame(episode, member, image_index[member["item_id"]]),
                projection_protocol,
                lexical_protocol,
            )
            for member in members
        ]
        matches = {
            target["episode_id"]: successor.match_target(
                frames,
                target["entity_name"],
                protocol["name_token_contract"],
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
            no_match += 1
        else:
            candidate = None
            state = "AMBIGUOUS_MULTIPLE_MATCHES"
            ambiguous += 1
        rows.append(
            {
                "episode_id": episode_id,
                "producer_stratum": episode["producer_stratum"],
                "source_city": episode["source_city"],
                "target_way_id": episode["target_way_id"],
                "target_name": episode["target_name"],
                "human_truth": truth_index[episode_id]["human_truth"],
                "frames": frames,
                "target_matches": matches,
                "lexical": {
                    "state": state,
                    "matched_targets": matched_targets,
                    "candidate": candidate,
                    "wrong_target_matches": wrong_matches,
                    "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                    "portal_ownership_binding": None,
                },
            }
        )

    metrics = {
        "pixel_and_ocr_unseen_episodes": len(rows),
        "producer_strata": len({row["producer_stratum"] for row in rows}),
        "query_frames": sum(len(row["frames"]) for row in rows),
        "target_roster": len(target_ids),
        "lexical_correct_wrong_no_match_ambiguous": [correct, wrong, no_match, ambiguous],
        "positive_coverage": correct / len(rows),
        "wrong_target_lexical_trials": len(rows) * (len(target_ids) - 1),
        "wrong_target_lexical_matches": wrong_target_matches,
        "portal_ownership_bindings_emitted": 0,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    gate = {
        "two_valid_fresh_sequences_evaluated": len(rows) == 2,
        "two_producer_strata": metrics["producer_strata"] == 2,
        "six_query_frames": metrics["query_frames"] == 6,
        "two_of_two_unique_correct": correct == 2,
        "zero_wrong": wrong == 0,
        "zero_unknown": no_match == 0 and ambiguous == 0,
        "zero_wrong_target_lexical_matches": wrong_target_matches == 0,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "PIXEL_AND_OCR_UNSEEN_TARGET_WAY_ITEM_DISJOINT_SAME_PROVIDER_DEVELOPMENT_LEXICAL_ROUTING_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": lexical.sha256(protocol_path),
        "evaluator_sha256": lexical.sha256(Path(__file__).resolve()),
        "runtime": runtime,
        "metrics": metrics,
        "gate": gate,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
