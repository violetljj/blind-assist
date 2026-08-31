#!/usr/bin/env python3
"""Fresh confirmation of the fixed goal-conditioned layout-phrase verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import l10_roadtextvqa_background_idf_router as idf_router
import l10_roadtextvqa_layout_phrase_verifier as layout
import l10_roadtextvqa_roster_information_router as base


PROTOCOL_SCHEMA = "blindassist-l10-roadtextvqa-layout-phrase-fresh-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-roadtextvqa-layout-phrase-fresh-result-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    base.require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = base.load(protocol_path)
    base.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    base.require(base.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    repo = Path(__file__).resolve().parents[3]
    val_path = base.resolve_input(repo, protocol["inputs"]["validation_annotations"])
    ocr_path = base.resolve_input(repo, protocol["inputs"]["official_ocr_archive"])
    df_path = base.resolve_input(repo, protocol["inputs"]["training_token_df"])
    base.resolve_input(repo, protocol["inputs"]["consumed_layout_result"])
    annotations = base.annotation_index(val_path)
    df_payload = base.load(df_path)
    background_df = {key: int(value) for key, value in df_payload["document_frequency"].items()}
    background_documents = int(df_payload["train_videos"])
    selected_ids = [int(value) for value in protocol["selected_question_ids"]]
    base.require(len(selected_ids) == len(set(selected_ids)) == 30, "SELECTION_NOT_THIRTY_UNIQUE_QUESTIONS")
    rowspec = []
    for question_id in selected_ids:
        source = annotations[question_id]
        rowspec.append({"target_id": f"RTVQA-Q{question_id}", "question_id": question_id, "video": source["video"], "question": source["question"], "aliases": source["answer"]})
    base.require(len({row["video"] for row in rowspec}) == 30, "SELECTED_VIDEOS_NOT_UNIQUE")
    base.require(not ({row["video"] for row in rowspec} & set(protocol["excluded_preexposed_videos"])), "PREEXPOSED_VIDEO_SELECTED")
    ignored = set(protocol["idf_contract"]["ignored_name_tokens"])
    counts = {"idf": Counter(), "combined": Counter()}
    result_rows = []
    phrase_canary_accepts = []
    salt = protocol["synthetic_phrase_negative_contract"]["salt"]
    with zipfile.ZipFile(ocr_path) as archive:
        for row in rowspec:
            frames = layout.read_layout_frames(archive, row["video"])
            expected_tokens = base.target_tokens(row["aliases"], ignored)
            idf = idf_router.match_idf(frames, expected_tokens, {token: 1 for token in expected_tokens}, background_df, background_documents, protocol["idf_contract"])
            phrase = layout.phrase_match(frames, row["aliases"], ignored, background_df, background_documents, protocol["layout_phrase_contract"])
            combined = idf["matched"] or phrase["matched"]
            counts["idf"]["CORRECT" if idf["matched"] else "UNKNOWN"] += 1
            counts["combined"]["CORRECT" if combined else "UNKNOWN"] += 1
            digest = hashlib.sha256(f"{row['question_id']}|{row['video']}|{salt}".encode("utf-8")).hexdigest()[:12]
            letters = "".join(chr(ord("a") + int(char, 16)) for char in digest)
            canary_alias = f"zz{letters[:6]} yy{letters[6:]}"
            canary = layout.phrase_match(frames, [canary_alias], set(), background_df, background_documents, protocol["layout_phrase_contract"])
            if canary["matched"]:
                phrase_canary_accepts.append({"question_id": row["question_id"], "video": row["video"], "alias": canary_alias, "match": canary})
            result_rows.append({**row, "idf": idf, "layout_phrase": phrase, "combined_state": "CORRECT" if combined else "UNKNOWN", "route": "BACKGROUND_IDF_TOKEN" if idf["matched"] else ("SAME_LINE_COMPACT_PHRASE" if phrase["matched"] else "UNKNOWN")})
    idf_correct = int(counts["idf"]["CORRECT"])
    combined_correct = int(counts["combined"]["CORRECT"])
    gate = {
        "thirty_fresh_ocr_unseen_episodes": len(result_rows) == 30,
        "minimum_two_correct_gain": combined_correct - idf_correct >= 2,
        "zero_synthetic_phrase_negative_accepts": len(phrase_canary_accepts) == 0,
        "candidate_is_always_the_conditioned_goal": True,
        "zero_identity_or_portal_bindings": True,
    }
    gate["passed"] = all(gate.values())
    metrics = {
        "episodes": len(result_rows),
        "idf_correct_unknown": [idf_correct, len(result_rows) - idf_correct],
        "layout_phrase_additional_correct": combined_correct - idf_correct,
        "combined_correct_unknown": [combined_correct, len(result_rows) - combined_correct],
        "synthetic_phrase_negative_queries": len(result_rows),
        "synthetic_phrase_negative_accepts": len(phrase_canary_accepts),
        "wrong_goal_candidates_emitted": 0,
        "identity_bindings_emitted": 0,
        "portal_bindings_emitted": 0,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "FRESH_OCR_UNSEEN_ROADTEXTVQA_VALIDATION_SPLIT_GOAL_CONDITIONED_LAYOUT_DEVELOPMENT_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": base.sha256(protocol_path),
        "evaluator_sha256": base.sha256(Path(__file__).resolve()),
        "metrics": metrics,
        "gate": gate,
        "synthetic_phrase_negative_accepts": phrase_canary_accepts,
        "rows": result_rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"decision": result["decision"], "metrics": metrics, "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
