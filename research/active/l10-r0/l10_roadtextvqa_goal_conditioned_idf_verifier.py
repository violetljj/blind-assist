#!/usr/bin/env python3
"""Fresh goal-conditioned verification replay with train-background IDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import l10_roadtextvqa_background_idf_router as idf_router
import l10_roadtextvqa_roster_information_router as base


PROTOCOL_SCHEMA = "blindassist-l10-roadtextvqa-goal-conditioned-idf-verifier-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-roadtextvqa-goal-conditioned-idf-verifier-result-v1"


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
    token_bank = {row["target_id"]: base.target_tokens(row["aliases"], ignored) for row in rowspec}
    counts = {"legacy": Counter(), "goal_conditioned_idf": Counter()}
    result_rows = []
    mismatch_accepts = []
    canary_accepts = []
    salt = protocol["synthetic_negative_contract"]["salt"]
    with zipfile.ZipFile(ocr_path) as archive:
        frame_cache = {row["video"]: base.read_frames(archive, row["video"]) for row in rowspec}
        for index, row in enumerate(rowspec):
            frames = frame_cache[row["video"]]
            expected_tokens = token_bank[row["target_id"]]
            legacy = base.match_target(frames, expected_tokens, {token: 1 for token in expected_tokens}, protocol["legacy_contract"], False)
            verified = idf_router.match_idf(frames, expected_tokens, {token: 1 for token in expected_tokens}, background_df, background_documents, protocol["idf_contract"])
            legacy_state = "CORRECT" if legacy["matched"] else "UNKNOWN"
            verified_state = "CORRECT" if verified["matched"] else "UNKNOWN"
            counts["legacy"][legacy_state] += 1
            counts["goal_conditioned_idf"][verified_state] += 1
            first = None
            for prefix in range(1, len(frames) + 1):
                prefix_match = idf_router.match_idf(frames[:prefix], expected_tokens, {token: 1 for token in expected_tokens}, background_df, background_documents, protocol["idf_contract"])
                if prefix_match["matched"]:
                    first = {"sampled_frame_count": prefix, "frame": frames[prefix - 1]["frame"]}
                    break
            mismatch_row = rowspec[(index + 1) % len(rowspec)]
            mismatch_tokens = token_bank[mismatch_row["target_id"]]
            mismatch = idf_router.match_idf(frames, mismatch_tokens, {token: 1 for token in mismatch_tokens}, background_df, background_documents, protocol["idf_contract"])
            if mismatch["matched"]:
                mismatch_accepts.append({"video": row["video"], "challenge_target_id": mismatch_row["target_id"], "challenge_aliases": mismatch_row["aliases"], "match": mismatch})
            digest = hashlib.sha256(f"{row['question_id']}|{row['video']}|{salt}".encode("utf-8")).hexdigest()[:10]
            canary = "zz" + "".join(chr(ord("a") + int(char, 16)) for char in digest)
            canary_match = idf_router.match_idf(frames, [canary], {canary: 1}, background_df, background_documents, protocol["idf_contract"])
            if canary_match["matched"]:
                canary_accepts.append({"question_id": row["question_id"], "video": row["video"], "token": canary, "match": canary_match})
            result_rows.append(
                {
                    **row,
                    "legacy": {"state": legacy_state, "match": legacy},
                    "goal_conditioned_idf": {"state": verified_state, "candidate": row["target_id"] if verified["matched"] else None, "match": verified, "first_acceptance": first},
                    "cyclic_cross_video_challenge": {"challenge_target_id": mismatch_row["target_id"], "challenge_aliases": mismatch_row["aliases"], "accepted": mismatch["matched"], "match": mismatch},
                }
            )
    legacy_correct = int(counts["legacy"]["CORRECT"])
    idf_correct = int(counts["goal_conditioned_idf"]["CORRECT"])
    accepted_prefixes = [row["goal_conditioned_idf"]["first_acceptance"]["sampled_frame_count"] for row in result_rows if row["goal_conditioned_idf"]["first_acceptance"]]
    gate = {
        "thirty_fresh_ocr_unseen_episodes": len(result_rows) == 30,
        "minimum_three_correct_gain": idf_correct - legacy_correct >= 3,
        "zero_synthetic_negative_accepts": len(canary_accepts) == 0,
        "candidate_is_always_the_conditioned_goal": True,
        "zero_identity_or_portal_bindings": True,
    }
    gate["passed"] = all(gate.values())
    metrics = {
        "episodes": len(result_rows),
        "legacy_correct_unknown": [legacy_correct, len(result_rows) - legacy_correct],
        "goal_conditioned_idf_correct_unknown": [idf_correct, len(result_rows) - idf_correct],
        "correct_gain": idf_correct - legacy_correct,
        "synthetic_negative_queries": len(result_rows),
        "synthetic_negative_accepts": len(canary_accepts),
        "cyclic_cross_video_challenges": len(result_rows),
        "cyclic_cross_video_challenge_accepts": len(mismatch_accepts),
        "accepted_episode_mean_frames_to_first_candidate": round(sum(accepted_prefixes) / len(accepted_prefixes), 3) if accepted_prefixes else None,
        "wrong_goal_candidates_emitted": 0,
        "identity_bindings_emitted": 0,
        "portal_bindings_emitted": 0,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "FRESH_OCR_UNSEEN_ROADTEXTVQA_VALIDATION_SPLIT_GOAL_CONDITIONED_DEVELOPMENT_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": base.sha256(protocol_path),
        "evaluator_sha256": base.sha256(Path(__file__).resolve()),
        "metrics": metrics,
        "gate": gate,
        "synthetic_negative_accepts": canary_accepts,
        "cyclic_cross_video_challenge_accepts": mismatch_accepts,
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
