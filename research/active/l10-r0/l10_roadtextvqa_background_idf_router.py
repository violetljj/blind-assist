#!/usr/bin/env python3
"""Fresh replay of a train-background-calibrated RoadTextVQA lexical router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import l10_roadtextvqa_roster_information_router as base


PROTOCOL_SCHEMA = "blindassist-l10-roadtextvqa-background-idf-router-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-roadtextvqa-background-idf-router-result-v1"


def match_idf(
    frames: list[dict[str, Any]],
    candidate_tokens: list[str],
    roster_df: dict[str, int],
    background_df: dict[str, int],
    background_documents: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    witnesses: dict[str, list[dict[str, Any]]] = {}
    bits: dict[str, float] = {}
    for target in candidate_tokens:
        if len(target) < int(contract["minimum_token_length"]) or roster_df.get(target) != 1:
            continue
        maximum_distance = 0 if len(target) <= int(contract["short_exact_maximum_length"]) else int(contract["long_maximum_edit_distance"])
        hits = []
        for frame in frames:
            best = None
            for observed in frame["tokens"]:
                if abs(len(observed) - len(target)) > maximum_distance:
                    continue
                distance = base.edit_distance(observed, target)
                if distance <= maximum_distance and (best is None or (distance, observed) < best):
                    best = (distance, observed)
            if best is not None:
                hits.append({"frame": frame["frame"], "observed": best[1], "edit_distance": best[0]})
        if hits:
            token_bits = math.log2((background_documents + 1) / (int(background_df.get(target, 0)) + 1))
            witnesses[target] = hits
            bits[target] = round(token_bits, 6)
    total = round(sum(bits.values()), 6)
    return {
        "matched": total >= float(contract["minimum_information_bits"]),
        "information_bits": total,
        "required_information_bits": float(contract["minimum_information_bits"]),
        "matched_tokens": sorted(witnesses),
        "token_information_bits": bits,
        "witnesses": witnesses,
    }


def evaluate(
    frames: list[dict[str, Any]],
    roster: list[dict[str, Any]],
    token_bank: dict[str, list[str]],
    roster_df: dict[str, int],
    background_df: dict[str, int],
    background_documents: int,
    contract: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    matches = {
        row["target_id"]: match_idf(frames, token_bank[row["target_id"]], roster_df, background_df, background_documents, contract)
        for row in roster
    }
    accepted = sorted(target_id for target_id, result in matches.items() if result["matched"])
    return accepted, matches


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
    df_payload = base.load(df_path)
    base.require(df_payload["schema"] == "blindassist-l10-roadtextvqa-train-token-df-v1", "UNEXPECTED_DF_SCHEMA")
    annotations = base.annotation_index(val_path)
    selected_ids = [int(value) for value in protocol["selected_question_ids"]]
    base.require(len(selected_ids) == len(set(selected_ids)) == 30, "SELECTION_NOT_THIRTY_UNIQUE_QUESTIONS")
    roster = []
    for question_id in selected_ids:
        source = annotations[question_id]
        roster.append({"target_id": f"RTVQA-Q{question_id}", "question_id": question_id, "video": source["video"], "question": source["question"], "aliases": source["answer"]})
    base.require(len({row["video"] for row in roster}) == 30, "SELECTED_VIDEOS_NOT_UNIQUE")
    base.require(not ({row["video"] for row in roster} & set(protocol["excluded_preexposed_videos"])), "PREEXPOSED_VIDEO_SELECTED")
    ignored = set(protocol["token_contract"]["ignored_name_tokens"])
    token_bank = {row["target_id"]: base.target_tokens(row["aliases"], ignored) for row in roster}
    roster_df = Counter(token for target_id in token_bank for token in set(token_bank[target_id]))
    background_df = {key: int(value) for key, value in df_payload["document_frequency"].items()}
    background_documents = int(df_payload["train_videos"])
    counts = {name: Counter() for name in ("legacy", "background_idf")}
    rows = []
    with zipfile.ZipFile(ocr_path) as archive:
        for row in roster:
            frames = base.read_frames(archive, row["video"])
            legacy_accepted, legacy_matches = base.evaluate_prefix(frames, roster, token_bank, dict(roster_df), protocol["legacy_contract"], False)
            idf_accepted, idf_matches = evaluate(frames, roster, token_bank, dict(roster_df), background_df, background_documents, protocol["token_contract"])
            legacy_state = base.outcome(row["target_id"], legacy_accepted)
            idf_state = base.outcome(row["target_id"], idf_accepted)
            counts["legacy"][legacy_state] += 1
            counts["background_idf"][idf_state] += 1
            first = None
            for prefix in range(1, len(frames) + 1):
                prefix_accepted, _ = evaluate(frames[:prefix], roster, token_bank, dict(roster_df), background_df, background_documents, protocol["token_contract"])
                if prefix_accepted:
                    first = {"sampled_frame_count": prefix, "frame": frames[prefix - 1]["frame"], "accepted": prefix_accepted}
                    break
            rows.append(
                {
                    "target_id": row["target_id"],
                    "question_id": row["question_id"],
                    "video": row["video"],
                    "question": row["question"],
                    "aliases": row["aliases"],
                    "legacy": {"state": legacy_state, "accepted_target_ids": legacy_accepted, "expected_match": legacy_matches[row["target_id"]]},
                    "background_idf": {
                        "state": idf_state,
                        "accepted_target_ids": idf_accepted,
                        "expected_match": idf_matches[row["target_id"]],
                        "non_query_matches": {key: value for key, value in idf_matches.items() if key != row["target_id"] and value["matched"]},
                        "first_acceptance": first,
                    },
                }
            )
        salt = protocol["synthetic_negative_contract"]["salt"]
        canary_accepts = []
        for row in roster:
            digest = hashlib.sha256(f"{row['question_id']}|{row['video']}|{salt}".encode("utf-8")).hexdigest()[:10]
            token = "zz" + "".join(chr(ord("a") + int(char, 16)) for char in digest)
            match = match_idf(base.read_frames(archive, row["video"]), [token], {token: 1}, background_df, background_documents, protocol["token_contract"])
            if match["matched"]:
                canary_accepts.append({"question_id": row["question_id"], "video": row["video"], "token": token, "match": match})
    def metric(counter: Counter[str]) -> dict[str, int]:
        return {key.lower(): int(counter[key]) for key in ("CORRECT", "WRONG", "UNKNOWN", "AMBIGUOUS")}
    legacy = metric(counts["legacy"])
    idf = metric(counts["background_idf"])
    accepted_prefixes = [row["background_idf"]["first_acceptance"]["sampled_frame_count"] for row in rows if row["background_idf"]["first_acceptance"]]
    gate = {
        "thirty_fresh_ocr_unseen_episodes": len(rows) == 30,
        "minimum_three_correct_gain": idf["correct"] - legacy["correct"] >= 3,
        "zero_successor_wrong": idf["wrong"] == 0,
        "zero_successor_ambiguous": idf["ambiguous"] == 0,
        "zero_synthetic_negative_accepts": len(canary_accepts) == 0,
        "zero_identity_or_portal_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "FRESH_OCR_UNSEEN_ROADTEXTVQA_VALIDATION_SPLIT_DEVELOPMENT_SEARCH_PRIORITY_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": base.sha256(protocol_path),
        "evaluator_sha256": base.sha256(Path(__file__).resolve()),
        "metrics": {
            "episodes": len(rows),
            "sampled_frames_per_episode": 10,
            "legacy": legacy,
            "background_idf": idf,
            "correct_gain": idf["correct"] - legacy["correct"],
            "unknown_reduction": legacy["unknown"] - idf["unknown"],
            "synthetic_negative_queries": len(rows),
            "synthetic_negative_accepts": len(canary_accepts),
            "accepted_episode_mean_frames_to_first_candidate": round(sum(accepted_prefixes) / len(accepted_prefixes), 3) if accepted_prefixes else None,
            "identity_bindings_emitted": 0,
            "portal_bindings_emitted": 0,
        },
        "gate": gate,
        "synthetic_negative_accepts": canary_accepts,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"decision": result["decision"], "metrics": result["metrics"], "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
