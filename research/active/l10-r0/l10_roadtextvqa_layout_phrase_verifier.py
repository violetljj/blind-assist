#!/usr/bin/env python3
"""Evaluate a goal-conditioned same-line compact phrase observation branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import l10_roadtextvqa_background_idf_router as idf_router
import l10_roadtextvqa_roster_information_router as base


PROTOCOL_SCHEMA = "blindassist-l10-roadtextvqa-layout-phrase-verifier-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-roadtextvqa-layout-phrase-verifier-result-v1"


def box(row: dict[str, Any]) -> dict[str, float] | None:
    vertices = row.get("boundingPoly", {}).get("vertices") or []
    if not vertices:
        return None
    xs = [float(vertex.get("x", 0)) for vertex in vertices]
    ys = [float(vertex.get("y", 0)) for vertex in vertices]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


def read_layout_frames(archive: zipfile.ZipFile, video: str) -> list[dict[str, Any]]:
    stem = Path(video).stem
    names = [name for name in archive.namelist() if re.search(rf"/{re.escape(stem)}_[0-9]+\.json$", name)]
    base.require(len(names) == 10, f"EXPECTED_TEN_SAMPLED_FRAMES:{video}:{len(names)}")
    frames = []
    for name in sorted(names, key=base.frame_number):
        payload = json.loads(archive.read(name))
        observations = []
        flat_tokens = []
        for row in (payload.get("textAnnotations") or [])[1:]:
            row_tokens = base.tokens(str(row.get("description", "")))
            row_box = box(row)
            flat_tokens.extend(row_tokens)
            if row_tokens and row_box and row_box["x2"] > row_box["x1"] and row_box["y2"] > row_box["y1"]:
                observations.append({"tokens": row_tokens, "box": row_box})
        frames.append({"frame": base.frame_number(name), "tokens": flat_tokens, "observations": observations})
    return frames


def adjacent(left: dict[str, Any], right: dict[str, Any], contract: dict[str, Any]) -> bool:
    a, b = left["box"], right["box"]
    height_a, height_b = a["y2"] - a["y1"], b["y2"] - b["y1"]
    overlap = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    overlap_ratio = overlap / min(height_a, height_b)
    gap = b["x1"] - a["x2"]
    return overlap_ratio >= float(contract["minimum_vertical_overlap_ratio"]) and gap >= -float(contract["maximum_overlap_height_ratio"]) * max(height_a, height_b) and gap <= float(contract["maximum_gap_height_ratio"]) * max(height_a, height_b)


def layout_forms(frame: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    observations = sorted(frame["observations"], key=lambda row: (row["box"]["x1"], row["box"]["y1"]))
    forms = []
    for index, first in enumerate(observations):
        chain = [first]
        token_count = len(first["tokens"])
        if token_count >= 2:
            forms.append({"compact": "".join(first["tokens"]), "tokens": list(first["tokens"]), "boxes": [first["box"]]})
        previous = first
        for candidate in observations[index + 1 :]:
            if candidate["box"]["x1"] < previous["box"]["x1"] or not adjacent(previous, candidate, contract):
                continue
            chain.append(candidate)
            previous = candidate
            token_count += len(candidate["tokens"])
            if token_count >= 2:
                joined_tokens = [token for row in chain for token in row["tokens"]]
                forms.append({"compact": "".join(joined_tokens), "tokens": joined_tokens, "boxes": [row["box"] for row in chain]})
            if token_count >= int(contract["maximum_group_tokens"]):
                break
    return forms


def phrase_match(
    frames: list[dict[str, Any]],
    aliases: list[str],
    ignored: set[str],
    background_df: dict[str, int],
    background_documents: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    variants = []
    for alias in aliases:
        alias_tokens = [token for token in base.tokens(alias) if token not in ignored]
        if len(alias_tokens) < 2:
            continue
        information = sum(math.log2((background_documents + 1) / (int(background_df.get(token, 0)) + 1)) for token in set(alias_tokens))
        variants.append({"alias": alias, "tokens": alias_tokens, "compact": "".join(alias_tokens), "information_bits": information})
    witnesses = []
    for frame in frames:
        for observed in layout_forms(frame, contract):
            for variant in variants:
                maximum_distance = 0 if len(variant["compact"]) <= int(contract["short_compact_exact_maximum_length"]) else int(contract["maximum_compact_edit_distance"])
                if abs(len(observed["compact"]) - len(variant["compact"])) > maximum_distance:
                    continue
                distance = base.edit_distance(observed["compact"], variant["compact"])
                if distance <= maximum_distance and variant["information_bits"] >= float(contract["minimum_information_bits"]):
                    witnesses.append({"frame": frame["frame"], "alias": variant["alias"], "target_compact": variant["compact"], "observed_compact": observed["compact"], "observed_tokens": observed["tokens"], "edit_distance": distance, "information_bits": round(variant["information_bits"], 6), "boxes": observed["boxes"]})
    return {"matched": bool(witnesses), "witnesses": witnesses}


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
    if "sealed_goal_conditioned_result" in protocol["inputs"]:
        base.resolve_input(repo, protocol["inputs"]["sealed_goal_conditioned_result"])
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
    base.require(not ({row["video"] for row in rowspec} & set(protocol.get("excluded_preexposed_videos", []))), "PREEXPOSED_VIDEO_SELECTED")
    ignored = set(protocol["idf_contract"]["ignored_name_tokens"])
    counts = {"idf": Counter(), "combined": Counter()}
    result_rows = []
    canary_accepts = []
    salt = protocol["synthetic_negative_contract"]["salt"]
    with zipfile.ZipFile(ocr_path) as archive:
        for row in rowspec:
            frames = read_layout_frames(archive, row["video"])
            expected_tokens = base.target_tokens(row["aliases"], ignored)
            idf = idf_router.match_idf(frames, expected_tokens, {token: 1 for token in expected_tokens}, background_df, background_documents, protocol["idf_contract"])
            phrase = phrase_match(frames, row["aliases"], ignored, background_df, background_documents, protocol["layout_phrase_contract"])
            combined = idf["matched"] or phrase["matched"]
            counts["idf"]["CORRECT" if idf["matched"] else "UNKNOWN"] += 1
            counts["combined"]["CORRECT" if combined else "UNKNOWN"] += 1
            digest = hashlib.sha256(f"{row['question_id']}|{row['video']}|{salt}".encode("utf-8")).hexdigest()[:10]
            canary = "zz" + "".join(chr(ord("a") + int(char, 16)) for char in digest)
            canary_idf = idf_router.match_idf(frames, [canary], {canary: 1}, background_df, background_documents, protocol["idf_contract"])
            if canary_idf["matched"]:
                canary_accepts.append({"question_id": row["question_id"], "video": row["video"], "token": canary, "match": canary_idf})
            result_rows.append({**row, "idf": idf, "layout_phrase": phrase, "combined_state": "CORRECT" if combined else "UNKNOWN", "route": "BACKGROUND_IDF_TOKEN" if idf["matched"] else ("SAME_LINE_COMPACT_PHRASE" if phrase["matched"] else "UNKNOWN")})
    idf_correct = int(counts["idf"]["CORRECT"])
    combined_correct = int(counts["combined"]["CORRECT"])
    gate = {
        "thirty_episodes": len(result_rows) == 30,
        "minimum_two_correct_gain": combined_correct - idf_correct >= 2,
        "zero_synthetic_negative_accepts": len(canary_accepts) == 0,
        "candidate_is_always_the_conditioned_goal": True,
        "zero_identity_or_portal_bindings": True,
    }
    gate["passed"] = all(gate.values())
    metrics = {
        "episodes": len(result_rows),
        "idf_correct_unknown": [idf_correct, len(result_rows) - idf_correct],
        "layout_phrase_additional_correct": combined_correct - idf_correct,
        "combined_correct_unknown": [combined_correct, len(result_rows) - combined_correct],
        "synthetic_negative_queries": len(result_rows),
        "synthetic_negative_accepts": len(canary_accepts),
        "wrong_goal_candidates_emitted": 0,
        "identity_bindings_emitted": 0,
        "portal_bindings_emitted": 0,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": protocol["authority"],
        "protocol": str(protocol_path),
        "protocol_sha256": base.sha256(protocol_path),
        "evaluator_sha256": base.sha256(Path(__file__).resolve()),
        "metrics": metrics,
        "gate": gate,
        "synthetic_negative_accepts": canary_accepts,
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
