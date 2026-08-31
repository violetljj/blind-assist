#!/usr/bin/env python3
"""One-shot RoadTextVQA replay for roster-conditioned lexical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist-l10-roadtextvqa-roster-information-router-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-roadtextvqa-roster-information-router-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def tokens(value: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z0-9]+", folded)


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def resolve_input(repo: Path, spec: dict[str, Any]) -> Path:
    path = (repo / spec["path"]).resolve() if not Path(spec["path"]).is_absolute() else Path(spec["path"]).resolve()
    require(path.is_file(), f"MISSING_INPUT:{path}")
    require(path.stat().st_size == int(spec["bytes"]), f"INPUT_SIZE_MISMATCH:{path}")
    require(sha256(path) == spec["sha256"], f"INPUT_HASH_MISMATCH:{path}")
    return path


def annotation_index(path: Path) -> dict[int, dict[str, Any]]:
    payload = load(path)
    require(payload.get("dataset_name") == "RoadTextVQA", "UNEXPECTED_DATASET")
    return {int(row["questionId"]): row for row in payload["data"]}


def frame_number(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[1])


def read_frames(archive: zipfile.ZipFile, video: str) -> list[dict[str, Any]]:
    stem = Path(video).stem
    names = [name for name in archive.namelist() if re.search(rf"/{re.escape(stem)}_[0-9]+\.json$", name)]
    require(len(names) == 10, f"EXPECTED_TEN_SAMPLED_FRAMES:{video}:{len(names)}")
    frames = []
    for name in sorted(names, key=frame_number):
        payload = json.loads(archive.read(name))
        annotations = payload.get("textAnnotations") or []
        observed = []
        for row in annotations[1:]:
            observed.extend(tokens(str(row.get("description", ""))))
        frames.append({"frame": frame_number(name), "tokens": observed})
    return frames


def target_tokens(aliases: list[str], ignored: set[str]) -> list[str]:
    return sorted({token for alias in aliases for token in tokens(alias) if token not in ignored})


def match_target(
    frames: list[dict[str, Any]],
    candidate_tokens: list[str],
    document_frequency: dict[str, int],
    contract: dict[str, Any],
    information_branch: bool,
) -> dict[str, Any]:
    long_min = int(contract["legacy_minimum_target_token_length"])
    long_two = int(contract["legacy_long_token_length_for_two_units"])
    max_edit = int(contract["legacy_maximum_edit_distance"])
    short_min = int(contract["information_short_exact_minimum_length"])
    short_max = int(contract["information_short_exact_maximum_length"])
    witnesses: dict[str, list[dict[str, Any]]] = {}
    evidence: dict[str, int] = {}
    routes: dict[str, str] = {}
    for target in candidate_tokens:
        hits = []
        if len(target) >= long_min:
            for frame in frames:
                best = None
                for observed in frame["tokens"]:
                    if len(observed) < long_min - max_edit:
                        continue
                    distance = edit_distance(observed, target)
                    if distance <= max_edit and (best is None or (distance, observed) < best):
                        best = (distance, observed)
                if best is not None:
                    hits.append({"frame": frame["frame"], "observed": best[1], "edit_distance": best[0]})
            if hits:
                witnesses[target] = hits
                evidence[target] = 2 if len(target) >= long_two else 1
                routes[target] = "LEGACY_EXACT_OR_ONE_EDIT_LONG_TOKEN"
        elif (
            information_branch
            and short_min <= len(target) <= short_max
            and document_frequency.get(target) == 1
            and target.isalpha()
        ):
            for frame in frames:
                if target in frame["tokens"]:
                    hits.append({"frame": frame["frame"], "observed": target, "edit_distance": 0})
            if hits:
                witnesses[target] = hits
                evidence[target] = int(contract["information_short_exact_evidence_units"])
                routes[target] = "ROSTER_UNIQUE_SHORT_EXACT_INFORMATION_TOKEN"
    total = sum(evidence.values())
    return {
        "matched": total >= int(contract["minimum_distinctive_evidence_units"]),
        "evidence_units": total,
        "matched_tokens": sorted(witnesses),
        "routes": routes,
        "witnesses": witnesses,
    }


def evaluate_prefix(
    frames: list[dict[str, Any]],
    roster: list[dict[str, Any]],
    token_bank: dict[str, list[str]],
    document_frequency: dict[str, int],
    contract: dict[str, Any],
    information_branch: bool,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    matches = {
        row["target_id"]: match_target(frames, token_bank[row["target_id"]], document_frequency, contract, information_branch)
        for row in roster
    }
    accepted = sorted(target_id for target_id, result in matches.items() if result["matched"])
    return accepted, matches


def outcome(expected: str, accepted: list[str]) -> str:
    if accepted == [expected]:
        return "CORRECT"
    if not accepted:
        return "UNKNOWN"
    if len(accepted) > 1:
        return "AMBIGUOUS"
    return "WRONG"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = load(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    repo = Path(__file__).resolve().parents[3]
    val_path = resolve_input(repo, protocol["inputs"]["validation_annotations"])
    ocr_path = resolve_input(repo, protocol["inputs"]["official_ocr_archive"])
    annotations = annotation_index(val_path)
    ignored = set(protocol["token_contract"]["ignored_name_tokens"])
    selected_ids = [int(value) for value in protocol["selected_question_ids"]]
    require(len(selected_ids) == len(set(selected_ids)) == 30, "SELECTION_NOT_THIRTY_UNIQUE_QUESTIONS")
    roster = []
    for question_id in selected_ids:
        source = annotations[question_id]
        roster.append(
            {
                "target_id": f"RTVQA-Q{question_id}",
                "question_id": question_id,
                "video": source["video"],
                "question": source["question"],
                "aliases": source["answer"],
            }
        )
    require(len({row["video"] for row in roster}) == 30, "SELECTED_VIDEOS_NOT_UNIQUE")
    token_bank = {row["target_id"]: target_tokens(row["aliases"], ignored) for row in roster}
    df = Counter(token for target_id in token_bank for token in set(token_bank[target_id]))
    rows = []
    counts = {name: Counter() for name in ("legacy", "information")}
    with zipfile.ZipFile(ocr_path) as archive:
        for row in roster:
            frames = read_frames(archive, row["video"])
            record = {"target_id": row["target_id"], "question_id": row["question_id"], "video": row["video"], "aliases": row["aliases"], "sampled_frames": [f["frame"] for f in frames]}
            for name, enabled in (("legacy", False), ("information", True)):
                accepted, matches = evaluate_prefix(frames, roster, token_bank, dict(df), protocol["token_contract"], enabled)
                state = outcome(row["target_id"], accepted)
                counts[name][state] += 1
                first = None
                for prefix in range(1, len(frames) + 1):
                    prefix_accepted, _ = evaluate_prefix(frames[:prefix], roster, token_bank, dict(df), protocol["token_contract"], enabled)
                    if prefix_accepted:
                        first = {"sampled_frame_count": prefix, "frame": frames[prefix - 1]["frame"], "accepted": prefix_accepted}
                        break
                record[name] = {
                    "state": state,
                    "accepted_target_ids": accepted,
                    "expected_match": matches[row["target_id"]],
                    "non_query_matches": {key: value for key, value in matches.items() if key != row["target_id"] and value["matched"]},
                    "first_acceptance": first,
                }
            rows.append(record)
        canaries = []
        salt = protocol["synthetic_negative_contract"]["salt"]
        for row in roster:
            digest = hashlib.sha256(f"{row['question_id']}|{row['video']}|{salt}".encode("utf-8")).hexdigest()[:10]
            token = "zz" + "".join(chr(ord("a") + int(char, 16)) for char in digest)
            canaries.append({"question_id": row["question_id"], "video": row["video"], "token": token})
        canary_accepts = []
        for canary in canaries:
            frames = read_frames(archive, canary["video"])
            match = match_target(frames, [canary["token"]], {canary["token"]: 1}, protocol["token_contract"], True)
            if match["matched"]:
                canary_accepts.append({**canary, "match": match})
    def metric(counter: Counter[str]) -> dict[str, int]:
        return {key.lower(): int(counter[key]) for key in ("CORRECT", "WRONG", "UNKNOWN", "AMBIGUOUS")}
    legacy = metric(counts["legacy"])
    information = metric(counts["information"])
    gate = {
        "thirty_fresh_ocr_unseen_episodes": len(rows) == 30,
        "minimum_three_correct_gain": information["correct"] - legacy["correct"] >= 3,
        "zero_successor_wrong": information["wrong"] == 0,
        "zero_successor_ambiguous": information["ambiguous"] == 0,
        "zero_synthetic_negative_accepts": len(canary_accepts) == 0,
        "zero_identity_or_portal_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "FRESH_OCR_UNSEEN_ROADTEXTVQA_VALIDATION_SPLIT_DEVELOPMENT_SEARCH_PRIORITY_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "metrics": {
            "episodes": len(rows),
            "sampled_frames_per_episode": 10,
            "legacy": legacy,
            "roster_information": information,
            "correct_gain": information["correct"] - legacy["correct"],
            "unknown_reduction": legacy["unknown"] - information["unknown"],
            "synthetic_negative_queries": len(canaries),
            "synthetic_negative_accepts": len(canary_accepts),
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
