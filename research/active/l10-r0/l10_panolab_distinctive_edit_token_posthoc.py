#!/usr/bin/env python3
"""Post-hoc replay of a conservative distinctive-token lexical branch.

This consumes already sealed OCR rows. It is a mechanism probe, not fresh
confirmation: thresholds were chosen from the observed PS01/PS02 failures and
must not be retuned on this panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-distinctive-edit-token-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-distinctive-edit-token-posthoc-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"MISSING_FROZEN_INPUT:{path}")
    require(sha256(path) == spec["sha256"], f"HASH_MISMATCH:{path}")
    require(path.stat().st_size == int(spec["bytes"]), f"BYTE_COUNT_MISMATCH:{path}")
    return path


def ascii_tokens(value: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z0-9]+", folded)


def significant_name_tokens(value: str, contract: dict[str, Any]) -> list[str]:
    ignored = set(contract["ignored_name_tokens"])
    return [token for token in ascii_tokens(value) if token not in ignored]


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def row_forms(row: dict[str, Any]) -> list[str]:
    tokens = [str(token) for token in row["ascii_tokens"] if token]
    forms = list(tokens)
    if 2 <= len(tokens) <= 3:
        joined = "".join(tokens)
        if joined not in forms:
            forms.append(joined)
    return forms


def match_target(
    frames: list[dict[str, Any]],
    entity_name: str,
    lexical_contract: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    target_tokens = significant_name_tokens(entity_name, lexical_contract)
    minimum_score = float(contract["minimum_ocr_row_score"])
    minimum_token_length = int(contract["minimum_target_token_length"])
    long_token_length = int(contract["long_token_length_for_two_units"])
    maximum_distance = int(contract["maximum_edit_distance"])
    witnesses: dict[str, list[dict[str, Any]]] = {token: [] for token in target_tokens}
    for frame in frames:
        for row in frame["ocr_rows"]:
            if float(row["score"]) < minimum_score:
                continue
            for target_token in target_tokens:
                if len(target_token) < minimum_token_length:
                    continue
                best: tuple[int, str] | None = None
                for observed in row_forms(row):
                    if len(observed) < minimum_token_length - maximum_distance:
                        continue
                    distance = edit_distance(observed, target_token)
                    if distance <= maximum_distance and (best is None or (distance, observed) < best):
                        best = (distance, observed)
                if best is not None:
                    witnesses[target_token].append(
                        {
                            "frame_key": frame["frame_key"],
                            "panorama_item_id": frame["panorama_item_id"],
                            "ocr_text": row["text"],
                            "ocr_score": row["score"],
                            "observed_form": best[1],
                            "edit_distance": best[0],
                            "match_kind": "EXACT" if best[0] == 0 else "EDIT_NEIGHBOR",
                            "box_xyxy": row["box_xyxy"],
                        }
                    )
    observed = [token for token in target_tokens if witnesses[token]]
    evidence_units = sum(2 if len(token) >= long_token_length else 1 for token in observed)
    matched = evidence_units >= int(contract["minimum_distinctive_evidence_units"])
    return {
        "matched": matched,
        "tier": "DISTINCTIVE_EXACT_OR_ONE_EDIT_TOKEN_BANK" if matched else "NONE",
        "target_tokens": target_tokens,
        "observed_target_tokens": observed,
        "distinctive_evidence_units": evidence_units,
        "required_distinctive_evidence_units": int(contract["minimum_distinctive_evidence_units"]),
        "witnesses": {token: witnesses[token] for token in observed},
        "portal_ownership_authority": "NONE",
    }


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

    prior_protocol = load(verify(protocol["frozen_inputs"]["failed_router_protocol"]))
    prior_result = load(verify(protocol["frozen_inputs"]["failed_router_result"]))
    appearance_result = load(verify(protocol["frozen_inputs"]["appearance_result"]))
    require(
        prior_result["decision"]
        == "L10_PANOLAB_PRODUCER_STRATIFIED_PIXEL_UNSEEN_LEXICAL_APPEARANCE_ROUTER_DEVELOPMENT_GATE_NOT_MET",
        "EXPECTED_FROZEN_FAILURE_NOT_PRESENT",
    )
    require(prior_result["metrics"]["combined_correct_wrong_unknown"] == [1, 0, 2], "UNEXPECTED_BASELINE")
    require(prior_protocol["evaluated_episodes"] == protocol["evaluated_episodes"], "EPISODE_ORDER_MISMATCH")
    require(prior_protocol["target_roster"] == protocol["target_roster"], "TARGET_ROSTER_MISMATCH")

    target_ids = [row["episode_id"] for row in protocol["target_roster"]]
    target_index = {row["episode_id"]: row for row in protocol["target_roster"]}
    prior_rows = {row["episode_id"]: row for row in prior_result["rows"]}
    appearance_rows = {row["episode_id"]: row for row in appearance_result["rows"]}
    result_rows = []
    lexical_correct = lexical_wrong = lexical_no_match = lexical_ambiguous = 0
    combined_correct = combined_wrong = combined_unknown = 0
    wrong_target_matches = 0
    for episode_id in protocol["evaluated_episodes"]:
        frozen_row = prior_rows[episode_id]
        frames = frozen_row["frames"]
        matches = {
            target_id: match_target(
                frames,
                target_index[target_id]["entity_name"],
                protocol["name_token_contract"],
                protocol["distinctive_edit_token_contract"],
            )
            for target_id in target_ids
        }
        matched_targets = [target_id for target_id in target_ids if matches[target_id]["matched"]]
        wrong_matches = [target_id for target_id in matched_targets if target_id != episode_id]
        wrong_target_matches += len(wrong_matches)
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

        appearance_candidate = appearance_rows[episode_id]["acceptance"]["candidate"]
        if lexical_state == "UNIQUE_OWN_TARGET_MATCH":
            candidate = lexical_candidate
            route = "POSTHOC_DISTINCTIVE_EDIT_TOKEN_SEARCH_PRIORITY_CANDIDATE"
        elif lexical_state == "NO_MATCH":
            candidate = appearance_candidate
            route = "TEMPORAL_APPEARANCE_SEARCH_PRIORITY_CANDIDATE" if candidate else "UNKNOWN_KEEP_SEARCHING"
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
                "target_name": target_index[episode_id]["entity_name"],
                "frozen_ocr_rows_reused": sum(len(frame["ocr_rows"]) for frame in frames),
                "target_matches": matches,
                "lexical": {
                    "state": lexical_state,
                    "matched_targets": matched_targets,
                    "candidate": lexical_candidate,
                    "wrong_target_matches": wrong_matches,
                },
                "appearance_fallback": {"candidate": appearance_candidate},
                "combined_router": {
                    "route": route,
                    "candidate": candidate,
                    "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                    "portal_ownership_binding": None,
                },
            }
        )

    appearance_correct = sum(
        appearance_rows[episode_id]["acceptance"]["candidate"] == episode_id
        for episode_id in protocol["evaluated_episodes"]
    )
    metrics = {
        "consumed_posthoc_episodes": len(result_rows),
        "lexical_correct_wrong_no_match_ambiguous": [
            lexical_correct,
            lexical_wrong,
            lexical_no_match,
            lexical_ambiguous,
        ],
        "appearance_fallback_correct_wrong_unknown": [
            appearance_correct,
            0,
            len(result_rows) - appearance_correct,
        ],
        "combined_correct_wrong_unknown": [combined_correct, combined_wrong, combined_unknown],
        "combined_correct_gain_over_appearance": combined_correct - appearance_correct,
        "wrong_target_lexical_trials": len(result_rows) * (len(target_ids) - 1),
        "wrong_target_lexical_matches": wrong_target_matches,
        "new_ocr_or_model_calls": 0,
        "portal_ownership_bindings_emitted": 0,
    }
    gate = {
        "three_consumed_sequences_replayed": len(result_rows) == 3,
        "three_of_three_combined_correct": combined_correct == 3,
        "minimum_two_correct_gain_over_appearance": combined_correct - appearance_correct >= 2,
        "zero_combined_wrong": combined_wrong == 0,
        "zero_combined_unknown": combined_unknown == 0,
        "zero_wrong_target_lexical_matches": wrong_target_matches == 0,
        "zero_new_ocr_or_model_calls": True,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "CONSUMED_POSTHOC_DEVELOPMENT_MECHANISM_ONLY_REQUIRES_FRESH_PIXEL_AND_OCR_UNSEEN_CONFIRMATION",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "metrics": metrics,
        "gate": gate,
        "rows": result_rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
