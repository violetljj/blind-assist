#!/usr/bin/env python3
"""Post-hoc progressive early exit over the consumed two-target fresh panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import l10_panolab_distinctive_edit_token_posthoc as successor


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-progressive-evidence-early-exit-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-progressive-evidence-early-exit-posthoc-result-v1"


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


def verify(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file(), f"MISSING_FROZEN_INPUT:{path}")
    require(sha256(path) == spec["sha256"], f"HASH_MISMATCH:{path}")
    require(path.stat().st_size == int(spec["bytes"]), f"BYTE_COUNT_MISMATCH:{path}")
    return path


def appearance_candidate(
    scores: dict[str, Any],
    target_ids: list[str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    ranked = sorted(
        ((target_id, float(scores[target_id]["score"])) for target_id in target_ids),
        key=lambda row: (-row[1], row[0]),
    )
    top_id, top_score = ranked[0]
    margin = top_score - ranked[1][1]
    accepted = (
        top_score >= float(thresholds["minimum_top1_score"])
        and margin >= float(thresholds["minimum_top1_margin"])
    )
    return {
        "prediction": top_id,
        "top1_score": top_score,
        "top1_margin": margin,
        "candidate": top_id if accepted else None,
        "score_gate": top_score >= float(thresholds["minimum_top1_score"]),
        "margin_gate": margin >= float(thresholds["minimum_top1_margin"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = load(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    lexical_protocol = load(verify(protocol["inputs"]["fresh_lexical_protocol"]))
    lexical_result = load(verify(protocol["inputs"]["fresh_lexical_result"]))
    combined_protocol = load(verify(protocol["inputs"]["fresh_combined_protocol"]))
    combined_result = load(verify(protocol["inputs"]["fresh_combined_result"]))
    require(lexical_result["metrics"]["lexical_correct_wrong_no_match_ambiguous"] == [1, 0, 1, 0], "UNEXPECTED_LEXICAL_BASELINE")
    require(combined_result["metrics"]["combined_correct_wrong_unknown"] == [2, 0, 0], "UNEXPECTED_COMBINED_BASELINE")
    require(protocol["thresholds"] == combined_protocol["accept_contract"], "APPEARANCE_THRESHOLDS_CHANGED")
    require(protocol["target_roster"] == lexical_protocol["target_roster"], "TARGET_ROSTER_CHANGED")

    targets = protocol["target_roster"]
    target_ids = [row["episode_id"] for row in targets]
    lexical_rows = {row["episode_id"]: row for row in lexical_result["rows"]}
    combined_rows = {row["episode_id"]: row for row in combined_result["rows"]}
    rows = []
    correct = wrong = unknown = 0
    lexical_routes = appearance_routes = 0
    observed_frames = 0
    wrong_target_candidates = 0
    for episode_id in protocol["evaluated_episodes"]:
        lexical_row = lexical_rows[episode_id]
        appearance_frames = {
            row["relation"]: row["scores"]
            for row in (combined_rows[episode_id]["appearance_fallback"] or {}).get("per_frame", [])
        }
        trace = []
        candidate = None
        route = "UNKNOWN_KEEP_SEARCHING"
        for prefix_length, frame in enumerate(lexical_row["frames"], start=1):
            prefix = lexical_row["frames"][:prefix_length]
            matches = {
                target["episode_id"]: successor.match_target(
                    prefix,
                    target["entity_name"],
                    lexical_protocol["name_token_contract"],
                    lexical_protocol["distinctive_edit_token_contract"],
                )
                for target in targets
            }
            matched_targets = [target_id for target_id in target_ids if matches[target_id]["matched"]]
            relation = frame["frame_key"].rsplit("_", 1)[-1]
            step: dict[str, Any] = {
                "prefix_length": prefix_length,
                "latest_relation": relation,
                "lexical_matched_targets": matched_targets,
            }
            if len(matched_targets) == 1:
                candidate = matched_targets[0]
                route = "PROGRESSIVE_DISTINCTIVE_TOKEN_EARLY_EXIT"
                lexical_routes += 1
                step["decision"] = route
                trace.append(step)
                break
            if len(matched_targets) > 1:
                route = "UNKNOWN_LEXICAL_CONFLICT"
                step["decision"] = route
                trace.append(step)
                break
            if relation in appearance_frames:
                appearance = appearance_candidate(
                    appearance_frames[relation], target_ids, protocol["thresholds"]
                )
                step["appearance"] = appearance
                if appearance["candidate"] is not None:
                    candidate = appearance["candidate"]
                    route = "PROGRESSIVE_APPEARANCE_EARLY_EXIT"
                    appearance_routes += 1
                    step["decision"] = route
                    trace.append(step)
                    break
            step["decision"] = "CONTINUE_OBSERVING"
            trace.append(step)
        consumed = len(trace)
        observed_frames += consumed
        if candidate == episode_id:
            correct += 1
        elif candidate is None:
            unknown += 1
        else:
            wrong += 1
            wrong_target_candidates += 1
        rows.append(
            {
                "episode_id": episode_id,
                "target_name": lexical_row["target_name"],
                "trace": trace,
                "frames_available": len(lexical_row["frames"]),
                "frames_consumed_before_exit": consumed,
                "saved_frames": len(lexical_row["frames"]) - consumed,
                "candidate": candidate,
                "route": route,
                "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                "portal_ownership_binding": None,
            }
        )

    full_frames = sum(len(row["frames"]) for row in lexical_rows.values())
    metrics = {
        "consumed_posthoc_episodes": len(rows),
        "fixed_router_frames_available": full_frames,
        "progressive_frames_consumed": observed_frames,
        "frames_saved": full_frames - observed_frames,
        "observation_reduction_rate": (full_frames - observed_frames) / full_frames,
        "correct_wrong_unknown": [correct, wrong, unknown],
        "lexical_early_exits": lexical_routes,
        "appearance_early_exits": appearance_routes,
        "wrong_target_candidates": wrong_target_candidates,
        "new_ocr_or_appearance_model_calls": 0,
        "portal_ownership_bindings_emitted": 0,
    }
    gate = {
        "two_consumed_sequences_replayed": len(rows) == 2,
        "two_of_two_correct": correct == 2,
        "zero_wrong": wrong == 0,
        "zero_unknown": unknown == 0,
        "minimum_fifty_percent_observation_reduction": metrics["observation_reduction_rate"] >= 0.5,
        "one_lexical_and_one_appearance_early_exit": lexical_routes == appearance_routes == 1,
        "zero_new_model_calls": True,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "CONSUMED_POSTHOC_PROGRESSIVE_OBSERVATION_MECHANISM_ONLY_REQUIRES_FRESH_CONFIRMATION",
        "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "evaluator_sha256": sha256(Path(__file__).resolve()), "metrics": metrics,
        "gate": gate, "rows": rows, "claim_boundary": protocol["claim_boundary"],
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
