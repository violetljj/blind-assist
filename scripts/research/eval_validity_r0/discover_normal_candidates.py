from __future__ import annotations

"""Source-mask-only discovery of strict normal-walkable review candidates.

This is deliberately separate from the older hazard-oriented discovery script.
It may authorize RGB review of a session but never creates a negative event fact.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .common import EXCLUSION_SCHEMA, PROTOCOL_ID, read_json

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from discover_sanpo_sequence_candidates import (  # noqa: E402
    DEFAULT_LENS,
    frame_number,
    mask_geometry,
    media_url,
    select_mask_view,
    session_ids,
    sparse_profile_evidence,
)


def normal_walkable_candidate(profile: dict[str, Any]) -> bool:
    """A strict source-mask shortlist, intentionally not an event/safety label."""
    return bool(
        profile.get("path_geometry_usable")
        and not profile.get("has_center_hazard")
        and not profile.get("step_curb")
        and not profile.get("center_obstacle")
        and not profile.get("center_lateral_target")
    )


def longest_run(values: list[bool]) -> int:
    current = best = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-session-index", type=int, default=0, help="Index after exclusion filtering.")
    parser.add_argument("--max-sessions", type=int, default=0, help="0 scans every remaining train session.")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--minimum-normal-samples", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.start_session_index < 0 or args.sample_count <= 0 or args.minimum_normal_samples <= 0 or args.minimum_normal_samples > args.sample_count or args.retries <= 0:
        raise SystemExit("invalid scan bounds")
    registry = read_json(args.exclusion_registry)
    if registry.get("schema_version") != EXCLUSION_SCHEMA or registry.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit("exclusion registry schema/protocol mismatch")
    excluded = set(registry.get("excluded_source_sessions", []))
    all_train = session_ids("train")
    eligible = [session for session in all_train if session not in excluded]
    sessions = eligible[args.start_session_index:]
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for ordinal, session_id in enumerate(sessions, start=1):
        try:
            selection = select_mask_view(session_id, "auto", args.sample_count)
            if selection is None:
                continue
            camera, objects = selection
            numbers = sorted(frame_number(item["name"]) for item in objects)
            sampled = [numbers[round(index * (len(numbers) - 1) / (args.sample_count - 1))] for index in range(args.sample_count)]
            by_number = {frame_number(item["name"]): item for item in objects}
            evidence: list[dict[str, Any]] = []
            for source_frame in sampled:
                item = by_number[source_frame]
                components, path = mask_geometry(media_url(item["name"], item.get("generation")), args.retries)
                evidence.append({"source_frame": source_frame, "profiles": sparse_profile_evidence(components, path)})
            normal = [normal_walkable_candidate(item["profiles"]) for item in evidence]
            if sum(normal) >= args.minimum_normal_samples and longest_run(normal) >= 2:
                matching = [item["source_frame"] for item, passed in zip(evidence, normal) if passed]
                candidates.append({
                    "session_id": session_id,
                    "official_split": "train",
                    "camera": camera,
                    "lens": DEFAULT_LENS,
                    "selection_profile": "strict_normal_walkable_source_mask_only",
                    "sampled_source_frames": sampled,
                    "geometry_matching_source_frames": matching,
                    "sparse_longest_consecutive_sample_run": longest_run(normal),
                    "sparse_frame_evidence": evidence,
                    "recommended_start_frame": max(0, matching[len(matching) // 2] - 15),
                    "next_gate": "download a continuous window and obtain two isolated causal RGB reviews; this is not normal-walkable event truth",
                })
            print(f"scanned=train-unseen:{ordinal}/{len(sessions)} records={len(candidates)}", flush=True)
        except Exception as error:  # keep source/network failures auditable
            failures.append({"session_id": session_id, "error": f"{type(error).__name__}: {error}"})
    payload = {
        "schema_version": "blindassist.eval_validity_r0.normal_source_mask_discovery.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH",
        "source": "SANPO-Real v0 public GCS source masks",
        "excluded_source_session_count": len(excluded),
        "available_official_train_session_count": len(all_train),
        "eligible_session_count_before_slice": len(eligible),
        "start_session_index_after_exclusion": args.start_session_index,
        "attempted_session_count": len(sessions),
        "selection_rule": "usable corridor path and no source-mask center hazard, center-lateral target, center obstacle or broad step/curb candidate in sufficient sparse samples",
        "important_limit": "A pass is source-mask-only shortlist evidence. It does not prove normal passage, no-alert, no-actionability or user safety.",
        "candidates": candidates,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"candidates={len(candidates)} failures={len(failures)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
