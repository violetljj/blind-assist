from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import EXCLUSION_SCHEMA, PROTOCOL_ID, read_json, sha256_file


ALLOWED_PROFILES = {"center_obstacle", "lateral_pedestrian_or_ebike", "step_curb"}


def select_unseen_candidates(discovery: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    if registry.get("schema_version") != EXCLUSION_SCHEMA or registry.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("exclusion registry schema/protocol mismatch")
    excluded = registry.get("excluded_source_sessions")
    if not isinstance(excluded, list) or not all(isinstance(value, str) and value for value in excluded):
        raise ValueError("exclusion registry has invalid sessions")
    records = discovery.get("candidates")
    if not isinstance(records, list):
        raise ValueError("discovery is missing candidates")
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("discovery candidate must be object")
        session, profile = record.get("session_id"), record.get("selection_profile")
        if not isinstance(session, str) or not session:
            raise ValueError("discovery candidate has no session_id")
        if profile not in ALLOWED_PROFILES:
            raise ValueError(f"discovery candidate has unsupported profile {profile!r}")
        if session in excluded:
            rejected.append({"session_id": session, "selection_profile": str(profile), "reason": "excluded_source_session"})
            continue
        # Keep the source-mask evidence intact.  This selection has no event truth,
        # and it cannot decide positive/negative bucket membership.
        selected.append(record)
    profile_counts = Counter(str(record["selection_profile"]) for record in selected)
    profile_session_counts = {
        profile: len({str(record["session_id"]) for record in selected if record["selection_profile"] == profile})
        for profile in sorted(profile_counts)
    }
    sessions = sorted({str(record["session_id"]) for record in selected})
    return {
        "schema_version": "blindassist.eval_validity_r0.unseen_discovery_selection.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH",
        "selection_rule": "Keep only source sessions absent from the frozen EVAL-VALIDITY exclusion registry. Do not infer bucket, reminder, clearance or normal-walkable truth from sparse geometry.",
        "eligible_candidates": selected,
        "eligible_candidate_count": len(selected),
        "eligible_source_sessions": sessions,
        "eligible_source_session_count": len(sessions),
        "eligible_profile_counts": dict(sorted(profile_counts.items())),
        "eligible_profile_session_counts": profile_session_counts,
        "rejected_candidates": rejected,
        "next_required_gate": "Materialize continuous RGB/source-mask windows, create two isolated causal RGB review packets, and freeze event facts before any arm trace access.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True, action="append", help="Repeat for disjoint sparse-discovery batches.")
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    combined = {"candidates": []}
    for discovery_path in args.discovery:
        combined["candidates"].extend(read_json(discovery_path).get("candidates", []))
    result = select_unseen_candidates(combined, read_json(args.exclusion_registry))
    result["input_sha256"] = {
        "discovery": {str(path).replace("\\", "/"): sha256_file(path) for path in args.discovery},
        "exclusion_registry": sha256_file(args.exclusion_registry),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"eligible_candidates={result['eligible_candidate_count']} sessions={result['eligible_source_session_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
