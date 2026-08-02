"""Preflight source-mask discovery for JUDGE_AUDIT_R0.

The preflight intentionally stops at candidate discovery.  It reports whether
there are enough independent source sessions to start blind review, but never
turns a source-mask profile into an action label or event truth.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json, sha256_file
from .judge_audit import DISCOVERY_ARMS, INDEPENDENT_DISCOVERY_ARMS, SCENARIO_CATEGORIES


def _candidate_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("session_id"), str) and isinstance(node.get("selection_profile"), str):
                records.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return records


def preflight(discoveries: list[tuple[str, dict[str, Any]] | tuple[str, dict[str, Any], str]], registry: dict[str, Any], minimum_events: int = 50, maximum_events: int = 100) -> dict[str, Any]:
    excluded = set(registry.get("excluded_source_sessions", []))
    by_session: dict[str, dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    duplicate_records = 0
    discovery_arm_counts: Counter[str] = Counter()
    for discovery in discoveries:
        if len(discovery) == 2:
            source_name, value = discovery
            discovery_arm = "source_mask"
        else:
            source_name, value, discovery_arm = discovery
        if discovery_arm not in DISCOVERY_ARMS:
            raise ValueError(f"{source_name}: invalid discovery arm {discovery_arm}")
        for record in _candidate_records(value):
            session_id = record["session_id"]
            if session_id in excluded:
                continue
            source_counts[source_name] += 1
            declared_record_arm = record.get("discovery_arm")
            if declared_record_arm is not None and declared_record_arm != discovery_arm:
                raise ValueError(f"{source_name}/{session_id}: record discovery arm conflicts with input arm")
            record_arm = discovery_arm
            if record_arm not in DISCOVERY_ARMS:
                raise ValueError(f"{source_name}/{session_id}: invalid discovery arm {record_arm}")
            if session_id in by_session:
                duplicate_records += 1
                existing_arms = by_session[session_id]["discovery_arms"]
                if record_arm not in existing_arms:
                    existing_arms.append(record_arm)
                    discovery_arm_counts[record_arm] += 1
                continue
            by_session[session_id] = {
                "session_id": session_id,
                "selection_profile": record["selection_profile"],
                "source_discovery": source_name,
                "discovery_arms": [record_arm],
                "next_gate": record.get("next_gate"),
            }
            discovery_arm_counts[record_arm] += 1
    profile_counts = Counter(item["selection_profile"] for item in by_session.values())
    artifacts = {
        "event_ledger": False,
        "causal_review_a": False,
        "causal_review_b": False,
        "retrospective_review": False,
        "counterfactual_pairs": False,
        "oracle_manifest": False,
        "unified_traces": False,
    }
    coverage = {
        category: {
            "status": "NOT_ESTABLISHED",
            "count": 0,
            "reason": "source-mask discovery is not RGB blind event truth",
        }
        for category in SCENARIO_CATEGORIES
    }
    enough_sessions = minimum_events <= len(by_session) <= maximum_events
    distinct_arms = set(discovery_arm_counts)
    independent_arms = distinct_arms & set(INDEPENDENT_DISCOVERY_ARMS)
    return {
        "schema_version": "blindassist.eval_validity_r0.judge_preflight.v2",
        "protocol_id": PROTOCOL_ID,
        "status": "HOLD_JUDGE_AUDIT_COHORT",
        "source_role": "SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH",
        "candidate_session_count": len(by_session),
        "requested_event_range": [minimum_events, maximum_events],
        "enough_candidate_sessions_to_start_review": enough_sessions,
        "candidate_record_count_by_discovery": dict(sorted(source_counts.items())),
        "duplicate_candidate_records_across_discoveries": duplicate_records,
        "discovery_arm_counts_by_unique_session": dict(sorted(discovery_arm_counts.items())),
        "formal_discovery_mix": {
            "status": "ESTABLISHED" if len(distinct_arms) >= 2 and independent_arms and "source_mask" in distinct_arms else "NOT_ESTABLISHED",
            "distinct_arms": sorted(distinct_arms),
            "minimum_distinct_arms": 2,
            "independent_arms_present": sorted(independent_arms),
            "required_independent_arm_count": 1,
            "source_mask_required": True,
        },
        "profile_counts_by_unique_session": dict(sorted(profile_counts.items())),
        "coverage": coverage,
        "required_artifacts": artifacts,
        "formal_review_access": False,
        "calibration_pilot": {
            "status": "NOT_STARTED",
            "mode": "CALIBRATION_BURNED",
            "required_event_range": [8, 12],
            "required_counterfactual_pair_range": [3, 4],
            "required_reviewers": ["CAUSAL_A", "CAUSAL_B", "RETROSPECTIVE_C"],
            "formal_denominator_inclusion": False,
            "purpose": "Test metadata blinding, primitive stability, native information ceiling and kernel conversion before formal cohort access.",
        },
        "burned_assets": [
            {
                "path": "artifacts.local/evidence/eval-validity-r0/rgb-probe-wz9-v1",
                "role": "CALIBRATION_BURNED",
                "benchmark_ready": False,
                "formal_denominator_inclusion": False,
                "reason": "50-frame RGB/mask chain probe; no event truth and benchmark_ready=false",
            }
        ],
        "candidate_sessions": sorted(by_session.values(), key=lambda item: item["session_id"]),
        "next_action": "Run only the burned 8-12 event primitive calibration pilot first; keep formal review access closed until it passes, then freeze the 50-100 event/session cohort.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, action="append", required=True)
    parser.add_argument("--discovery-arm", choices=DISCOVERY_ARMS, action="append")
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    discovery_arms = args.discovery_arm or ["source_mask"] * len(args.discovery)
    if len(discovery_arms) != len(args.discovery):
        raise SystemExit("--discovery-arm must be supplied once per --discovery")
    discoveries = [(str(path), read_json(path), arm) for path, arm in zip(args.discovery, discovery_arms)]
    registry = read_json(args.exclusion_registry)
    result = preflight(discoveries, registry)
    result["input_sha256"] = {
        "discoveries": [{"path": str(path), "sha256": sha256_file(path)} for path in args.discovery],
        "exclusion_registry": sha256_file(args.exclusion_registry),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} candidate_sessions={result['candidate_session_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
