#!/usr/bin/env python3
"""Audit source-session and ancestry role isolation for the D7 intake."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from pipeline import ContractError, load_jsonl, sha256_file, stable_id, utc_now, write_json


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"JSONL object required: {path}:{line_number}")
            yield row


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    session_path = root / "manifests" / "session_registry.jsonl"
    candidate_path = root / "candidates" / "candidate_index.jsonl"
    if not session_path.is_file() or not candidate_path.is_file():
        raise ContractError("D7 session/candidate registries are required")
    sessions = load_jsonl(session_path)
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_ancestry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        source_session_id = str(row.get("source_session_id", ""))
        ancestry_group = str(row.get("ancestry_group", ""))
        if not source_session_id or not ancestry_group:
            raise ContractError("session registry row missing source_session_id or ancestry_group")
        by_session[source_session_id].append(row)
        by_ancestry[ancestry_group].append(row)
    role_conflicts = []
    session_dataset_conflicts = []
    for session_id, rows in sorted(by_session.items()):
        roles = sorted({str(row.get("data_role")) for row in rows})
        datasets = sorted({str(row.get("dataset_id")) for row in rows})
        if len(roles) > 1:
            role_conflicts.append({"source_session_id": session_id, "roles": roles})
        if len(datasets) > 1:
            session_dataset_conflicts.append({"source_session_id": session_id, "datasets": datasets})
    ancestry_role_conflicts = []
    for ancestry, rows in sorted(by_ancestry.items()):
        roles = sorted({str(row.get("data_role")) for row in rows})
        sessions_in_group = sorted({str(row.get("source_session_id")) for row in rows})
        if len(roles) > 1:
            ancestry_role_conflicts.append({"ancestry_group": ancestry, "roles": roles, "source_session_ids": sessions_in_group})
    unknown_candidates = []
    candidate_count = 0
    for row in _iter_jsonl(candidate_path):
        candidate_count += 1
        if str(row.get("source_session_id", "")) not in by_session:
            unknown_candidates.append(str(row.get("candidate_id")))
    status = "PASS_WITHOUT_SPLIT_ASSIGNMENT" if not role_conflicts and not session_dataset_conflicts and not unknown_candidates and not ancestry_role_conflicts else "HOLD_ROLE_REVIEW"
    report = [
        "# HFTF D7 role isolation report",
        "",
        f"Status: `{status}`.",
        "",
        f"- Session registry rows: `{len(sessions)}`.",
        f"- Unique source sessions: `{len(by_session)}`.",
        f"- Unique ancestry groups: `{len(by_ancestry)}`.",
        f"- Candidate rows checked: `{candidate_count}`.",
        f"- Source-session role conflicts: `{len(role_conflicts)}`.",
        f"- Session dataset conflicts: `{len(session_dataset_conflicts)}`.",
        f"- Unknown candidate sessions: `{len(unknown_candidates)}`.",
        f"- Ancestry groups crossing data roles: `{len(ancestry_role_conflicts)}`.",
        "",
        "No split is authorized by this audit. It only checks registry identity/role consistency; it does not establish event truth or near-duplicate absence.",
    ]
    if role_conflicts:
        report.append("\n## Role conflicts\n")
        report.extend(f"- `{item['source_session_id']}`: {', '.join(item['roles'])}" for item in role_conflicts[:50])
    if ancestry_role_conflicts:
        report.append("\n## Ancestry conflicts\n")
        report.extend(f"- `{item['ancestry_group']}` crosses roles {', '.join(item['roles'])}" for item in ancestry_role_conflicts[:50])
    (root / "reports" / "role_isolation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    receipt = {
        "schema": "hftf_d7_public_real_role_isolation_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "status": status,
        "counts": {
            "session_rows": len(sessions),
            "source_sessions": len(by_session),
            "ancestry_groups": len(by_ancestry),
            "candidate_rows": candidate_count,
            "role_conflicts": len(role_conflicts),
            "session_dataset_conflicts": len(session_dataset_conflicts),
            "unknown_candidate_sessions": len(unknown_candidates),
            "ancestry_role_conflicts": len(ancestry_role_conflicts),
        },
        "session_registry_sha256": sha256_file(session_path),
        "candidate_index_sha256": sha256_file(candidate_path),
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
        "notes": [
            "Source session is the split isolation unit.",
            "Ancestry-group consistency is a registry audit only; it does not prove true physical ancestry.",
            "Near-duplicate image/time graph remains pending event review.",
        ],
    }
    write_json(root / "manifests" / "role_isolation_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
