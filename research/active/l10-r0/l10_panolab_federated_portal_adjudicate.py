#!/usr/bin/env python3
"""Join disjoint reference/query portal audits for the frozen federated cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SCHEMA = "blindassist-l10-panolab-federated-portal-source-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-federated-portal-source-admission-result-v1"
ALLOWED = {"ADMITTED", "AMBIGUOUS", "NOT_VISIBLE"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit(
    audit: dict[str, Any],
    source_path: Path,
    source: dict[str, Any],
    expected_role: str,
) -> dict[str, dict[str, Any]]:
    require(audit.get("role") == expected_role, f"ROLE_MISMATCH:{expected_role}")
    require(audit.get("source_sha256") == sha256(source_path), f"SOURCE_HASH_MISMATCH:{expected_role}")
    rows = audit.get("episodes") or []
    require(len(rows) == 5, f"AUDIT_COUNT_NOT_5:{expected_role}")
    expected = {row["episode_id"]: row[expected_role.lower()]["item_id"] for row in source["episodes"]}
    index = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        require(episode_id in expected and episode_id not in index, f"EPISODE_MISMATCH:{expected_role}:{episode_id}")
        require(str(row["item_id"]) == expected[episode_id], f"ITEM_MISMATCH:{expected_role}:{episode_id}")
        status = str(row.get("status"))
        require(status in ALLOWED, f"STATUS_INVALID:{expected_role}:{episode_id}")
        box = row.get("portal_box_xyxy")
        if status == "ADMITTED":
            require(isinstance(box, list) and len(box) == 4, f"BOX_MISSING:{expected_role}:{episode_id}")
            values = [float(value) for value in box]
            require(0 <= values[0] <= 512 <= values[2] <= 1024, f"BOX_RAY_OR_X_INVALID:{expected_role}:{episode_id}")
            require(0 <= values[1] < values[3] <= 768, f"BOX_Y_INVALID:{expected_role}:{episode_id}")
        else:
            require(box is None, f"NON_ADMITTED_BOX_PRESENT:{expected_role}:{episode_id}")
        index[episode_id] = row
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--query-audit", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source.resolve()
    result_path = args.result.resolve()
    require(not result_path.exists(), f"RESULT_ALREADY_EXISTS:{result_path}")
    source = load_json(source_path)
    require(source.get("schema") == SOURCE_SCHEMA and len(source.get("episodes") or []) == 5, "SOURCE_SCHEMA_OR_COUNT_MISMATCH")
    reference_path = args.reference_audit.resolve()
    query_path = args.query_audit.resolve()
    reference = validate_audit(load_json(reference_path), source_path, source, "REFERENCE")
    query = validate_audit(load_json(query_path), source_path, source, "QUERY")
    joined = []
    joint_ids = []
    for episode in source["episodes"]:
        episode_id = episode["episode_id"]
        joint = reference[episode_id]["status"] == "ADMITTED" and query[episode_id]["status"] == "ADMITTED"
        if joint:
            joint_ids.append(episode_id)
        joined.append(
            {
                "episode_id": episode_id,
                "target_way_id": episode["target_way_id"],
                "target_name": episode["target_name"],
                "reference_status": reference[episode_id]["status"],
                "query_status": query[episode_id]["status"],
                "joint_admitted": joint,
                "reference_portal_box_xyxy": reference[episode_id].get("portal_box_xyxy"),
                "query_portal_box_xyxy": query[episode_id].get("portal_box_xyxy"),
            }
        )
    reference_count = sum(row["status"] == "ADMITTED" for row in reference.values())
    query_count = sum(row["status"] == "ADMITTED" for row in query.values())
    joint_count = len(joint_ids)
    formal_ids = joint_ids[:3]
    if joint_count >= 3:
        decision = f"L10_PANOLAB_FEDERATED_PORTAL_SOURCE_ADMITTED_JOINT_{joint_count}_OF_5"
        stop_reason = None
        next_action = "Freeze the first three joint episodes and run the unchanged exact-portal patch transfer once."
    else:
        decision = (
            "L10_PANOLAB_FEDERATED_PORTAL_SOURCE_NOT_EVALUABLE_"
            f"REFERENCE_{reference_count}_OF_5_QUERY_{query_count}_OF_5_JOINT_{joint_count}_OF_5_NO_MATCHER_CALL"
        )
        stop_reason = "FEWER_THAN_THREE_JOINTLY_VISIBLE_UNIQUE_PORTALS_NO_MATCHER_CALL"
        next_action = "Replace the observation source; do not tune matcher, crop, threshold or cohort."
    result = {
        "schema": RESULT_SCHEMA,
        "decision": decision,
        "stop_reason": stop_reason,
        "source": "research/active/l10-r0/" + source_path.name,
        "source_sha256": sha256(source_path),
        "reference_audit": "research/active/l10-r0/" + reference_path.name,
        "reference_audit_sha256": sha256(reference_path),
        "query_audit": "research/active/l10-r0/" + query_path.name,
        "query_audit_sha256": sha256(query_path),
        "metrics": {
            "episode_count": 5,
            "reference_admitted": reference_count,
            "query_admitted": query_count,
            "joint_admitted": joint_count,
            "formal_episode_count": len(formal_ids),
            "matcher_calls": 0,
        },
        "formal_episode_ids": formal_ids,
        "episodes": joined,
        "next_action": next_action,
        "claim_boundary": source["claim_boundary"],
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, **result["metrics"], "formal_episode_ids": formal_ids}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
