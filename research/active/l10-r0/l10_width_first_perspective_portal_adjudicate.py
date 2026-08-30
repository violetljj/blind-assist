#!/usr/bin/env python3
"""Join disjoint role audits for the frozen width-first perspective cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SCHEMA = "blindassist-l10-width-first-perspective-portal-source-v1"
MANIFEST_SCHEMA = "blindassist-l10-width-first-perspective-portal-materialized-v1"
AUDIT_SCHEMA = "blindassist-l10-width-first-perspective-portal-role-audit-v1"
RESULT_SCHEMA = "blindassist-l10-width-first-perspective-portal-source-admission-result-v1"
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


def role_index(source: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for episode in source["episodes"]:
        episode_id = str(episode["episode_id"])
        roles = {str(row["role"]): row for row in episode.get("roles") or []}
        require(set(roles) == {"REFERENCE", "QUERY"}, f"SOURCE_ROLE_SET_INVALID:{episode_id}")
        index[episode_id] = roles
    return index


def review_index(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest.get("roles") or []:
        key = (str(row["episode_id"]), str(row["role"]))
        require(key not in index, f"MANIFEST_ROLE_DUPLICATE:{key[0]}:{key[1]}")
        index[key] = row
    require(len(index) == 6, "MANIFEST_ROLE_COUNT_NOT_6")
    return index


def validate_audit(
    audit: dict[str, Any],
    source_path: Path,
    manifest_path: Path,
    source_roles: dict[str, dict[str, dict[str, Any]]],
    manifest_roles: dict[tuple[str, str], dict[str, Any]],
    expected_role: str,
) -> dict[str, dict[str, Any]]:
    require(audit.get("schema") == AUDIT_SCHEMA, f"AUDIT_SCHEMA_MISMATCH:{expected_role}")
    require(audit.get("role") == expected_role, f"ROLE_MISMATCH:{expected_role}")
    require(audit.get("source_sha256") == sha256(source_path), f"SOURCE_HASH_MISMATCH:{expected_role}")
    require(
        audit.get("materialized_manifest_sha256") == sha256(manifest_path),
        f"MANIFEST_HASH_MISMATCH:{expected_role}",
    )
    rows = audit.get("episodes") or []
    require(len(rows) == 3, f"AUDIT_COUNT_NOT_3:{expected_role}")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        require(episode_id in source_roles and episode_id not in index, f"EPISODE_MISMATCH:{expected_role}:{episode_id}")
        expected_source = source_roles[episode_id][expected_role]
        expected_manifest = manifest_roles[(episode_id, expected_role)]
        require(str(row["item_id"]) == str(expected_source["item_id"]), f"SOURCE_ITEM_MISMATCH:{expected_role}:{episode_id}")
        require(str(row["item_id"]) == str(expected_manifest["item_id"]), f"MANIFEST_ITEM_MISMATCH:{expected_role}:{episode_id}")
        review_image = expected_manifest["review_image"]
        require(row.get("review_image_sha256") == review_image["sha256"], f"REVIEW_HASH_MISMATCH:{expected_role}:{episode_id}")
        require(row.get("review_image_dimensions") == review_image["dimensions"], f"REVIEW_DIMENSIONS_MISMATCH:{expected_role}:{episode_id}")
        status = str(row.get("status"))
        require(status in ALLOWED, f"STATUS_INVALID:{expected_role}:{episode_id}")
        box = row.get("portal_box_xyxy")
        if status == "ADMITTED":
            require(isinstance(box, list) and len(box) == 4, f"BOX_MISSING:{expected_role}:{episode_id}")
            values = [float(value) for value in box]
            require(0 <= values[0] < values[2] <= 1800, f"BOX_X_INVALID:{expected_role}:{episode_id}")
            require(0 <= values[1] < values[3] <= 1350, f"BOX_Y_INVALID:{expected_role}:{episode_id}")
        else:
            require(box is None, f"NON_ADMITTED_BOX_PRESENT:{expected_role}:{episode_id}")
        index[episode_id] = row
    require(set(index) == set(source_roles), f"AUDIT_EPISODE_SET_MISMATCH:{expected_role}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--materialized-manifest", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--query-audit", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source.resolve()
    manifest_path = args.materialized_manifest.resolve()
    result_path = args.result.resolve()
    require(not result_path.exists(), f"RESULT_ALREADY_EXISTS:{result_path}")

    source = load_json(source_path)
    require(source.get("schema") == SOURCE_SCHEMA, "SOURCE_SCHEMA_MISMATCH")
    require(len(source.get("episodes") or []) == 3, "SOURCE_EPISODE_COUNT_NOT_3")
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == MANIFEST_SCHEMA, "MANIFEST_SCHEMA_MISMATCH")
    require(manifest.get("source_sha256") == sha256(source_path), "MANIFEST_SOURCE_HASH_MISMATCH")
    source_roles = role_index(source)
    manifest_roles = review_index(manifest)

    reference_path = args.reference_audit.resolve()
    query_path = args.query_audit.resolve()
    reference = validate_audit(
        load_json(reference_path), source_path, manifest_path, source_roles, manifest_roles, "REFERENCE"
    )
    query = validate_audit(
        load_json(query_path), source_path, manifest_path, source_roles, manifest_roles, "QUERY"
    )

    joined = []
    joint_ids = []
    for episode in source["episodes"]:
        episode_id = str(episode["episode_id"])
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
    if joint_count == 3:
        decision = "L10_WIDTH_FIRST_PERSPECTIVE_PORTAL_SOURCE_ADMITTED_JOINT_3_OF_3"
        stop_reason = None
        next_action = "Run the unchanged portal-transfer matcher once on all three frozen episodes."
        formal_ids = joint_ids
    else:
        decision = (
            "L10_WIDTH_FIRST_PERSPECTIVE_PORTAL_SOURCE_NOT_EVALUABLE_"
            f"REFERENCE_{reference_count}_OF_3_QUERY_{query_count}_OF_3_JOINT_{joint_count}_OF_3_NO_MATCHER_CALL"
        )
        stop_reason = "DECLARED_CAMERA_ACCURACY_BANDS_DO_NOT_UNIQUIFY_PORTALS_NO_MATCHER_CALL"
        next_action = (
            "Change the pose or registration source to sub-metre camera-to-portal geometry, or supply a directly posed "
            "portal mask or mesh; do not tune matcher, threshold, crop, pose uncertainty or the consumed cohort."
        )
        formal_ids = []

    result = {
        "schema": RESULT_SCHEMA,
        "decision": decision,
        "stop_reason": stop_reason,
        "source": "research/active/l10-r0/" + source_path.name,
        "source_sha256": sha256(source_path),
        "materialized_manifest": "knowledge/l10-width-panoramax-perspective-v1/materialized/manifest.json",
        "materialized_manifest_sha256": sha256(manifest_path),
        "reference_audit": "research/active/l10-r0/" + reference_path.name,
        "reference_audit_sha256": sha256(reference_path),
        "query_audit": "research/active/l10-r0/" + query_path.name,
        "query_audit_sha256": sha256(query_path),
        "metrics": {
            "episode_count": 3,
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
    print(json.dumps({"decision": decision, **result["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
