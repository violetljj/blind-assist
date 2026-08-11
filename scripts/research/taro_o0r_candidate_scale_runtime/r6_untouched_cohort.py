#!/usr/bin/env python3
"""Deterministically plan a metadata-only untouched ARKitScenes R6 cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r6_factor_split as r6
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


SCHEMA = "blindassist.taro.o0r.r6_untouched_cohort_candidate.v1"
STATUS = "METADATA_ONLY_COHORT_FROZEN_DATA_USE_AUTHORITY_REQUIRED"
EXCLUSION_COMMIT = "0a871e9e8d6758ad6120b2d685b427bb24f90cbc"
SELECTION_SALT = "TARO_O0R_R6_UNTOUCHED_CONFIRMATION_V1"
ROLE = "UNTOUCHED_CONFIRMATION"
PARENT_COUNT = 8
UPSAMPLING_METADATA = "artifacts.local/downloads/ARKitScenes-7283761/depth_upsampling/upsampling_train_val_splits.csv"
RAW_METADATA = "artifacts.local/downloads/ARKitScenes-7283761/raw/raw_train_val_splits.csv"
THREEDOD_METADATA = "artifacts.local/downloads/ARKitScenes-7283761/threedod/3dod_train_val_splits.csv"
METADATA_BINDINGS = {
    UPSAMPLING_METADATA: (59280, "17935C5567F3004EA01BE394D6DC9EEC4ED96F0A7E097884C8B648578CCA2F6B"),
    RAW_METADATA: (132940, "F93CD6A1EC0AEA5E103313F3BB4660744B011A1EAA8AA44A992E2C7C2966B145"),
    THREEDOD_METADATA: (132310, "B753C50B830076A8396A352C60A49D060EC00D7A91290147BC9DC374697519CE"),
}
PATHSPECS = [
    ":(glob)docs/**/*.md",
    ":(glob)docs/**/*.json",
    ":(glob)scripts/**/*.md",
    ":(glob)scripts/**/*.json",
    "DATASET_MASTER_LEDGER.csv",
]
ID_PATTERN = re.compile(r"(?<!\d)\d{6,8}(?!\d)")
EXPECTED_ROSTER = [
    ("467175", "47333514", "006B82E342177EE9D2A2234F3E71CF76E9488DD21EF4B86E510352CDA8BE3CBD"),
    ("467312", "45261569", "008C884AF94F1A353E89DDA56C0DBF7095C387256F5E73E02201BDEEAB95AF8E"),
    ("435329", "42899445", "0096CA95F08E6D36D466B850DC1DBBF564D980DB2B1C234BA2A8251F5DB1FC2B"),
    ("423306", "42897745", "0099E6BC109C77093C3524E13AD8D97E0E28FA0B717B43CE818FDBB8CFA4341E"),
    ("466652", "45261100", "009EF1A8B00F53905E2C0A8E09F1B1D2E4C55A6611DF8D3F65C1E3793FEED05C"),
    ("469650", "47333562", "00C54929175ACCDACA0484CAE8991AC4930EB49A9B8E09195E81AF1B76104C37"),
    ("470439", "47115427", "00D05F4BC6B8C3C073BBE8B0DDE511BA8567477BF244D6A73C1CBBACE2F2E896"),
    ("469830", "47334055", "00D2ACB7CB1F86314BB532052E2B9DA873DCFB37F7CE9B3DC86BE0B338FD4BD7"),
]


class R6CohortError(RuntimeError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise R6CohortError(code)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _rank(visit_id: str, video_id: str) -> str:
    value = f"{SELECTION_SALT}:{ROLE}:{visit_id}:{video_id}"
    return hashlib.sha256(value.encode("ascii")).hexdigest().upper()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(bool(rows) and set(rows[0]) == {"video_id", "visit_id", "fold"}, "R6_COHORT_METADATA_SCHEMA")
    return rows


def _excluded_ids(repo_root: Path, known_ids: set[str]) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "grep", "-h", "-o", "-P", r"(?<![0-9])[0-9]{6,8}(?![0-9])", EXCLUSION_COMMIT, "--", *PATHSPECS],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return sorted({token for token in result.stdout.splitlines() if token in known_ids})


def _select(rows: Sequence[dict[str, str]], excluded: set[str]) -> list[dict[str, str]]:
    eligible = [
        {
            "visit_id": row["visit_id"],
            "video_id": row["video_id"],
            "official_fold": "Training",
            "selection_rank_sha256": _rank(row["visit_id"], row["video_id"]),
        }
        for row in rows
        if row["fold"] == "Training"
        and row["visit_id"] != "NA"
        and row["visit_id"] not in excluded
        and row["video_id"] not in excluded
    ]
    used_visits: set[str] = set()
    selected = []
    for row in sorted(eligible, key=lambda value: value["selection_rank_sha256"]):
        if row["visit_id"] in used_visits:
            continue
        selected.append(row)
        used_visits.add(row["visit_id"])
        if len(selected) == PARENT_COUNT:
            break
    _require(len(selected) == PARENT_COUNT, "R6_COHORT_INSUFFICIENT_UNIQUE_VISITS")
    return selected


def build_plan(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    for relative, (size, digest) in METADATA_BINDINGS.items():
        path = repo_root / relative
        _require(path.is_file() and path.stat().st_size == size and _sha256(path) == digest, f"R6_COHORT_METADATA_BINDING:{relative}")
    upsampling = _load_rows(repo_root / UPSAMPLING_METADATA)
    raw = _load_rows(repo_root / RAW_METADATA)
    threedod = _load_rows(repo_root / THREEDOD_METADATA)
    _require(len(upsampling) == 2257, "R6_COHORT_UPSAMPLING_ROW_COUNT")
    known_ids = {value for row in upsampling for value in (row["visit_id"], row["video_id"]) if value != "NA"}
    excluded = _excluded_ids(repo_root, known_ids)
    excluded_set = set(excluded)
    selected = _select(upsampling, excluded_set)
    raw_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in raw}
    threedod_keys = {(row["visit_id"], row["video_id"], row["fold"]) for row in threedod}
    for row in selected:
        key = (row["visit_id"], row["video_id"], row["official_fold"])
        _require(key in raw_keys and key in threedod_keys, f"R6_COHORT_CROSS_SPLIT_MISSING:{row['video_id']}")
        _require(row["visit_id"] not in r6.FORBIDDEN_FORMATION_PARENTS, f"R6_COHORT_FORMATION_PARENT_OVERLAP:{row['visit_id']}")
    exclusion_digest = hashlib.sha256(("\n".join(excluded) + "\n").encode("utf-8")).hexdigest().upper()
    required_assets = [
        {"asset": "upsampling.zip", "url_template": "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/upsampling/Training/{video_id}.zip"},
        {"asset": "lowres_wide_intrinsics.zip", "url_template": "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{video_id}/lowres_wide_intrinsics.zip"},
        {"asset": "lowres_wide.traj", "url_template": "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{video_id}/lowres_wide.traj"},
    ]
    expanded_requests = [
        {
            "visit_id": row["visit_id"],
            "video_id": row["video_id"],
            "asset": asset["asset"],
            "url": asset["url_template"].format(video_id=row["video_id"]),
        }
        for row in selected
        for asset in required_assets
    ]
    plan = {
        "schema": SCHEMA,
        "status": STATUS,
        "protocol_lock_sha256": r6.PROTOCOL_LOCK_SHA256,
        "implementation_lock_path": "docs/research/taro/TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK_2026-08-11.json",
        "source": {
            "repository": "https://github.com/apple/ARKitScenes",
            "repository_commit": "7283761bf26c27570ec59a5dc0f8686fbff07726",
            "official_fold": "Training",
            "metadata_bindings": [
                {"path": relative, "bytes": size, "sha256": digest}
                for relative, (size, digest) in METADATA_BINDINGS.items()
            ],
        },
        "selection": {
            "exclusion_snapshot_commit": EXCLUSION_COMMIT,
            "exclusion_scan_scope": PATHSPECS,
            "matched_official_identity_count": len(excluded),
            "matched_official_identities_sha256": exclusion_digest,
            "selection_salt": SELECTION_SALT,
            "selection_rule": "Rank SHA256('{salt}:{role}:{visit_id}:{video_id}') over official Training rows absent from the frozen repository exclusion snapshot; retain the first video per unique visit.",
            "eligible_row_count": sum(
                row["fold"] == "Training" and row["visit_id"] != "NA" and row["visit_id"] not in excluded_set and row["video_id"] not in excluded_set
                for row in upsampling
            ),
            "role": ROLE,
            "roster": selected,
        },
        "required_assets": required_assets,
        "request_plan": {
            "method": "HEAD",
            "response_body_bytes_allowed": 0,
            "request_count": len(expanded_requests),
            "expanded_requests_sha256": adapter.canonical_sha256(expanded_requests),
            "requests": expanded_requests,
        },
        "invariants": {
            "parent_count": len(selected),
            "video_count": len({row["video_id"] for row in selected}),
            "unique_visit_count": len({row["visit_id"] for row in selected}),
            "present_once_in_upsampling_raw_and_threedod_splits": True,
            "formation_parent_overlap": 0,
            "media_body_bytes_read": False,
            "model_outputs_read": False,
            "truth_payload_read": False,
        },
        "authority": {
            "metadata_selection": True,
            "head_requests": False,
            "source_download": False,
            "source_decode": False,
            "model_execution": False,
            "truth_scoring": False,
            "training": False,
        },
        "unique_successor": "EXPLICIT_R6_EXACT_COHORT_DATA_USE_AUTHORIZATION_AND_HEAD_LOCK",
    }
    return validate_plan(plan, repo_root=repo_root, recompute=False)


def validate_plan(value: Mapping[str, Any], *, repo_root: Path, recompute: bool = True) -> dict[str, Any]:
    plan = json.loads(json.dumps(dict(value)))
    _require(plan.get("schema") == SCHEMA and plan.get("status") == STATUS, "R6_COHORT_PLAN_IDENTITY")
    _require(plan.get("protocol_lock_sha256") == r6.PROTOCOL_LOCK_SHA256, "R6_COHORT_PROTOCOL_DRIFT")
    selection = plan.get("selection", {})
    roster = selection.get("roster")
    _require(isinstance(roster, list) and len(roster) == PARENT_COUNT, "R6_COHORT_ROSTER_COUNT")
    observed = [(row.get("visit_id"), row.get("video_id"), row.get("selection_rank_sha256")) for row in roster]
    _require(observed == EXPECTED_ROSTER, "R6_COHORT_ROSTER_DRIFT")
    _require(len({row[0] for row in observed}) == len({row[1] for row in observed}) == PARENT_COUNT, "R6_COHORT_ROSTER_DUPLICATE")
    _require(not ({row[0] for row in observed} & r6.FORBIDDEN_FORMATION_PARENTS), "R6_COHORT_FORMATION_PARENT_OVERLAP")
    authority = plan.get("authority", {})
    _require(authority == {"metadata_selection": True, "head_requests": False, "source_download": False, "source_decode": False, "model_execution": False, "truth_scoring": False, "training": False}, "R6_COHORT_AUTHORITY_DRIFT")
    requests = plan.get("request_plan", {})
    _require(requests.get("method") == "HEAD" and requests.get("response_body_bytes_allowed") == 0 and requests.get("request_count") == 24, "R6_COHORT_REQUEST_PLAN_DRIFT")
    _require(isinstance(requests.get("requests"), list) and adapter.canonical_sha256(requests["requests"]) == requests.get("expanded_requests_sha256"), "R6_COHORT_REQUEST_HASH_DRIFT")
    _require(plan.get("unique_successor") == "EXPLICIT_R6_EXACT_COHORT_DATA_USE_AUTHORIZATION_AND_HEAD_LOCK", "R6_COHORT_SUCCESSOR_DRIFT")
    if recompute:
        _require(plan == build_plan(repo_root), "R6_COHORT_RECOMPUTE_DRIFT")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    print(json.dumps(build_plan(args.repo_root), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
