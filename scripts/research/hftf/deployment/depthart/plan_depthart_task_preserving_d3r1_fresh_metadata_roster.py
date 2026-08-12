#!/usr/bin/env python3
"""Plan the fresh metadata-only recovery pool for DepthART D3R1."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable

from scripts.research.hftf.deployment.depthart.plan_depthart_task_preserving_d3_fresh_metadata_roster import (
    SOURCE_COMMIT,
    _git,
    _official_ids,
    metadata_blob,
)


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d3r1_recovery_protocol_v1"
ACTIVATION_SCHEMA = "blindassist_depthart_task_preserving_d3r1_metadata_activation_v1"
ROSTER_SCHEMA = "blindassist_depthart_task_preserving_d3r1_fresh_metadata_roster_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_RECOVERY"
ROLE = "D3R1_METADATA_CANDIDATE_POOL_ONLY"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, "exclusive write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def repository_root() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ValueError("repository root not found from planner path")


def verify_frozen_file(entry: dict[str, Any], label: str) -> Path:
    path = Path(entry["path"])
    if not path.is_absolute():
        path = repository_root() / path
    require(path.is_file(), f"{label} missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"{label} bytes drift: {path}")
    require(sha256_file(path) == entry["sha256"], f"{label} SHA drift: {path}")
    return path


def load_python_pool_identity_ids(
    path: Path,
    constant_name: str,
    expected_parent_count: int,
    expected_ordered_tuple_sha256: str,
) -> set[str]:
    """Read a literal pool constant without importing or executing its module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    literals: list[Any] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        names = (
            [target.id for target in node.targets if isinstance(target, ast.Name)]
            if isinstance(node, ast.Assign)
            else [node.target.id] if isinstance(node.target, ast.Name) else []
        )
        if constant_name in names:
            literals.append(ast.literal_eval(node.value))
    require(
        len(literals) == 1,
        f"expected exactly one literal constant {constant_name}: {path}",
    )
    literal = literals[0]
    require(
        isinstance(literal, (list, tuple))
        and len(literal) == expected_parent_count,
        f"{constant_name} parent count drift",
    )
    identities: set[str] = set()
    visits: set[str] = set()
    sessions: set[str] = set()
    digest_lines: list[str] = []
    for row in literal:
        require(
            isinstance(row, (list, tuple))
            and len(row) == 3
            and isinstance(row[0], str)
            and isinstance(row[1], str)
            and isinstance(row[2], str),
            f"malformed {constant_name} row",
        )
        visit_id, video_id, rank = row
        require(visit_id.isdigit() and video_id.isdigit(), f"non-numeric {constant_name} identity")
        require(
            len(rank) == 64 and all(character in "0123456789ABCDEF" for character in rank),
            f"malformed {constant_name} rank",
        )
        require(visit_id not in visits, f"duplicate {constant_name} visit")
        require(video_id not in sessions, f"duplicate {constant_name} session")
        visits.add(visit_id)
        sessions.add(video_id)
        identities.update((visit_id, video_id))
        digest_lines.append(f"{visit_id}:{video_id}:{rank}")
    require(
        len(identities) == expected_parent_count * 2,
        f"{constant_name} identity count drift",
    )
    digest = hashlib.sha256(("\n".join(digest_lines) + "\n").encode("ascii")).hexdigest().upper()
    require(digest == expected_ordered_tuple_sha256, f"{constant_name} ordered digest drift")
    return identities


def scan_commit_exclusions(
    repo: Path, commit: str, known_ids: set[str]
) -> dict[str, Any]:
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            commit,
            "docs/research",
        ],
        check=True,
        capture_output=True,
    ).stdout
    receipts: list[dict[str, Any]] = []
    excluded: set[str] = set()
    scanned_file_count = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tree:
        members = sorted(
            (
                member
                for member in tree.getmembers()
                if member.isfile()
                and Path(member.name).suffix.lower() in {".json", ".md"}
            ),
            key=lambda member: member.name,
        )
        scanned_file_count = len(members)
        for member in members:
            stream = tree.extractfile(member)
            require(stream is not None, f"cannot read Git archive member: {member.name}")
            data = stream.read()
            matches = _official_ids(data, known_ids)
            if not matches:
                continue
            excluded.update(matches)
            receipts.append(
                {
                    "path": member.name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest().upper(),
                    "matched_official_identities": matches,
                }
            )
    manifest = json.dumps(receipts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "scope": f"Git tree {commit} docs/research JSON and Markdown",
        "commit": commit,
        "scanned_file_count": scanned_file_count,
        "matched_file_count": len(receipts),
        "matched_official_identity_count": len(excluded),
        "matched_official_identities": sorted(excluded),
        "file_receipts": receipts,
        "file_receipts_sha256": hashlib.sha256(manifest).hexdigest().upper(),
    }


def binomial_tail_probability(trials: int, probability: float, successes: int) -> float:
    require(0 <= probability <= 1, "probability must be in [0, 1]")
    require(0 <= successes <= trials, "success threshold outside trial range")
    return math.fsum(
        math.comb(trials, value)
        * probability**value
        * (1.0 - probability) ** (trials - value)
        for value in range(successes, trials + 1)
    )


def validate_bindings(
    protocol_path: Path, activation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    activation = load_json(activation_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id drift")
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    require(
        activation["bindings"]["protocol"]["sha256"] == sha256_file(protocol_path),
        "activation protocol SHA drift",
    )
    for entry, label in (
        (protocol["planner"], "planner"),
        (protocol["validator"], "validator"),
        (protocol["planner_test"], "planner test"),
        (protocol["validator_test"], "validator test"),
        (protocol["owning_d3_protocol"], "owning D3 protocol"),
        (protocol["predecessor"], "D3 predecessor result"),
        (protocol["predecessor_roster"], "D3 predecessor roster"),
    ):
        verify_frozen_file(entry, label)
    for index, entry in enumerate(protocol["dependencies"]):
        verify_frozen_file(entry, f"dependency {index}")
    for name, entry in protocol["concurrent_identity_firewalls"].items():
        verify_frozen_file(entry, f"concurrent identity firewall {name}")
    require(protocol["planner"]["sha256"] == sha256_file(Path(__file__)), "planner self SHA drift")
    require(
        protocol["source"]["repository_commit"] == SOURCE_COMMIT
        and protocol["source"]["metadata_path"]
        == "threedod/3dod_train_val_splits.csv",
        "source binding drift",
    )
    require(
        activation["bindings"]["predecessor"]["sha256"]
        == protocol["predecessor"]["sha256"],
        "activation predecessor drift",
    )
    require(
        activation["authority"]["metadata_roster"] is True
        and activation["authority"]["media_head"] is False
        and activation["authority"]["media_body"] is False
        and activation["authority"]["truth_or_model"] is False
        and activation["authority"]["phase_a_selection"] is False,
        "activation authority drift",
    )
    sizing = protocol["pool_sizing"]
    pool_count = int(protocol["pool_count"])
    target = int(protocol["future_phase_a_target"])
    lower_bound = float(sizing["one_sided_95_clopper_pearson_lower_bound"])
    tail = binomial_tail_probability(pool_count, lower_bound, target)
    previous = binomial_tail_probability(pool_count - 1, lower_bound, target)
    require(abs(tail - float(sizing["planned_tail_probability"])) <= 1e-12, "pool tail drift")
    require(tail >= 0.95 and previous < 0.95, "pool count is not the frozen minimal sizing")
    return protocol, activation


def select(
    rows: Iterable[dict[str, str]], excluded: set[str], pool_count: int
) -> list[dict[str, Any]]:
    require(pool_count > 0, "pool_count must be positive")
    eligible = [
        row
        for row in rows
        if row["fold"] == "Training"
        and row["visit_id"] != "NA"
        and row["visit_id"] not in excluded
        and row["video_id"] not in excluded
    ]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{row['visit_id']}:{row['video_id']}".encode("ascii")
        ).hexdigest()
    )
    selected: list[dict[str, Any]] = []
    visits: set[str] = set()
    sessions: set[str] = set()
    for row in eligible:
        if row["visit_id"] in visits or row["video_id"] in sessions:
            continue
        selection_digest = hashlib.sha256(
            f"{row['visit_id']}:{row['video_id']}".encode("ascii")
        ).hexdigest().upper()
        selected.append(
            {
                "pool_order": len(selected) + 1,
                "visit_id": row["visit_id"],
                "video_id": row["video_id"],
                "fold": row["fold"],
                "role": ROLE,
                "selection_sha256": selection_digest,
            }
        )
        visits.add(row["visit_id"])
        sessions.add(row["video_id"])
        if len(selected) == pool_count:
            break
    require(
        len(selected) == pool_count,
        f"only {len(selected)} unique eligible visits for requested {pool_count}",
    )
    return selected


def plan(
    metadata: Path,
    repo: Path,
    protocol_path: Path,
    activation_path: Path,
) -> dict[str, Any]:
    protocol, _ = validate_bindings(protocol_path, activation_path)
    pool_count = int(protocol["pool_count"])
    blob = metadata_blob(metadata)
    require(
        len(blob) == int(protocol["source"]["metadata_bytes"])
        and hashlib.sha256(blob).hexdigest().upper()
        == protocol["source"]["metadata_sha256"],
        "source metadata blob drift",
    )
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8"))))
    require(
        bool(rows) and set(rows[0]) == {"video_id", "visit_id", "fold"},
        "unexpected ARKitScenes split CSV schema",
    )
    require(
        len(rows) == int(protocol["source"]["official_rows"]),
        "source metadata row count drift",
    )
    known_ids = {
        value
        for row in rows
        for value in (row["video_id"], row["visit_id"])
        if value != "NA"
    }
    exclusion_commit = protocol["exclusion_snapshot"]["commit"]
    snapshot = scan_commit_exclusions(repo, exclusion_commit, known_ids)
    require(
        snapshot["matched_official_identity_count"]
        == int(protocol["exclusion_snapshot"]["expected_official_identity_count"])
        and snapshot["matched_file_count"]
        == int(protocol["exclusion_snapshot"]["expected_matched_file_count"])
        and snapshot["file_receipts_sha256"]
        == protocol["exclusion_snapshot"]["file_receipts_sha256"],
        "frozen research exclusion snapshot drift",
    )
    workspace_excluded = set(snapshot["matched_official_identities"])
    concurrent_firewall: dict[str, Any] = {}
    concurrent_identity_sets: dict[str, set[str]] = {}
    concurrent_excluded: set[str] = set()
    for name, entry in protocol["concurrent_identity_firewalls"].items():
        path = verify_frozen_file(entry, f"concurrent identity firewall {name}")
        identities = load_python_pool_identity_ids(
            path,
            entry["pool_constant"],
            int(entry["expected_parent_count"]),
            entry["ordered_tuple_sha256"],
        )
        require(identities <= known_ids, f"{name} contains non-official identities")
        require(
            len(identities) == int(entry["expected_official_identity_count"]),
            f"{name} official identity count drift",
        )
        concurrent_excluded.update(identities)
        concurrent_identity_sets[name] = identities
        concurrent_firewall[name] = {
            **entry,
            "official_identity_count": len(identities),
            "workspace_snapshot_overlap_count": len(identities & workspace_excluded),
        }
    effective_excluded = workspace_excluded | concurrent_excluded
    eligible_rows = [
        row
        for row in rows
        if row["fold"] == "Training"
        and row["visit_id"] != "NA"
        and row["visit_id"] not in effective_excluded
        and row["video_id"] not in effective_excluded
    ]
    capacity = {
        "training_row_count": sum(row["fold"] == "Training" for row in rows),
        "eligible_row_count": len(eligible_rows),
        "eligible_unique_visit_count": len(
            {row["visit_id"] for row in eligible_rows}
        ),
        "eligible_unique_session_count": len(
            {row["video_id"] for row in eligible_rows}
        ),
    }
    require(capacity == protocol["expected_capacity"], "eligible capacity drift")
    selected = select(rows, effective_excluded, pool_count)
    selected_ids = {
        value for row in selected for value in (row["visit_id"], row["video_id"])
    }
    for name, entry in concurrent_firewall.items():
        entry["selection_overlap_count"] = len(
            selected_ids & concurrent_identity_sets[name]
        )
        require(entry["selection_overlap_count"] == 0, f"D3R1 selection overlaps {name}")

    anchor_firewall: dict[str, Any] = {}
    predecessor_ids: set[str] = set()
    for name, entry in protocol["anchor_firewall"].items():
        path = verify_frozen_file(entry, f"anchor {name}")
        data = path.read_bytes()
        anchor_ids = set(_official_ids(data, known_ids))
        overlap = selected_ids & anchor_ids
        anchor_firewall[name] = {
            **entry,
            "official_identity_count": len(anchor_ids),
            "selection_overlap_count": len(overlap),
        }
        require(
            len(anchor_ids) == int(entry["expected_official_identity_count"]),
            f"{name} anchor identity count drift",
        )
        require(not overlap, f"D3R1 selection overlaps {name}")
        if name == "d3_predecessor_roster":
            predecessor_ids = anchor_ids
    require(len(predecessor_ids) == 96, "D3 predecessor roster identity count drift")
    require(
        predecessor_ids <= workspace_excluded,
        "D3 predecessor identities missing from exclusion snapshot",
    )

    return {
        "schema": ROSTER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "D3R1_FRESH_METADATA_POOL_127_LOCKED_MEDIA_UNOPENED",
        "bindings": {
            "protocol_sha256": sha256_file(protocol_path),
            "activation_sha256": sha256_file(activation_path),
            "planner_sha256": sha256_file(Path(__file__)),
        },
        "source": {
            "repository": "https://github.com/apple/ARKitScenes.git",
            "repository_commit": SOURCE_COMMIT,
            "metadata_path": "threedod/3dod_train_val_splits.csv",
            "metadata_bytes": len(blob),
            "metadata_sha256": hashlib.sha256(blob).hexdigest().upper(),
            "official_rows": len(rows),
        },
        "workspace_snapshot": {
            **snapshot,
        },
        "concurrent_identity_firewall": concurrent_firewall,
        "anchor_firewall": anchor_firewall,
        "selection": {
            "rule": protocol["selection_rule"],
            **capacity,
            "pool": selected,
        },
        "planning_only_sizing": protocol["pool_sizing"],
        "future_stages_not_activated": {
            "source_scope": "register exact D3R1 127-identity Phase-A intrinsics/trajectory use before any HEAD",
            "phase_a": "process all 127 and lock the first 32 label-blind portrait/pose-qualified identities",
            "phase_b": "take the first 16 source-truth-support-qualified identities",
            "role_assignment": "first 8 qualified become D3R1 TRAIN; next 8 become sealed D3R1 DEVELOPMENT",
            "frames_per_identity": 300,
        },
        "invariants": {
            "pool_count": len(selected),
            "unique_parent_count": len({row["visit_id"] for row in selected}),
            "unique_session_count": len({row["video_id"] for row in selected}),
            "workspace_excluded_identity_count": len(workspace_excluded),
            "concurrent_excluded_identity_count": len(concurrent_excluded),
            "effective_excluded_identity_count": len(effective_excluded),
            "selection_overlap_with_workspace_snapshot": len(
                selected_ids & workspace_excluded
            ),
            "selection_overlap_with_concurrent_identity_firewalls": len(
                selected_ids & concurrent_excluded
            ),
            "selection_overlap_with_d3_predecessor_pool": len(
                selected_ids & predecessor_ids
            ),
            "media_head_requests": 0,
            "media_body_bytes_read": 0,
            "truth_read": False,
            "model_outputs_read": False,
            "training": False,
            "development_outcome_access": "NONE",
            "r2_cohort_access": "NONE",
            "source_scope_registered": False,
            "download_authorized": False,
        },
        "next_gate": "EXPLICIT_D3R1_SOURCE_SCOPE_REGISTRATION_FOR_EXACT_127_METADATA_ROSTER",
        "authority": "Metadata-only D3R1 fresh pool and identity firewall; no source-use registration, media HEAD/body, truth, selection, training, Development, R2, performance, production or safety authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--activation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = plan(
        args.metadata,
        args.repo,
        args.protocol,
        args.activation,
    )
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_exclusive(args.output, encoded)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pool_count": result["invariants"]["pool_count"],
                "excluded_identity_count": result["workspace_snapshot"][
                    "matched_official_identity_count"
                ],
                "sha256": hashlib.sha256(encoded).hexdigest().upper(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
