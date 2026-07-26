from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


CANDIDATE_ID = "BONN_METADATA_BLIND_AUTHORITY_AND_COHORT_FREEZE_R0"
DESIGN_LOCK_SHA256 = (
    "e49d1f88f13e2a190714211cfe46bb7d9f8518eaca93b5981736dcdd7231c9e9"
)
HISTORICAL_MANIFEST_SHA256 = (
    "f02bd9f1313def45cc107d72ace5f7c7803f4ab816bf6e98c5f9173fa3bb1cc6"
)
OFFICIAL_PAGE_SHA256 = (
    "2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186"
)
RECEIPT_SCHEMA_VERSION = "rcle.phase_b.bonn_metadata_gate.receipt.v1"
VALIDATION_SCHEMA_VERSION = "rcle.phase_b.bonn_metadata_gate.recompute.v1"
EXPECTED_UNIVERSE_COUNT = 26
MAXIMUM_DISPLAY_SIZE_MB = 550.0
SELECTION_SALT = "rcle-phase-b-bonn-entry-r1"
SELECTED_SEQUENCE_COUNT = 6
TERMINAL_PASS = (
    "METADATA_ONLY_PASS_FORMAL_PHASE_B_REMAINS_CLOSED_PENDING_SEPARATE_AUTHORIZATION"
)
TERMINAL_HOLD = "HOLD_NOT_EVALUABLE"
TERMINAL_CLOSE = "CLOSE_BONN_METADATA_BLIND_AUTHORITY_AND_COHORT_FREEZE_R0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def environment_manifest() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
    }


def _git(args: Sequence[str], repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def selection_hash(sequence_id: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SALT}\t{sequence_id}".encode("utf-8")
    ).hexdigest()


def parse_official_inventory(html: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"Name:\s*(rgbd_bonn_[a-z0-9_]+)<br\s*/>\s*"
        r"Size:\s*([0-9.]+)\s*(MB|GB)<br\s*/>\s*"
        r'<a href="(https://www\.ipb\.uni-bonn\.de/html/projects/'
        r'rgbd_dynamic2019/\1\.zip)">Download</a>',
        re.IGNORECASE,
    )
    inventory: list[dict[str, Any]] = []
    for sequence_id, size_text, unit, url in pattern.findall(html):
        size_mb = float(size_text) * (1024.0 if unit.upper() == "GB" else 1.0)
        inventory.append(
            {
                "sequence_id": sequence_id,
                "display_size": f"{size_text} {unit.upper()}",
                "display_size_mb_normalized": size_mb,
                "url": url,
            }
        )
    inventory.sort(key=lambda row: row["sequence_id"])
    identities = [row["sequence_id"] for row in inventory]
    if len(inventory) != EXPECTED_UNIVERSE_COUNT:
        raise ValueError(
            f"OFFICIAL_UNIVERSE_COUNT_DRIFT:{len(inventory)}"
        )
    if len(set(identities)) != EXPECTED_UNIVERSE_COUNT:
        raise ValueError("OFFICIAL_UNIVERSE_IDENTITY_DUPLICATE")
    return inventory


def load_historical_exclusions(
    manifest_path: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    if sha256_file(manifest_path) != HISTORICAL_MANIFEST_SHA256:
        raise ValueError("HISTORICAL_EXCLUSION_MANIFEST_HASH_MISMATCH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("excluded_sequences", [])
    if manifest.get("exclusion_count") != 9 or len(rows) != 9:
        raise ValueError("HISTORICAL_EXCLUSION_COUNT_MISMATCH")
    exclusions = {
        row["sequence_id"]: row["historical_role"] for row in rows
    }
    if len(exclusions) != 9:
        raise ValueError("HISTORICAL_EXCLUSION_IDENTITY_DUPLICATE")
    if manifest.get("future_formal_cohort_use") != "DENY":
        raise ValueError("HISTORICAL_EXCLUSION_DENY_MISMATCH")
    return exclusions, manifest


def build_metadata_decision(
    inventory: Iterable[dict[str, Any]],
    historical_exclusions: dict[str, str],
) -> dict[str, Any]:
    ordered = sorted(inventory, key=lambda row: row["sequence_id"])
    universe_ids = {row["sequence_id"] for row in ordered}
    missing_historical = sorted(set(historical_exclusions) - universe_ids)
    if missing_historical:
        raise ValueError(
            "HISTORICAL_IDENTITY_MISSING_FROM_OFFICIAL_UNIVERSE:"
            + ",".join(missing_historical)
        )

    eligible = [
        row
        for row in ordered
        if row["sequence_id"] not in historical_exclusions
        and row["display_size_mb_normalized"] <= MAXIMUM_DISPLAY_SIZE_MB
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            selection_hash(row["sequence_id"]),
            row["sequence_id"],
        ),
    )
    rank_by_id = {
        row["sequence_id"]: index
        for index, row in enumerate(ranked, start=1)
    }
    selected_ids = {
        row["sequence_id"] for row in ranked[:SELECTED_SEQUENCE_COUNT]
    }

    denominator: list[dict[str, Any]] = []
    for universe_rank, row in enumerate(ordered, start=1):
        sequence_id = row["sequence_id"]
        historical_role = historical_exclusions.get(sequence_id)
        over_size = (
            row["display_size_mb_normalized"] > MAXIMUM_DISPLAY_SIZE_MB
        )
        if historical_role is not None:
            disposition = "EXCLUDED_HISTORICAL"
            reason = historical_role
        elif over_size:
            disposition = "EXCLUDED_DISPLAY_SIZE"
            reason = "DISPLAY_SIZE_EXCEEDS_550_MB"
        elif sequence_id in selected_ids:
            disposition = "SELECTED"
            reason = "FIRST_6_BY_FROZEN_SELECTION_HASH"
        else:
            disposition = "ELIGIBLE_NOT_SELECTED"
            reason = "RANK_AFTER_FIRST_6"
        denominator.append(
            {
                **row,
                "universe_rank": universe_rank,
                "selection_hash": selection_hash(sequence_id),
                "eligible_rank": rank_by_id.get(sequence_id),
                "disposition": disposition,
                "reason": reason,
                "selected": sequence_id in selected_ids,
            }
        )

    if len(ranked) < SELECTED_SEQUENCE_COUNT:
        terminal = TERMINAL_CLOSE
        gate_pass = False
    else:
        terminal = TERMINAL_PASS
        gate_pass = True
    selected = sorted(
        (row for row in denominator if row["selected"]),
        key=lambda row: row["eligible_rank"],
    )
    cohort_lines = [
        "\t".join(
            [
                str(row["eligible_rank"]),
                row["sequence_id"],
                row["display_size"],
                row["url"],
                row["selection_hash"],
            ]
        )
        for row in selected
    ]
    cohort_sha256 = hashlib.sha256(
        ("\n".join(cohort_lines) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "gate_pass": gate_pass,
        "terminal_state": terminal,
        "official_universe_count": len(ordered),
        "metadata_selection_denominator_count": len(denominator),
        "historical_exclusion_count": len(historical_exclusions),
        "eligible_unseen_count": len(ranked),
        "selected_sequence_count": len(selected),
        "selected_sequence_ids": [row["sequence_id"] for row in selected],
        "cohort_identity_sha256": cohort_sha256,
        "metadata_selection_denominator": denominator,
    }


def control_manifest(
    repo_root: Path, lock: dict[str, Any] | None = None
) -> dict[str, str]:
    if lock is not None:
        paths = sorted(lock["control_source_manifest"])
    else:
        base = (
            repo_root
            / "scripts"
            / "research"
            / "egomotion_compensated_looming"
        )
        paths = [
            (
                base
                / "rcle_phase_b_bonn_entry_r0"
                / "__init__.py"
            ).relative_to(repo_root).as_posix(),
            (
                base / "rcle_phase_b_bonn_entry_r0" / "gate.py"
            ).relative_to(repo_root).as_posix(),
            (
                base / "run_phase_b_bonn_metadata_gate_r0.py"
            ).relative_to(repo_root).as_posix(),
            (
                base
                / "schemas"
                / "rcle_phase_b_bonn_metadata_gate_receipt_r0.schema.json"
            ).relative_to(repo_root).as_posix(),
            (
                base
                / "tests_phase_b_bonn_entry_r0"
                / "__init__.py"
            ).relative_to(repo_root).as_posix(),
            (
                base
                / "tests_phase_b_bonn_entry_r0"
                / "test_metadata_gate_r0.py"
            ).relative_to(repo_root).as_posix(),
        ]
    return {path: sha256_file(repo_root / path) for path in paths}


def validate_implementation_lock(
    repo_root: Path, lock_path: Path
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    design_path = (
        repo_root
        / "docs"
        / "research"
        / "rcle"
        / "RCLE_PHASE_B_BONN_ENTRY_DESIGN_LOCK_2026-07-26.json"
    )
    historical_path = (
        repo_root
        / "docs"
        / "research"
        / "rcle"
        / "RCLE_PHASE_B_BONN_HISTORICAL_EXCLUSION_MANIFEST_2026-07-26.json"
    )
    checks = {
        "candidate": lock.get("candidate_id") == CANDIDATE_ID,
        "design_lock": (
            lock.get("design_lock_sha256") == DESIGN_LOCK_SHA256
            and sha256_file(design_path) == DESIGN_LOCK_SHA256
        ),
        "historical_manifest": (
            lock.get("historical_exclusion_manifest_sha256")
            == HISTORICAL_MANIFEST_SHA256
            and sha256_file(historical_path) == HISTORICAL_MANIFEST_SHA256
        ),
        "official_page": (
            lock.get("official_page_sha256") == OFFICIAL_PAGE_SHA256
        ),
        "universe": lock.get("required_unique_sequence_count")
        == EXPECTED_UNIVERSE_COUNT,
        "selected_count": lock.get("required_selected_sequence_count")
        == SELECTED_SEQUENCE_COUNT,
        "environment": lock.get("environment") == environment_manifest(),
        "controls": lock.get("control_source_manifest")
        == control_manifest(repo_root, lock),
        "schema": lock.get("receipt_schema_version")
        == RECEIPT_SCHEMA_VERSION,
        "maximum_runs": lock.get("maximum_gate_runs") == 1,
        "payload": lock.get("payload_access_authorized") is False,
        "phase_b": lock.get("formal_phase_b_authorized") is False,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            "IMPLEMENTATION_LOCK_MISMATCH:" + ",".join(failures)
        )
    return lock


def build_receipt(
    repo_root: Path,
    official_page_path: Path,
    lock_path: Path,
    command: Sequence[str],
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    lock = validate_implementation_lock(repo_root, lock_path)
    if sha256_file(official_page_path) != OFFICIAL_PAGE_SHA256:
        raise ValueError("OFFICIAL_PAGE_HASH_MISMATCH")
    inventory = parse_official_inventory(
        official_page_path.read_text(encoding="utf-8")
    )
    historical_path = (
        repo_root
        / "docs"
        / "research"
        / "rcle"
        / "RCLE_PHASE_B_BONN_HISTORICAL_EXCLUSION_MANIFEST_2026-07-26.json"
    )
    exclusions, _ = load_historical_exclusions(historical_path)
    decision = build_metadata_decision(inventory, exclusions)
    status_short = _git(["status", "--short"], repo_root).splitlines()
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "historical_exclusion_manifest_sha256": HISTORICAL_MANIFEST_SHA256,
        "implementation_lock_sha256": sha256_file(lock_path),
        "official_page": {
            "sha256": sha256_file(official_page_path),
            "bytes": official_page_path.stat().st_size,
            "payload_members_read": 0,
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "command": list(command),
        "repo": {
            "head": _git(["rev-parse", "HEAD"], repo_root),
            "branch": _git(["branch", "--show-current"], repo_root),
            "dirty": bool(status_short),
            "status_short": status_short,
        },
        "environment": environment_manifest(),
        "control_source_manifest": lock["control_source_manifest"],
        "selection_contract": {
            "official_universe_sha256": OFFICIAL_PAGE_SHA256,
            "selection_salt": SELECTION_SALT,
            "selection_hash": "SHA256(selection_salt + TAB + sequence_id)",
            "maximum_display_size_mb": MAXIMUM_DISPLAY_SIZE_MB,
            "required_selected_sequence_count": SELECTED_SEQUENCE_COUNT,
            "replacement_permitted": False,
        },
        **decision,
        "read_firewall": {
            "rgb_payload_members_read": 0,
            "depth_payload_members_read": 0,
            "pose_numeric_values_read": 0,
            "static_map_points_read": 0,
            "video_visual_inspections": 0,
            "legacy_trace_support_residual_score_reads": 0,
            "candidate_signal_computed": False,
            "phase_b_metrics_computed": False,
        },
        "authority": (
            "METADATA_ONLY_SOURCE_AUTHORITY_AND_COHORT_IDENTITY"
        ),
        "formal_phase_b_authorized": False,
        "payload_access_authorized": False,
    }


def validate_receipt_shape(receipt: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "candidate_id",
        "design_lock_sha256",
        "historical_exclusion_manifest_sha256",
        "implementation_lock_sha256",
        "official_page",
        "selection_contract",
        "metadata_selection_denominator",
        "selected_sequence_ids",
        "cohort_identity_sha256",
        "read_firewall",
        "gate_pass",
        "terminal_state",
        "authority",
        "formal_phase_b_authorized",
        "payload_access_authorized",
    }
    missing = sorted(required - receipt.keys())
    if missing:
        raise ValueError("RECEIPT_SCHEMA_MISSING:" + ",".join(missing))
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError("RECEIPT_SCHEMA_VERSION_MISMATCH")
    if len(receipt["metadata_selection_denominator"]) != 26:
        raise ValueError("RECEIPT_DENOMINATOR_COUNT_MISMATCH")
    if receipt["selected_sequence_count"] != 6:
        raise ValueError("RECEIPT_SELECTED_COUNT_MISMATCH")
    if receipt["formal_phase_b_authorized"]:
        raise ValueError("RECEIPT_FORMAL_PHASE_B_AUTHORITY_VIOLATION")
    if receipt["payload_access_authorized"]:
        raise ValueError("RECEIPT_PAYLOAD_AUTHORITY_VIOLATION")
    firewall = receipt["read_firewall"]
    numeric_zero = [
        key
        for key, value in firewall.items()
        if isinstance(value, int) and not isinstance(value, bool) and value != 0
    ]
    boolean_false = [
        key
        for key, value in firewall.items()
        if isinstance(value, bool) and value
    ]
    if numeric_zero or boolean_false:
        raise ValueError("RECEIPT_READ_FIREWALL_VIOLATION")


def validate_existing(
    repo_root: Path,
    official_page_path: Path,
    lock_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_receipt_shape(stored)
    recomputed = build_receipt(
        repo_root=repo_root,
        official_page_path=official_page_path,
        lock_path=lock_path,
        command=stored["command"],
        started_at=stored["started_at"],
        finished_at=stored["finished_at"],
    )
    volatile = {"repo"}
    stored_comparable = {
        key: value for key, value in stored.items() if key not in volatile
    }
    recomputed_comparable = {
        key: value for key, value in recomputed.items() if key not in volatile
    }
    if canonical_json(stored_comparable) != canonical_json(
        recomputed_comparable
    ):
        raise ValueError("RECEIPT_RECOMPUTE_MISMATCH")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "VALID",
        "receipt_sha256": sha256_file(receipt_path),
        "implementation_lock_sha256": sha256_file(lock_path),
        "official_page_sha256": sha256_file(official_page_path),
        "metadata_selection_denominator_count": stored[
            "metadata_selection_denominator_count"
        ],
        "selected_sequence_count": stored["selected_sequence_count"],
        "cohort_identity_sha256": stored["cohort_identity_sha256"],
        "gate_pass": stored["gate_pass"],
        "terminal_state": stored["terminal_state"],
        "payload_access_authorized": False,
        "formal_phase_b_authorized": False,
    }

