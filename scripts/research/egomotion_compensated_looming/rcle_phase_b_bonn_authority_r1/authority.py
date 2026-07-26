from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_entry_r0.gate import (
    build_metadata_decision,
    canonical_json,
    load_historical_exclusions,
    parse_official_inventory,
    sha256_file,
)


CANDIDATE_ID = "BONN_CANONICAL_METADATA_AUTHORITY_ADOPTION_R1"
DESIGN_LOCK_SHA256 = (
    "f9724b044ac4ae0cd5e34ec18b8de834b50616b0b6d626a6c790a9c22d1ea25e"
)
R0_RECEIPT_SHA256 = (
    "4386bbe3b617abca3b73fc3070a65cef403fe270c12fd25f5034a579882f1764"
)
R0_IMPLEMENTATION_LOCK_SHA256 = (
    "a47cd39ea82c10828290def8bae54f61b28676190c8ab06acc93217b1590a617"
)
OFFICIAL_PAGE_SHA256 = (
    "2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186"
)
HISTORICAL_MANIFEST_SHA256 = (
    "f02bd9f1313def45cc107d72ace5f7c7803f4ab816bf6e98c5f9173fa3bb1cc6"
)
COHORT_IDENTITY_SHA256 = (
    "513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e"
)
RECEIPT_SCHEMA_VERSION = "rcle.phase_b.bonn_metadata_authority_r1.receipt.v1"
VALIDATION_SCHEMA_VERSION = (
    "rcle.phase_b.bonn_metadata_authority_r1.recompute.v1"
)
PASS_TERMINAL = (
    "CANONICAL_METADATA_AUTHORITY_PASS_FORMAL_PHASE_B_PROTOCOL_MAY_BE_FROZEN"
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


def canonical_paths(repo_root: Path) -> dict[str, Path]:
    module = (
        repo_root / "scripts" / "research" / "egomotion_compensated_looming"
    )
    output = (
        repo_root
        / "artifacts.local"
        / "evidence"
        / "rcle_phase_b_bonn_entry_r1"
        / "authority_gate_r1"
    )
    return {
        "design_lock": (
            repo_root
            / "docs"
            / "research"
            / "rcle"
            / "RCLE_PHASE_B_BONN_METADATA_AUTHORITY_R1_DESIGN_LOCK_2026-07-26.json"
        ),
        "historical_manifest": (
            repo_root
            / "docs"
            / "research"
            / "rcle"
            / "RCLE_PHASE_B_BONN_HISTORICAL_EXCLUSION_MANIFEST_2026-07-26.json"
        ),
        "r0_receipt": (
            repo_root
            / "artifacts.local"
            / "evidence"
            / "rcle_phase_b_bonn_entry_r0"
            / "metadata_gate_r0"
            / "receipt.json"
        ),
        "r0_implementation_lock": (
            module
            / "rcle_phase_b_bonn_entry_r0"
            / "RCLE_PHASE_B_BONN_METADATA_GATE_R0_IMPLEMENTATION_LOCK.json"
        ),
        "official_page": (
            repo_root
            / "artifacts.local"
            / "datasets"
            / "egomotion_compensated_looming_r1"
            / "bonn_metadata_r0"
            / "official_page.html"
        ),
        "implementation_lock": (
            module
            / "rcle_phase_b_bonn_authority_r1"
            / "RCLE_PHASE_B_BONN_METADATA_AUTHORITY_R1_IMPLEMENTATION_LOCK.json"
        ),
        "output": output,
        "run_claim": output / "run_claim.json",
        "receipt": output / "receipt.json",
        "validation": output / "receipt_validation.json",
    }


def create_exclusive_claim(path: Path, claim: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        payload = (
            json.dumps(claim, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def control_manifest(
    repo_root: Path, lock: dict[str, Any]
) -> dict[str, str]:
    return {
        path: sha256_file(repo_root / path)
        for path in sorted(lock["control_source_manifest"])
    }


def validate_implementation_lock(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = canonical_paths(repo_root)
    lock = json.loads(paths["implementation_lock"].read_text(encoding="utf-8"))
    expected_paths = {
        name: str(paths[name])
        for name in [
            "official_page",
            "r0_receipt",
            "r0_implementation_lock",
            "output",
            "run_claim",
            "receipt",
            "validation",
        ]
    }
    checks = {
        "candidate": lock.get("candidate_id") == CANDIDATE_ID,
        "design": (
            lock.get("design_lock_sha256") == DESIGN_LOCK_SHA256
            and sha256_file(paths["design_lock"]) == DESIGN_LOCK_SHA256
        ),
        "r0_receipt": (
            lock.get("r0_receipt_sha256") == R0_RECEIPT_SHA256
            and sha256_file(paths["r0_receipt"]) == R0_RECEIPT_SHA256
        ),
        "r0_lock": (
            lock.get("r0_implementation_lock_sha256")
            == R0_IMPLEMENTATION_LOCK_SHA256
            and sha256_file(paths["r0_implementation_lock"])
            == R0_IMPLEMENTATION_LOCK_SHA256
        ),
        "official": (
            lock.get("official_page_sha256") == OFFICIAL_PAGE_SHA256
            and sha256_file(paths["official_page"]) == OFFICIAL_PAGE_SHA256
        ),
        "historical": (
            lock.get("historical_manifest_sha256")
            == HISTORICAL_MANIFEST_SHA256
            and sha256_file(paths["historical_manifest"])
            == HISTORICAL_MANIFEST_SHA256
        ),
        "cohort": lock.get("cohort_identity_sha256")
        == COHORT_IDENTITY_SHA256,
        "paths": lock.get("canonical_paths") == expected_paths,
        "environment": lock.get("environment") == environment_manifest(),
        "controls": lock.get("control_source_manifest")
        == control_manifest(repo_root, lock),
        "claim": lock.get("exclusive_run_claim_before_metadata_read") is True,
        "runs": lock.get("maximum_materialization_claims") == 1,
        "schema": lock.get("receipt_schema_version")
        == RECEIPT_SCHEMA_VERSION,
        "payload": lock.get("payload_inventory_authorized") is False,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError("R1_IMPLEMENTATION_LOCK_MISMATCH:" + ",".join(failed))
    return lock, paths


def recompute_fixed_content(
    paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    r0 = json.loads(paths["r0_receipt"].read_text(encoding="utf-8"))
    inventory = parse_official_inventory(
        paths["official_page"].read_text(encoding="utf-8")
    )
    exclusions, _ = load_historical_exclusions(paths["historical_manifest"])
    decision = build_metadata_decision(inventory, exclusions)
    checks = {
        "r0_schema": r0.get("schema_version")
        == "rcle.phase_b.bonn_metadata_gate.receipt.v1",
        "r0_content": r0.get("metadata_selection_denominator")
        == decision["metadata_selection_denominator"],
        "r0_selected": r0.get("selected_sequence_ids")
        == decision["selected_sequence_ids"],
        "cohort": decision["cohort_identity_sha256"]
        == COHORT_IDENTITY_SHA256,
        "counts": (
            decision["official_universe_count"] == 26
            and decision["historical_exclusion_count"] == 9
            and decision["selected_sequence_count"] == 6
        ),
        "r0_firewall": all(
            value is False if isinstance(value, bool) else value == 0
            for value in r0["read_firewall"].values()
        ),
        "r0_no_authority": (
            r0.get("payload_access_authorized") is False
            and r0.get("formal_phase_b_authorized") is False
        ),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError("R1_FIXED_CONTENT_MISMATCH:" + ",".join(failed))
    return r0, decision


def build_receipt(
    repo_root: Path,
    started_at: str,
    finished_at: str,
    command: Sequence[str],
) -> dict[str, Any]:
    lock, paths = validate_implementation_lock(repo_root)
    claim_sha = sha256_file(paths["run_claim"])
    r0, decision = recompute_fixed_content(paths)
    status_short = _git(["status", "--short"], repo_root).splitlines()
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "implementation_lock_sha256": sha256_file(
            paths["implementation_lock"]
        ),
        "run_claim_sha256": claim_sha,
        "r0_receipt_sha256": R0_RECEIPT_SHA256,
        "r0_role": "NONAUTHORITATIVE_DIAGNOSTIC_CONTENT_INPUT",
        "official_page_sha256": OFFICIAL_PAGE_SHA256,
        "historical_manifest_sha256": HISTORICAL_MANIFEST_SHA256,
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
        "official_universe_count": decision["official_universe_count"],
        "historical_exclusion_count": decision[
            "historical_exclusion_count"
        ],
        "selected_sequence_count": decision["selected_sequence_count"],
        "selected_sequence_ids": decision["selected_sequence_ids"],
        "cohort_identity_sha256": decision["cohort_identity_sha256"],
        "metadata_selection_denominator_sha256": hashlib.sha256(
            canonical_json(
                decision["metadata_selection_denominator"]
            ).encode("utf-8")
        ).hexdigest(),
        "r0_denominator_sha256": hashlib.sha256(
            canonical_json(r0["metadata_selection_denominator"]).encode(
                "utf-8"
            )
        ).hexdigest(),
        "read_firewall": {
            "archive_payload_reads": 0,
            "rgb_depth_pose_map_reads": 0,
            "legacy_outcome_reads": 0,
            "phase_b_metric_reads_or_computations": 0,
        },
        "gate_pass": True,
        "terminal_state": PASS_TERMINAL,
        "authority": "CANONICAL_METADATA_COHORT_IDENTITY_ONLY",
        "payload_inventory_authorized": False,
        "phase_b_metrics_authorized": False,
    }


def validate_existing(repo_root: Path) -> dict[str, Any]:
    lock, paths = validate_implementation_lock(repo_root)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    r0, decision = recompute_fixed_content(paths)
    checks = {
        "schema": receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION,
        "lock": receipt.get("implementation_lock_sha256")
        == sha256_file(paths["implementation_lock"]),
        "claim": receipt.get("run_claim_sha256")
        == sha256_file(paths["run_claim"]),
        "r0": receipt.get("r0_receipt_sha256") == R0_RECEIPT_SHA256,
        "selected": receipt.get("selected_sequence_ids")
        == decision["selected_sequence_ids"],
        "cohort": receipt.get("cohort_identity_sha256")
        == COHORT_IDENTITY_SHA256,
        "denominator": receipt.get("metadata_selection_denominator_sha256")
        == hashlib.sha256(
            canonical_json(
                decision["metadata_selection_denominator"]
            ).encode("utf-8")
        ).hexdigest(),
        "r0_denominator": receipt.get("r0_denominator_sha256")
        == hashlib.sha256(
            canonical_json(r0["metadata_selection_denominator"]).encode(
                "utf-8"
            )
        ).hexdigest(),
        "terminal": receipt.get("terminal_state") == PASS_TERMINAL,
        "payload": receipt.get("payload_inventory_authorized") is False,
        "controls": receipt.get("control_source_manifest")
        == lock["control_source_manifest"],
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError("R1_RECEIPT_RECOMPUTE_MISMATCH:" + ",".join(failed))
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "VALID",
        "receipt_sha256": sha256_file(paths["receipt"]),
        "run_claim_sha256": sha256_file(paths["run_claim"]),
        "implementation_lock_sha256": sha256_file(
            paths["implementation_lock"]
        ),
        "selected_sequence_count": 6,
        "cohort_identity_sha256": COHORT_IDENTITY_SHA256,
        "gate_pass": True,
        "terminal_state": PASS_TERMINAL,
        "payload_inventory_authorized": False,
        "phase_b_metrics_authorized": False,
    }
