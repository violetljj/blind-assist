from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_authority_r1.authority import (
    canonical_json,
    recompute_fixed_content,
)
from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_entry_r0.gate import (
    sha256_file,
)


CANDIDATE_ID = "BONN_PRECLAIM_FIRST_METADATA_AUTHORITY_ADOPTION_R2"
DESIGN_LOCK_SHA256 = (
    "3dd53083b4fd47b0c5332926a441823ce2e70ed82086d30ca01d125aade22ed9"
)
R0_RECEIPT_SHA256 = (
    "4386bbe3b617abca3b73fc3070a65cef403fe270c12fd25f5034a579882f1764"
)
R1_RECEIPT_SHA256 = (
    "c2efac24585890f83fe9311e2d0bd6fd6155746a4c585934e5ad42fb27e9ed92"
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
RECEIPT_SCHEMA_VERSION = "rcle.phase_b.bonn_metadata_authority_r2.receipt.v1"
VALIDATION_SCHEMA_VERSION = (
    "rcle.phase_b.bonn_metadata_authority_r2.recompute.v1"
)
PASS_TERMINAL = "CANONICAL_METADATA_AUTHORITY_R2_PASS_FORMAL_PHASE_B_B0_READY"


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
        / "rcle_phase_b_bonn_entry_r2"
        / "authority_gate_r2"
    )
    return {
        "design_lock": (
            repo_root
            / "docs"
            / "research"
            / "rcle"
            / "RCLE_PHASE_B_BONN_METADATA_AUTHORITY_R2_DESIGN_LOCK_2026-07-26.json"
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
        "r1_receipt": (
            repo_root
            / "artifacts.local"
            / "evidence"
            / "rcle_phase_b_bonn_entry_r1"
            / "authority_gate_r1"
            / "receipt.json"
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
            / "rcle_phase_b_bonn_authority_r2"
            / "RCLE_PHASE_B_BONN_METADATA_AUTHORITY_R2_IMPLEMENTATION_LOCK.json"
        ),
        "output": output,
        "setup_marker": output / "directory_setup.json",
        "run_claim": output / "run_claim.json",
        "receipt": output / "receipt.json",
        "validation": output / "receipt_validation.json",
    }


def create_preclaim_first(path: Path, claim: dict[str, Any]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        os.write(
            descriptor,
            (json.dumps(claim, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
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
    repo_root: Path, paths: dict[str, Path]
) -> dict[str, Any]:
    lock = json.loads(paths["implementation_lock"].read_text(encoding="utf-8"))
    expected_paths = {
        key: str(paths[key])
        for key in [
            "official_page",
            "r0_receipt",
            "r1_receipt",
            "output",
            "setup_marker",
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
        "r0": (
            lock.get("r0_receipt_sha256") == R0_RECEIPT_SHA256
            and sha256_file(paths["r0_receipt"]) == R0_RECEIPT_SHA256
        ),
        "r1": (
            lock.get("r1_diagnostic_receipt_sha256") == R1_RECEIPT_SHA256
            and sha256_file(paths["r1_receipt"]) == R1_RECEIPT_SHA256
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
        "setup": lock.get("directory_setup_sha256")
        == sha256_file(paths["setup_marker"]),
        "preclaim": lock.get("first_application_file_operation")
        == "EXCLUSIVE_CREATE_CANONICAL_RUN_CLAIM",
        "runs": lock.get("maximum_materialization_claims") == 1,
        "schema": lock.get("receipt_schema_version")
        == RECEIPT_SCHEMA_VERSION,
        "payload": lock.get("payload_reads_authorized") is False,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError("R2_IMPLEMENTATION_LOCK_MISMATCH:" + ",".join(failed))
    return lock


def recompute_content(
    repo_root: Path, paths: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, Any]]:
    r1_paths = {
        **paths,
        "r0_receipt": paths["r0_receipt"],
        "official_page": paths["official_page"],
        "historical_manifest": paths["historical_manifest"],
    }
    r0, decision = recompute_fixed_content(r1_paths)
    r1 = json.loads(paths["r1_receipt"].read_text(encoding="utf-8"))
    checks = {
        "r1_role": r1.get("r0_role")
        == "NONAUTHORITATIVE_DIAGNOSTIC_CONTENT_INPUT",
        "r1_cohort": r1.get("cohort_identity_sha256")
        == COHORT_IDENTITY_SHA256,
        "r1_selected": r1.get("selected_sequence_ids")
        == decision["selected_sequence_ids"],
        "r1_no_payload": r1.get("payload_inventory_authorized") is False,
        "counts": (
            decision["official_universe_count"] == 26
            and decision["historical_exclusion_count"] == 9
            and decision["selected_sequence_count"] == 6
        ),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError("R2_CONTENT_MISMATCH:" + ",".join(failed))
    return r1, decision


def build_receipt(
    repo_root: Path,
    paths: dict[str, Path],
    started_at: str,
    finished_at: str,
    command: Sequence[str],
) -> dict[str, Any]:
    lock = validate_implementation_lock(repo_root, paths)
    r1, decision = recompute_content(repo_root, paths)
    status_short = _git(["status", "--short"], repo_root).splitlines()
    denominator_hash = hashlib.sha256(
        canonical_json(
            decision["metadata_selection_denominator"]
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "implementation_lock_sha256": sha256_file(
            paths["implementation_lock"]
        ),
        "run_claim_sha256": sha256_file(paths["run_claim"]),
        "directory_setup_sha256": sha256_file(paths["setup_marker"]),
        "r0_receipt_sha256": R0_RECEIPT_SHA256,
        "r1_diagnostic_receipt_sha256": R1_RECEIPT_SHA256,
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
        "official_universe_count": 26,
        "historical_exclusion_count": 9,
        "selected_sequence_count": 6,
        "selected_sequence_ids": decision["selected_sequence_ids"],
        "cohort_identity_sha256": COHORT_IDENTITY_SHA256,
        "metadata_selection_denominator_sha256": denominator_hash,
        "r1_denominator_sha256": r1[
            "metadata_selection_denominator_sha256"
        ],
        "read_firewall": {
            "archive_payload_reads": 0,
            "rgb_depth_pose_map_reads": 0,
            "legacy_outcome_reads": 0,
            "phase_b_metric_reads_or_computations": 0,
        },
        "gate_pass": True,
        "terminal_state": PASS_TERMINAL,
        "authority": "PRECLAIM_FIRST_CANONICAL_METADATA_COHORT_IDENTITY",
        "formal_phase_b_b0_ready": True,
        "payload_reads_authorized": False,
        "phase_b_metrics_authorized": False,
    }


def validate_existing(repo_root: Path) -> dict[str, Any]:
    paths = canonical_paths(repo_root)
    lock = validate_implementation_lock(repo_root, paths)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    _, decision = recompute_content(repo_root, paths)
    denominator_hash = hashlib.sha256(
        canonical_json(
            decision["metadata_selection_denominator"]
        ).encode("utf-8")
    ).hexdigest()
    checks = {
        "schema": receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION,
        "lock": receipt.get("implementation_lock_sha256")
        == sha256_file(paths["implementation_lock"]),
        "claim": receipt.get("run_claim_sha256")
        == sha256_file(paths["run_claim"]),
        "cohort": receipt.get("cohort_identity_sha256")
        == COHORT_IDENTITY_SHA256,
        "denominator": receipt.get("metadata_selection_denominator_sha256")
        == denominator_hash,
        "selected": receipt.get("selected_sequence_ids")
        == decision["selected_sequence_ids"],
        "terminal": receipt.get("terminal_state") == PASS_TERMINAL,
        "controls": receipt.get("control_source_manifest")
        == lock["control_source_manifest"],
        "payload": receipt.get("payload_reads_authorized") is False,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError("R2_RECEIPT_RECOMPUTE_MISMATCH:" + ",".join(failed))
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "VALID",
        "receipt_sha256": sha256_file(paths["receipt"]),
        "run_claim_sha256": sha256_file(paths["run_claim"]),
        "implementation_lock_sha256": sha256_file(
            paths["implementation_lock"]
        ),
        "cohort_identity_sha256": COHORT_IDENTITY_SHA256,
        "selected_sequence_count": 6,
        "gate_pass": True,
        "terminal_state": PASS_TERMINAL,
        "formal_phase_b_b0_ready": True,
        "payload_reads_authorized": False,
        "phase_b_metrics_authorized": False,
    }
