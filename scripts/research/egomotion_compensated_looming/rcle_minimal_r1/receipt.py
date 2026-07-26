from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import cv2
import matplotlib
import numpy as np
from PIL import __version__ as pillow_version

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    PROTOCOL_SHA256,
    load_protocol,
    sha256_file,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.receipt import (
    canonical_json,
    read_jsonl,
    runtime_summary,
    write_json,
    write_jsonl,
)

from .evaluation import IMPLEMENTATION_REVISION, summarize_and_decide


def _git(args: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _manifest(root: Path, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = excluded or set()
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            result[relative] = sha256_file(path)
    return result


def source_manifest(repo_root: Path) -> dict[str, str]:
    module = (
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
    )
    files: list[Path] = []
    for root in (module / "rcle_minimal_r1", module / "tests_r1"):
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".md", ".txt"}
        )
    dependencies = [
        module / "run_synthetic_signal_audit_r1.py",
        module / "rcle_minimal" / "evaluation.py",
        module / "rcle_minimal" / "protocol.py",
        module / "rcle_minimal" / "receipt.py",
        module / "rcle_minimal" / "rotation_compensation.py",
        module / "rcle_minimal" / "synthetic_generator.py",
        module / "rcle_minimal" / "visualization.py",
        module / "configs" / "phase_a_synthetic_signal_audit_r0.json",
        module / "configs" / "phase_a_synthetic_signal_audit_r0.lock.json",
    ]
    files.extend(path for path in dependencies if path.exists())
    return {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in sorted(set(files))
    }


def build_receipt(
    repo_root: Path,
    output_root: Path,
    dataset_root: Path,
    command: Sequence[str],
    summary: dict[str, Any],
    started_at: str,
    finished_at: str,
    worker_count: int,
    implementation_preregistration_sha256: str,
) -> dict[str, Any]:
    dirty = _git(["status", "--short"], repo_root)
    return {
        "schema_version": "rcle.phase_a.coverage_revision.receipt.v1",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "implementation_preregistration_sha256": (
            implementation_preregistration_sha256
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "started_at": started_at,
        "finished_at": finished_at,
        "cwd": str(repo_root),
        "command": list(command),
        "worker_count": worker_count,
        "repo": {
            "head": _git(["rev-parse", "HEAD"], repo_root),
            "branch": _git(["branch", "--show-current"], repo_root),
            "dirty": bool(dirty),
            "status_short": dirty.splitlines(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": pillow_version,
            "matplotlib": matplotlib.__version__,
            "opencv_threads_in_worker": 1,
        },
        "source_manifest": source_manifest(repo_root),
        "dataset_manifest": _manifest(dataset_root),
        "output_manifest": _manifest(
            output_root,
            excluded={"receipt.json", "receipt_validation.json"},
        ),
        "scientific_summary_sha256": hashlib.sha256(
            canonical_json(summary).encode("utf-8")
        ).hexdigest(),
        "verdict": summary["verdict"],
        "authority": summary["authority"],
        "retained_r0_receipt_sha256": (
            "14ed23e38bacc913207aaa56903a7b2cd3bebe52631338c4760f02dc5c2041ca"
        ),
        "recompute_command": [
            r"E:\codex-tools\bin\blindassist-python.cmd",
            (
                "scripts/research/egomotion_compensated_looming/"
                "run_synthetic_signal_audit_r1.py"
            ),
            "--validate-existing",
            "--output-root",
            str(output_root),
            "--dataset-root",
            str(dataset_root),
        ],
    }


def validate_existing(
    repo_root: Path, output_root: Path, dataset_root: Path
) -> dict[str, Any]:
    protocol = load_protocol()
    receipt_path = output_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("RECEIPT_PROTOCOL_HASH_MISMATCH")
    if receipt["implementation_revision"] != IMPLEMENTATION_REVISION:
        raise ValueError("IMPLEMENTATION_REVISION_MISMATCH")
    rows = read_jsonl(output_root / "trial_metrics.jsonl")
    if any(
        row.get("implementation_revision") != IMPLEMENTATION_REVISION
        for row in rows
    ):
        raise ValueError("TRIAL_IMPLEMENTATION_REVISION_MISMATCH")
    recomputed = summarize_and_decide(rows, protocol)
    stored = json.loads(
        (output_root / "scientific_summary.json").read_text(encoding="utf-8")
    )
    if canonical_json(recomputed) != canonical_json(stored):
        raise ValueError("SCIENTIFIC_SUMMARY_RECOMPUTE_MISMATCH")
    scientific_hash = hashlib.sha256(
        canonical_json(recomputed).encode("utf-8")
    ).hexdigest()
    if scientific_hash != receipt["scientific_summary_sha256"]:
        raise ValueError("SCIENTIFIC_SUMMARY_HASH_MISMATCH")
    if source_manifest(repo_root) != receipt["source_manifest"]:
        raise ValueError("SOURCE_MANIFEST_MISMATCH")
    if _manifest(dataset_root) != receipt["dataset_manifest"]:
        raise ValueError("DATASET_MANIFEST_MISMATCH")
    current_output = _manifest(
        output_root,
        excluded={"receipt.json", "receipt_validation.json"},
    )
    if current_output != receipt["output_manifest"]:
        raise ValueError("OUTPUT_MANIFEST_MISMATCH")
    return {
        "schema_version": "rcle.phase_a.coverage_revision.validation.v1",
        "status": "VALID",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "receipt_sha256": sha256_file(receipt_path),
        "protocol_sha256": PROTOCOL_SHA256,
        "trial_count": len(rows),
        "verdict": recomputed["verdict"],
    }


__all__ = [
    "build_receipt",
    "runtime_summary",
    "source_manifest",
    "validate_existing",
    "write_json",
    "write_jsonl",
]
