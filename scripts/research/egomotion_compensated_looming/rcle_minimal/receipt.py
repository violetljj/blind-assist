from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Sequence

import cv2
import matplotlib
import numpy as np
from PIL import __version__ as pillow_version

from .evaluation import summarize_and_decide
from .protocol import PROTOCOL_SHA256, load_protocol, sha256_file


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"INVALID_JSONL {path}:{line_number}"
                    ) from error
    return rows


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
    roots = [
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "rcle_minimal",
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "tests",
    ]
    files: list[Path] = []
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".md", ".txt"}
        )
    runner = (
        repo_root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "run_synthetic_signal_audit_r0.py"
    )
    if runner.exists():
        files.append(runner)
    return {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in sorted(set(files))
    }


def runtime_summary(runtime_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    modules: dict[str, list[float]] = {}
    for row in runtime_rows:
        pair_count = max(int(row["pair_count"]), 1)
        for name, total_ms in row["module_total_milliseconds"].items():
            modules.setdefault(name, []).append(float(total_ms) / pair_count)
        modules.setdefault("total", []).append(
            float(row["total_milliseconds"]) / pair_count
        )
    summaries: dict[str, Any] = {}
    for name, values in sorted(modules.items()):
        array = np.asarray(values, dtype=np.float64)
        summaries[name] = {
            "n_trials": int(array.size),
            "mean_ms_per_pair": float(np.mean(array)),
            "median_ms_per_pair": float(np.median(array)),
            "p95_ms_per_pair": float(np.quantile(array, 0.95)),
        }
    return {
        "schema_version": "rcle.phase_a.runtime_summary.v1",
        "modules": summaries,
        "measurement_note": (
            "Host offline timing under formal-run worker contention; "
            "not an Android or Kill Gate measurement."
        ),
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
) -> dict[str, Any]:
    dirty = _git(["status", "--short"], repo_root)
    return {
        "schema_version": "rcle.phase_a.receipt.v1",
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
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
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
        "recompute_command": [
            r"E:\codex-tools\bin\blindassist-python.cmd",
            (
                "scripts/research/egomotion_compensated_looming/"
                "run_synthetic_signal_audit_r0.py"
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
    rows = read_jsonl(output_root / "trial_metrics.jsonl")
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
    current_sources = source_manifest(repo_root)
    if current_sources != receipt["source_manifest"]:
        raise ValueError("SOURCE_MANIFEST_MISMATCH")
    current_dataset = _manifest(dataset_root)
    if current_dataset != receipt["dataset_manifest"]:
        raise ValueError("DATASET_MANIFEST_MISMATCH")
    current_output = _manifest(
        output_root,
        excluded={"receipt.json", "receipt_validation.json"},
    )
    if current_output != receipt["output_manifest"]:
        raise ValueError("OUTPUT_MANIFEST_MISMATCH")
    if receipt["verdict"] != recomputed["verdict"]:
        raise ValueError("RECEIPT_VERDICT_MISMATCH")
    return {
        "schema_version": "rcle.phase_a.receipt_validation.v1",
        "status": "VALID",
        "receipt_sha256": sha256_file(receipt_path),
        "protocol_sha256": PROTOCOL_SHA256,
        "trial_count": len(rows),
        "verdict": recomputed["verdict"],
    }
