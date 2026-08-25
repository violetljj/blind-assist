#!/usr/bin/env python3
"""Guarded host runner for one resumable R1C-L Docker collection role."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


DOCKER = Path(os.environ["BLINDASSIST_DOCKER"]) if os.environ.get("BLINDASSIST_DOCKER") else Path(shutil.which("docker") or "docker")
IMAGE = "blindassist-grail-procthor:5.0.0-r1b"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _docker(*arguments: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(DOCKER), *arguments], check=False, text=True,
                          capture_output=capture, shell=False)


def launch(repo: Path, dataset: Path, manifest_path: Path, role: str, output: Path) -> int:
    repo = repo.resolve()
    dataset = dataset.resolve()
    manifest_path = manifest_path.resolve()
    output = output.resolve()
    if not DOCKER.is_file():
        raise RuntimeError("R1C-L Docker executable is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blindassist_grail_r1c_l_manifest_v3":
        raise ValueError("R1C-L guarded collection requires manifest v3")
    inspect = _docker("image", "inspect", IMAGE, "--format", "{{.Id}}", capture=True)
    if inspect.returncode != 0 or inspect.stdout.strip() != manifest["source"]["runtime_image_id"]:
        raise RuntimeError("R1C-L Docker image identity mismatch")
    artifact_root = (repo / "artifacts.local").resolve()
    try:
        dataset.relative_to(artifact_root)
        output.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("R1C-L dataset/output must resolve under artifacts.local") from error
    cache = (repo / "artifacts.local" / "models" / "ai2thor5-cache").resolve()
    container_name = f"grail-r1cl-v3-{role}"
    existing = _docker("ps", "-a", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}",
                       capture=True)
    if existing.returncode != 0:
        raise RuntimeError(existing.stderr.strip() or "R1C-L cannot inventory Docker containers")
    if existing.stdout.strip():
        raise RuntimeError(f"R1C-L task container already exists: {container_name}")
    dataset_inside = f"/repo/artifacts.local/{dataset.relative_to(artifact_root).as_posix()}"
    output_inside = f"/repo/artifacts.local/{output.relative_to(artifact_root).as_posix()}"
    manifest_inside = f"/repo/{manifest_path.relative_to(repo).as_posix()}"
    command = [
        str(DOCKER), "run", "--rm", "--name", container_name,
        "--mount", f"type=bind,source={repo},target=/repo,readonly",
        "--mount", f"type=bind,source={artifact_root},target=/repo/artifacts.local",
        "--mount", f"type=bind,source={cache},target=/root/.ai2thor",
        "-w", "/repo/research/active/grail-r1cl", IMAGE, "python",
        "collect_grail_pairwise_owner_coordinate_r1cl.py",
        "--dataset", dataset_inside, "--manifest", manifest_inside,
        "--role", role, "--output", output_inside,
    ]
    role_root = output / role
    failure_path = role_root / "failure.json"
    progress_path = role_root / "progress.json"
    try:
        process = subprocess.run(command, check=False, shell=False)
        if process.returncode != 0:
            _atomic_json(failure_path, {
                "schema": "blindassist_grail_r1c_l_collection_failure_v1", "role": role,
                "exit_code": process.returncode, "at": _now(),
            })
            completed = total = 0
            if progress_path.exists():
                try:
                    previous = json.loads(progress_path.read_text(encoding="utf-8"))
                    completed, total = previous.get("completed_units", 0), previous.get("total_units", 0)
                except (OSError, json.JSONDecodeError):
                    pass
            _atomic_json(progress_path, {
                "phase": "collection", "completed_units": completed, "total_units": total,
                "throughput": 0.0, "eta_seconds": None, "last_progress_at": _now(), "status": "failed",
            })
        return process.returncode
    finally:
        remaining = _docker("ps", "-a", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}",
                            capture=True)
        if remaining.returncode == 0 and remaining.stdout.strip() == container_name:
            _docker("rm", "-f", container_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return launch(args.repo, args.dataset, args.manifest, args.role, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
