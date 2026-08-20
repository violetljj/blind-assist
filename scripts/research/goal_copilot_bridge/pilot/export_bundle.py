"""Create and verify the immutable development SearchTaskBundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
PROTOCOL_ID = "GOAL-COPILOT-1-SKY-PILOT"
PAYLOADS = (
    "dev_scenarios.json", "evaluator.py", "initial_policy.py",
    "pilot_protocol.json", "search_prompt.md", "task_api.py",
)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksums(root: Path) -> dict[str, str]:
    return {name: sha256(root / name) for name in sorted(PAYLOADS)}


def verify_bundle(root: Path) -> dict[str, Any]:
    actual_members = {item.name for item in root.iterdir() if item.is_file()}
    expected_members = {*PAYLOADS, "checksums.json", "manifest.json"}
    if actual_members != expected_members:
        raise ValueError(f"bundle member mismatch: {sorted(actual_members)}")
    checksums = json.loads((root / "checksums.json").read_text())
    if checksums != _checksums(root):
        raise ValueError("bundle checksum mismatch")
    digest = hashlib.sha256(canonical(checksums)).hexdigest()
    manifest = json.loads((root / "manifest.json").read_text())
    if root.name != digest or manifest.get("bundle_digest") != digest:
        raise ValueError("bundle identity mismatch")
    if manifest.get("fresh_material_exported") is not False:
        raise ValueError("fresh material leakage")
    return manifest


def export_bundle(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="pilot-bundle-", dir=output_root))
    try:
        for name in PAYLOADS:
            shutil.copy2(HERE / name, staging / name)
        checksums = _checksums(staging)
        digest = hashlib.sha256(canonical(checksums)).hexdigest()
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        (staging / "checksums.json").write_bytes(canonical(checksums))
        (staging / "manifest.json").write_bytes(canonical({
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "bundle_digest": digest,
            "blindassist_commit": commit,
            "search_authority": "SKYDISCOVER_PROPOSAL_ONLY",
            "acceptance_authority": "BLINDASSIST_ONLY",
            "payload_files": sorted(PAYLOADS),
            "fresh_material_exported": False,
        }))
        destination = output_root / "SearchTaskBundle" / digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(staging)
        else:
            os.replace(staging, destination)
        verify_bundle(destination)
        return destination
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(export_bundle(args.output_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
