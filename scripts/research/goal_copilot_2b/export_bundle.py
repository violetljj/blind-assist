"""Create and verify the immutable public GC2-B SearchTaskBundle."""

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
REPO_ROOT = HERE.parents[2]
GC1 = HERE.parent / "goal_copilot_bridge" / "pilot"
GC2A = HERE.parent / "goal_copilot_2a"
PROTOCOL_ID = "GOAL-COPILOT-2B"
FILES = {
    HERE / "search_evaluator.py": Path("evaluator.py"),
    HERE / "search_prompt.md": Path("search_prompt.md"),
    HERE / "protocol.json": Path("protocol.json"),
    GC2A / "frozen_gc1_winner.py": Path("initial_policy.py"),
    GC1 / "dev_scenarios.json": Path("dev_scenarios.json"),
    GC1 / "task_api.py": Path("scripts/research/goal_copilot_bridge/pilot/task_api.py"),
    GC1 / "evaluator.py": Path("scripts/research/goal_copilot_bridge/pilot/evaluator.py"),
    GC2A / "noise.py": Path("scripts/research/goal_copilot_2a/noise.py"),
    GC2A / "evaluator.py": Path("scripts/research/goal_copilot_2a/evaluator.py"),
}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksums(root: Path) -> dict[str, str]:
    return {
        relative.as_posix(): sha256(root / relative)
        for relative in sorted(FILES.values(), key=lambda item: item.as_posix())
    }


def verify_bundle(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    expected = {path.as_posix() for path in FILES.values()}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "checksums.json"}
    }
    if actual != expected or checksums != _checksums(root):
        raise ValueError("bundle member or checksum mismatch")
    digest = hashlib.sha256(canonical(checksums)).hexdigest()
    if root.name != digest or manifest.get("bundle_digest") != digest:
        raise ValueError("bundle identity mismatch")
    if manifest.get("heldout_material_exported") is not False:
        raise ValueError("held-out material leakage")
    if any("heldout" in path.name.lower() or "fresh" in path.name.lower() for path in root.rglob("*")):
        raise ValueError("hidden material named in public bundle")
    return manifest


def export_bundle(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="gc2b-bundle-", dir=output_root))
    try:
        for source, relative in FILES.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        checksums = _checksums(staging)
        digest = hashlib.sha256(canonical(checksums)).hexdigest()
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (staging / "checksums.json").write_bytes(canonical(checksums))
        (staging / "manifest.json").write_bytes(
            canonical(
                {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "bundle_digest": digest,
                    "blindassist_commit": commit,
                    "payload_files": sorted(checksums),
                    "starting_policy_sha256": checksums["initial_policy.py"],
                    "search_authority": "SKYDISCOVER_PROPOSAL_ONLY",
                    "acceptance_authority": "BLINDASSIST_ONLY",
                    "scenario_evidence_role": "CONSUMED_DEVELOPMENT_NOT_FRESH",
                    "heldout_material_exported": False,
                    "fresh_material_exported": False,
                }
            )
        )
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
