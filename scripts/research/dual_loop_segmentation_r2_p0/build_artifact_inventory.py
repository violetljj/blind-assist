"""Build a machine-readable SHA inventory for the complete R2-P0 delivery."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import PROTOCOL_ID
from .build_readiness_lock import sha256_file


TRACKED_SINGLETONS = [
    "DEVELOPMENT_LOG.md",
    "docs/README.md",
    "docs/research/dual-loop/README.md",
    "docs/research/dual-loop/"
    "DUAL_LOOP_SEGMENTATION_R2_P0_PROTOCOL_DRAFT_2026-07-31.json",
    "docs/research/dual-loop/"
    "DUAL_LOOP_SEGMENTATION_R2_P0_DATASET_ROLE_LEDGER_2026-07-31.json",
    "docs/research/dual-loop/"
    "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_CONSUMED_ROLE_AMENDMENT_2026-08-01.json",
    "docs/research/dual-loop/"
    "DUAL_LOOP_SEGMENTATION_R2_P0_RESULT_2026-08-01.md",
    "scripts/README.md",
    "scripts/research/dual_loop_segmentation_model_selection/README.md",
]
TRACKED_TREES = [
    "configs/dual_loop_segmentation_r2_p0",
    "scripts/research/dual_loop_segmentation_r2_p0",
]
EVIDENCE_ROOT = (
    "artifacts.local/evidence/dual-loop-segmentation-r2-p0"
)
EVIDENCE_EXCLUDED_PREFIXES = [
    f"{EVIDENCE_ROOT}/canonical-view/masks/",
    f"{EVIDENCE_ROOT}/rehearsal-ddrnet-baseline/",
]
SELF_PATH = (
    f"{EVIDENCE_ROOT}/artifact-inventory.json"
)


def _relative_files(repo_root: Path, roots: list[str]) -> list[str]:
    values: list[str] = []
    for root_text in roots:
        root = repo_root / root_text
        if root.is_file():
            values.append(root_text)
            continue
        if not root.is_dir():
            raise FileNotFoundError(root)
        values.extend(
            path.relative_to(repo_root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    return sorted(set(values))


def _identity(repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = repo_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build(repo_root: Path) -> dict[str, Any]:
    tracked_paths = _relative_files(
        repo_root,
        TRACKED_SINGLETONS + TRACKED_TREES,
    )
    if SELF_PATH in tracked_paths:
        tracked_paths.remove(SELF_PATH)
    evidence_paths = _relative_files(repo_root, [EVIDENCE_ROOT])
    evidence_paths = [
        path
        for path in evidence_paths
        if path != SELF_PATH
        and not any(path.startswith(prefix) for prefix in EVIDENCE_EXCLUDED_PREFIXES)
    ]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    test_paths = [
        path
        for path in tracked_paths
        if Path(path).name.startswith("test_") and path.endswith(".py")
    ]
    return {
        "schema_version":
            "blindassist.dual_loop_segmentation_r2_p0.artifact_inventory.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "SHA_INVENTORY_COMPLETE",
        "base_head_before_delivery_commit": head,
        "tracked_file_count_excluding_inventory_self": len(tracked_paths),
        "tracked_files": [
            _identity(repo_root, path) for path in tracked_paths
        ],
        "test_file_count": len(test_paths),
        "test_files": test_paths,
        "local_evidence_file_count": len(evidence_paths),
        "local_evidence_files": [
            _identity(repo_root, path) for path in evidence_paths
        ],
        "excluded_from_per_file_inventory": {
            f"{EVIDENCE_ROOT}/canonical-view/masks/":
                "924 canonical masks are individually SHA-closed by canonical-view/manifest.jsonl",
            f"{EVIDENCE_ROOT}/rehearsal-ddrnet-baseline/":
                "superseded pre-v2 rehearsal output; not cited as evidence",
            SELF_PATH:
                "inventory cannot contain its own stable hash",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_atomic(output: Path, value: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact inventory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    inventory = build(args.repo_root.resolve())
    write_atomic(args.output.resolve(), inventory)
    print(
        json.dumps(
            {
                "status": inventory["status"],
                "tracked_files":
                    inventory["tracked_file_count_excluding_inventory_self"],
                "tests": inventory["test_file_count"],
                "local_evidence_files":
                    inventory["local_evidence_file_count"],
            }
        )
    )
