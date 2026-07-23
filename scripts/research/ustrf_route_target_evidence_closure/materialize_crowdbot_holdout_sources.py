#!/usr/bin/env python3
"""Sequentially stream, normalize, verify, and release selected CrowdBot bags."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_bundle(sequence_dir: Path) -> bool:
    bundle_path = sequence_dir / "bundle.json"
    frames_path = sequence_dir / "frames.jsonl"
    if not bundle_path.is_file() or not frames_path.is_file():
        return False
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("candidate_outputs_executed") is not False:
        return False
    if bundle.get("frames_sha256") != sha256_file(frames_path):
        return False
    rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != bundle.get("rgb_frame_count"):
        return False
    for row in rows:
        rgb_path = sequence_dir / row["rgb_path"]
        if not rgb_path.is_file():
            return False
        if row["exact_aligned_depth"]:
            depth_path = sequence_dir / row["aligned_depth_path"]
            if not depth_path.is_file():
                return False
    return True


def find_receipt(evidence_root: Path, entry_name: str) -> Path | None:
    for path in evidence_root.glob("*bag-receipt-r1.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("entry") == entry_name:
                return path
        except (OSError, json.JSONDecodeError):
            continue
    return None


def run_checked(command: list[str], attempts: int = 4) -> None:
    for attempt in range(attempts):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2**attempt, 8))


def unlink_with_retry(path: Path, attempts: int = 12) -> None:
    """Release a verified raw bag despite transient Windows file-handle lag."""
    for attempt in range(attempts):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.5 * (2**attempt), 5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, metavar="SOURCE_ID=INVENTORY")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--replacement", type=Path)
    parser.add_argument("--range-workers", type=int)
    parser.add_argument("--range-parts", type=int)
    parser.add_argument("--compressed-cache-root", type=Path)
    args = parser.parse_args()
    if args.range_workers is not None and (
        args.range_workers < 1 or args.range_workers > 12
    ):
        raise RuntimeError("--range-workers must be between 1 and 12")
    if (
        args.range_workers is not None
        and args.range_workers > 1
        and args.compressed_cache_root is None
    ):
        raise RuntimeError("--compressed-cache-root is required when --range-workers is greater than 1")
    if args.range_parts is not None and args.range_workers is None:
        raise RuntimeError("--range-parts requires --range-workers")
    if args.range_parts is not None and (
        args.range_parts < args.range_workers or args.range_parts > 128
    ):
        raise RuntimeError("--range-parts must be between --range-workers and 128")
    script_root = Path(__file__).resolve().parent
    sources_by_id: dict[str, list[tuple[Path, dict[str, Any], list[dict[str, Any]]]]] = {}
    for value in args.source:
        if "=" not in value:
            raise RuntimeError("--source must use SOURCE_ID=INVENTORY")
        source_id, inventory_text = value.split("=", 1)
        inventory_path = Path(inventory_text)
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        entries = [row for row in inventory["entries"] if not row["is_directory"]]
        sources_by_id.setdefault(source_id, []).append((inventory_path, inventory, entries))
    sources = []
    for source_id, inventory_groups in sources_by_id.items():
        rows: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
        seen_entry_names: set[str] = set()
        for inventory_path, inventory, entries in inventory_groups:
            for entry in entries:
                if entry["name"] in seen_entry_names:
                    raise RuntimeError(f"duplicate raw entry across inventories: {source_id} {entry['name']}")
                seen_entry_names.add(entry["name"])
                rows.append((inventory_path, inventory, entry))
        sources.append((source_id, rows))
    if args.replacement is not None:
        replacement = json.loads(args.replacement.read_text(encoding="utf-8"))
        planned = replacement["planned_outputs"]
        repo = Path.cwd().resolve()
        if args.dataset_root.resolve() != (repo / planned["dataset_root"]).resolve():
            raise RuntimeError("replacement dataset root differs from preregistration")
        if args.evidence_root.resolve() != (repo / planned["evidence_root"]).resolve():
            raise RuntimeError("replacement evidence root differs from preregistration")
        if args.state.resolve() != (repo / planned["materialization_state"]).resolve():
            raise RuntimeError("replacement materialization state differs from preregistration")
        expected_sources = {
            row["source_id"]: {
                (repo / binding["path"]).resolve(): binding["sha256"]
                for binding in row["raw_inventories"]
            }
            for row in replacement["replacement_sources"]
        }
        actual_sources = {
            source_id: {inventory_path.resolve(): sha256_file(inventory_path) for inventory_path, _, _ in rows}
            for source_id, rows in sources_by_id.items()
        }
        if actual_sources != expected_sources:
            raise RuntimeError("replacement source inventory bindings differ from preregistration")
        if replacement["execution_boundaries"]["candidate_outputs_executed"] is not False:
            raise RuntimeError("replacement holdout is not candidate blind")
    state: dict[str, Any] = {
        "schema": "blindassist_crowdbot_holdout_materialization_state_r1",
        "candidate_outputs_executed": False,
        "status": "running",
        "replacement_preregistration_sha256": sha256_file(args.replacement) if args.replacement else None,
        "sequence_total": sum(len(source[1]) for source in sources),
        "sequence_completed": 0,
        "transport": {
            "selection": (
                "child_sidecar_auto_discovery"
                if args.range_workers is None
                else "explicit_materializer_arguments"
            ),
            "range_workers": args.range_workers,
            "range_parts": args.range_parts,
            "compressed_cache_root": (
                args.compressed_cache_root.as_posix() if args.compressed_cache_root is not None else None
            ),
        },
        "sources": {},
    }
    for source_id, rows in sources:
        source_state = state["sources"].setdefault(source_id, {"sequence_total": len(rows), "sequence_completed": 0})
        ordered_entries = sorted(
            rows,
            key=lambda row: (
                not valid_bundle(args.dataset_root / source_id / "sequences" / Path(row[2]["name"]).stem),
                row[2]["name"],
            ),
        )
        for inventory_path, _inventory, entry in ordered_entries:
            sequence_id = Path(entry["name"]).stem
            sequence_dir = args.dataset_root / source_id / "sequences" / sequence_id
            bag_path = args.dataset_root / source_id / "raw" / entry["name"]
            receipt_path = find_receipt(args.evidence_root, entry["name"])
            if valid_bundle(sequence_dir):
                if bag_path.exists():
                    if receipt_path is None:
                        raise RuntimeError(f"cannot release bag without receipt: {bag_path}")
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if receipt.get("output_sha256") != sha256_file(bag_path):
                        raise RuntimeError(f"bag receipt mismatch before release: {bag_path}")
                    unlink_with_retry(bag_path)
                    write_json(
                        sequence_dir / "raw-cleanup-receipt.json",
                        {
                            "schema": "blindassist_crowdbot_raw_cleanup_receipt_r1",
                            "deleted_path": bag_path.as_posix(),
                            "deleted_sha256": receipt["output_sha256"],
                            "recoverable_from": receipt["url"],
                            "derived_bundle": (sequence_dir / "bundle.json").as_posix(),
                        },
                    )
                source_state["sequence_completed"] += 1
                state["sequence_completed"] += 1
                write_json(args.state, state)
                print(json.dumps({"status": "sequence_already_complete", "source_id": source_id, "sequence_id": sequence_id, "completed": state["sequence_completed"]}), flush=True)
                continue
            if receipt_path is None:
                receipt_path = args.evidence_root / f"{source_id}_{sequence_id}_bag-receipt-r1.json"
            if not bag_path.exists():
                if receipt_path.exists():
                    raise RuntimeError(f"receipt exists but bag and derived bundle are missing: {receipt_path}")
                bag_path.parent.mkdir(parents=True, exist_ok=True)
                run_checked(
                    [
                        sys.executable,
                        str(script_root / "stream_remote_zip_entry.py"),
                        "--inventory",
                        str(inventory_path),
                        "--entry",
                        entry["name"],
                        "--output",
                        str(bag_path),
                        "--receipt",
                        str(receipt_path),
                        "--max-compressed-bytes",
                        str(int(entry["compressed_size"])),
                        "--max-uncompressed-bytes",
                        str(int(entry["uncompressed_size"])),
                    ]
                    + (
                        ["--range-workers", str(args.range_workers)]
                        if args.range_workers is not None
                        else []
                    )
                    + (
                        ["--range-parts", str(args.range_parts)]
                        if args.range_parts is not None
                        else []
                    )
                    + (
                        ["--compressed-cache-root", str(args.compressed_cache_root)]
                        if args.compressed_cache_root is not None
                        else []
                    )
                )
            run_checked(
                [
                    sys.executable,
                    str(script_root / "materialize_crowdbot_rgbd_sequence.py"),
                    "--bag",
                    str(bag_path),
                    "--bag-receipt",
                    str(receipt_path),
                    "--source-id",
                    source_id,
                    "--sequence-id",
                    sequence_id,
                    "--output-dir",
                    str(sequence_dir),
                ]
            )
            if not valid_bundle(sequence_dir):
                raise RuntimeError(f"derived sequence failed validation: {sequence_dir}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            unlink_with_retry(bag_path)
            write_json(
                sequence_dir / "raw-cleanup-receipt.json",
                {
                    "schema": "blindassist_crowdbot_raw_cleanup_receipt_r1",
                    "deleted_path": bag_path.as_posix(),
                    "deleted_sha256": receipt["output_sha256"],
                    "recoverable_from": receipt["url"],
                    "derived_bundle": (sequence_dir / "bundle.json").as_posix(),
                },
            )
            source_state["sequence_completed"] += 1
            state["sequence_completed"] += 1
            write_json(args.state, state)
            print(json.dumps({"status": "sequence_complete", "source_id": source_id, "sequence_id": sequence_id, "completed": state["sequence_completed"]}), flush=True)
    state["status"] = "complete"
    write_json(args.state, state)
    print(json.dumps({"status": "complete", "sequence_completed": state["sequence_completed"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
