#!/usr/bin/env python3
"""Create or verify the immutable 90-frame SANPO v3 regression baseline.

The source and destination are local, Git-ignored test artifacts.  A baseline
is write-once: the command refuses an existing output directory and records
every RGB and source-mask digest, the source manifest digest, the Git revision,
the benchmark configuration digest, and the same-device report reference.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


BASELINE_ID = "sanpo-v3-regression-90f"
LOCK_NAME = "baseline_lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_revision(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def source_mask_path(root: Path, row: dict[str, Any]) -> Path:
    # SANPO v2 manifests store source masks by sample ID rather than a row path.
    explicit = row.get("source_mask_path")
    return (root / str(explicit)).resolve() if explicit else (root / "source_masks" / "test" / f"{row['id']}.png").resolve()


def build_lock(
    source_root: Path,
    rows: list[dict[str, Any]],
    configuration: Path,
    device_report: str,
    project_root: Path,
) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"duplicate or missing sample id: {sample_id}")
        seen_ids.add(sample_id)
        image = (source_root / str(row.get("image_path", ""))).resolve()
        mask = source_mask_path(source_root, row)
        if not image.is_file() or not mask.is_file():
            raise ValueError(f"{sample_id}: missing image or source mask")
        image_digest = sha256_file(image)
        if image_digest in seen_images:
            raise ValueError(f"{sample_id}: duplicate RGB SHA256")
        seen_images.add(image_digest)
        records.append({
            "id": sample_id,
            "image_path": str(Path(row["image_path"]).as_posix()),
            "source_mask_path": str(mask.relative_to(source_root).as_posix()),
            "image_sha256": image_digest,
            "source_mask_sha256": sha256_file(mask),
        })
    if len(records) != 90:
        raise ValueError(f"{BASELINE_ID} must contain exactly 90 frames, got {len(records)}")
    source_manifest = source_root / "manifest.jsonl"
    return {
        "format": "blindassist_sanpo_regression_lock_v1",
        "baseline_id": BASELINE_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source_root": str(source_root),
        "source_manifest_sha256": sha256_file(source_manifest),
        "git_revision": git_revision(project_root),
        "benchmark_configuration": {
            "path": configuration.name,
            "sha256": sha256_file(configuration),
        },
        "same_device_report_reference": device_report,
        "frame_count": len(records),
        "frames": records,
    }


def verify_lock(root: Path) -> list[str]:
    lock_path = root / LOCK_NAME
    if not lock_path.is_file():
        return [f"missing {LOCK_NAME}"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if lock.get("baseline_id") != BASELINE_ID:
        errors.append("unexpected baseline_id")
    frames = lock.get("frames", [])
    if lock.get("frame_count") != 90 or len(frames) != 90:
        errors.append("baseline must lock exactly 90 frames")
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        errors.append("missing frozen manifest.jsonl")
    elif lock.get("frozen_manifest_sha256") != sha256_file(manifest):
        errors.append("frozen manifest SHA256 differs from lock")
    for record in frames:
        sample_id = record.get("id", "<unknown>")
        for path_key, digest_key in (("image_path", "image_sha256"), ("source_mask_path", "source_mask_sha256")):
            candidate = (root / str(record.get(path_key, ""))).resolve()
            if not candidate.is_file():
                errors.append(f"{sample_id}: missing {path_key}")
            elif sha256_file(candidate) != record.get(digest_key):
                errors.append(f"{sample_id}: {path_key} SHA256 differs from lock")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--device-report", help="Existing same-device benchmark report directory or immutable report ID.")
    parser.add_argument("--verify", action="store_true", help="Verify an existing frozen baseline without writing it.")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.verify:
        errors = verify_lock(output)
        print(json.dumps({"ok": not errors, "baseline": str(output), "errors": errors}, ensure_ascii=False))
        return 0 if not errors else 1
    if not args.source_root or not args.benchmark_config or not args.device_report:
        parser.error("--source-root, --benchmark-config and --device-report are required when freezing")
    source = args.source_root.resolve()
    configuration = args.benchmark_config.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable baseline root: {output}")
    if not (source / "manifest.jsonl").is_file() or not configuration.is_file():
        raise SystemExit("source-root must be finalized and benchmark-config must exist")
    rows = load_jsonl(source / "manifest.jsonl")
    project_root = Path(__file__).resolve().parents[1]
    lock = build_lock(source, rows, configuration, args.device_report, project_root)
    output.mkdir(parents=True)
    for directory in ("images", "source_masks"):
        shutil.copytree(source / directory, output / directory)
    shutil.copy2(source / "manifest.jsonl", output / "manifest.jsonl")
    shutil.copy2(configuration, output / configuration.name)
    lock["frozen_manifest_sha256"] = sha256_file(output / "manifest.jsonl")
    (output / LOCK_NAME).write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = verify_lock(output)
    if errors:
        raise SystemExit("freeze verification failed: " + "; ".join(errors))
    print(f"baseline_lock_ok=true baseline={output} frames=90")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
