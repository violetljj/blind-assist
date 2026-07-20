#!/usr/bin/env python3
"""Extract only predeclared RGB members from a quarantined public archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ExtractionError(ValueError):
    """The archive cannot be safely reduced to its RGB-only candidate view."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExtractionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExtractionError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rgb_members(names: list[str], *, prefix: str, expected_count: int) -> list[str]:
    if not prefix.endswith("/") or not isinstance(expected_count, int) or expected_count <= 0:
        raise ExtractionError("invalid RGB-member extraction contract")
    selected = [name for name in names if name.startswith(prefix) and name.lower().endswith(".png")]
    if len(selected) != expected_count:
        raise ExtractionError(f"expected {expected_count} RGB PNG members, found {len(selected)}")
    if any("/labels/" in name.lower() or "/depth/" in name.lower() for name in selected):
        raise ExtractionError("RGB selection leaked annotation or depth members")
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        import py7zr
    except ImportError as error:
        print(json.dumps({"ok": False, "error": "py7zr is required in the isolated tool path"}))
        return 2
    try:
        config = load_json(args.config)
        receipt = load_json(args.candidate_dir / "public_candidate_receipt.json")
        if receipt.get("download_status") != "raw_candidate_pending_deidentification":
            raise ExtractionError("candidate receipt is not a quarantined raw download")
        if receipt.get("training_execution_authorized") is not False:
            raise ExtractionError("candidate receipt unexpectedly authorizes training")
        archive = args.candidate_dir / config["candidate_file_name"]
        if not archive.is_file() or sha256_file(archive) != receipt.get("local_file_sha256"):
            raise ExtractionError("raw archive does not match the source receipt")
        if args.output_dir.exists():
            raise ExtractionError(f"refusing to overwrite extraction output: {args.output_dir}")
        with py7zr.SevenZipFile(archive, mode="r") as package:
            members = select_rgb_members(
                [entry.filename for entry in package.list()],
                prefix=config["rgb_member_prefix"],
                expected_count=config["expected_rgb_frame_count"],
            )
            package.extract(path=args.output_dir, targets=members)
        extracted = [args.output_dir / member for member in members]
        if not all(path.is_file() for path in extracted):
            raise ExtractionError("archive extraction did not produce every contracted RGB member")
        manifest = {
            "format": "blindassist_public_rgb_only_extraction_v1",
            "source_id": config["source_id"],
            "raw_archive_sha256": receipt["local_file_sha256"],
            "member_prefix": config["rgb_member_prefix"],
            "frame_count": len(extracted),
            "relative_rgb_paths": members,
            "privacy_processing_required": True,
            "source_labels_or_geometry_extracted": False,
            "human_event_truth_present": False,
            "training_execution_authorized": False,
            "production_model_replacement_authorized": False,
        }
        (args.output_dir / "rgb_extraction_receipt.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"ok": True, "frame_count": len(extracted), "output": str(args.output_dir.resolve())}, ensure_ascii=False))
    except (ExtractionError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
