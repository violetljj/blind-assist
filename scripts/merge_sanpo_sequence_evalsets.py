#!/usr/bin/env python3
"""Merge finalized local SANPO sequence evalsets into one immutable benchmark dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.jsonl").exists():
        raise SystemExit("Refusing to modify an existing canonical manifest")
    rows: list[dict] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    sources: list[dict] = []
    for root in [path.resolve() for path in args.input_root]:
        manifest = root / "manifest.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"Input is not finalized: {root}")
        source_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not source_rows:
            raise SystemExit(f"Empty canonical manifest: {manifest}")
        sources.append({"root": str(root), "manifest_sha256": sha256_file(manifest), "row_count": len(source_rows)})
        for row in source_rows:
            sample_id = str(row["id"])
            digest = str(row.get("source", {}).get("sha256", ""))
            if sample_id in seen_ids or not digest or digest in seen_hashes:
                raise SystemExit(f"Duplicate id or image hash while merging: {sample_id}")
            seen_ids.add(sample_id)
            seen_hashes.add(digest)
            for rel in (Path(row["image_path"]), Path("source_masks/test") / f"{sample_id}.png"):
                source = root / rel
                target = output / rel
                if not source.is_file():
                    raise SystemExit(f"Missing finalized asset: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            rows.append(row)
    rows.sort(key=lambda row: (str(row["sequence_id"]), int(row["frame_index"])))
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    spec = {
        "name": "blindassist_sanpo_v2_public_sequence_evalset",
        "task": "continuous-sequence traversability evaluation",
        "source_type": "SANPO-Real v0 official test split, locally merged from finalized sequence datasets",
        "license": "Creative Commons Attribution 4.0 International",
        "target_fps": 10,
        "sequence_count": len({row["sequence_id"] for row in rows}),
        "frame_count": len(rows),
        "inputs": sources,
        "privacy_policy": "Original RGB frames and masks remain ignored local test artifacts and are never committed to Git.",
    }
    (output / "dataset_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"merged_sequences={spec['sequence_count']} frames={len(rows)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
