#!/usr/bin/env python3
"""Build video-document frequency for individual RoadTextVQA OCR tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens(value: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z0-9]+", folded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--ocr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{output}")
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    video_stems = {Path(row["video"]).stem for row in annotations["data"]}
    document_tokens = {stem: set() for stem in video_stems}
    file_counts = Counter()
    with zipfile.ZipFile(args.ocr) as archive:
        for name in archive.namelist():
            match = re.search(r"/([^/]+)_([0-9]+)\.json$", name)
            if not match or match.group(1) not in video_stems:
                continue
            stem = match.group(1)
            payload = json.loads(archive.read(name))
            for row in (payload.get("textAnnotations") or [])[1:]:
                document_tokens[stem].update(tokens(str(row.get("description", ""))))
            file_counts[stem] += 1
    if set(file_counts.values()) != {10} or set(file_counts) != video_stems:
        missing = sorted(video_stems - set(file_counts))
        raise RuntimeError(f"INCOMPLETE_TRAIN_OCR:counts={dict(Counter(file_counts.values()))}:missing={missing[:10]}")
    df = Counter(token for values in document_tokens.values() for token in values)
    result = {
        "schema": "blindassist-l10-roadtextvqa-train-token-df-v1",
        "authority": "TRAIN_SPLIT_BACKGROUND_OCR_STATISTICS_ONLY",
        "inputs": {
            "annotations": {"path": str(args.annotations.resolve()), "bytes": args.annotations.stat().st_size, "sha256": sha256(args.annotations)},
            "ocr": {"path": str(args.ocr.resolve()), "bytes": args.ocr.stat().st_size, "sha256": sha256(args.ocr)},
        },
        "train_videos": len(video_stems),
        "sampled_ocr_files": sum(file_counts.values()),
        "vocabulary_size": len(df),
        "document_frequency": dict(sorted(df.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("train_videos", "sampled_ocr_files", "vocabulary_size")}, indent=2))


if __name__ == "__main__":
    main()
