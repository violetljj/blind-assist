"""Audit DSText V2 public media and annotation authority for L10.

This is a source-admission canary, not an OCR or controller evaluation.  It
inspects the official Zenodo record, one optional local ZIP sample, and the
official RRC downloads page.  Missing annotation authority terminates the
canary before any L10 replay or matcher change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import certifi

ZENODO_RECORD_URL = "https://zenodo.org/api/records/10010507"
RRC_DOWNLOADS_URL = "https://rrc.cvc.uab.es/?ch=30&com=downloads"
USER_AGENT = "BlindAssist-DSText-source-canary/1.0"
ANNOTATION_SUFFIXES = {".csv", ".json", ".jsonl", ".mat", ".txt", ".xml"}
MEDIA_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def fetch_bytes(url: str, timeout_s: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout_s, context=context) as response:
        return response.read()


def fetch_json(url: str) -> dict[str, Any]:
    value = json.loads(fetch_bytes(url).decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object from {url}")
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_archive(path: Path) -> dict[str, Any]:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
    media = [item for item in entries if Path(item.filename).suffix.lower() in MEDIA_SUFFIXES]
    annotations = [
        item for item in entries if Path(item.filename).suffix.lower() in ANNOTATION_SUFFIXES
    ]
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "entry_count": len(entries),
        "uncompressed_bytes": sum(item.file_size for item in entries),
        "media_entries": [item.filename for item in media],
        "annotation_entries": [item.filename for item in annotations],
    }


def run(sample_archive: Path | None, rrc_html_path: Path | None) -> dict[str, Any]:
    record = fetch_json(ZENODO_RECORD_URL)
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Zenodo record has no files")

    public_files = []
    for item in files:
        if not isinstance(item, dict):
            continue
        public_files.append(
            {
                "key": str(item.get("key", "")),
                "size_bytes": int(item.get("size", 0)),
                "checksum": item.get("checksum"),
                "download_url": (item.get("links") or {}).get("self"),
            }
        )

    zenodo_annotation_files = [
        item["key"]
        for item in public_files
        if Path(item["key"]).suffix.lower() in ANNOTATION_SUFFIXES
    ]
    archive_audit = audit_archive(sample_archive) if sample_archive else None

    if rrc_html_path:
        rrc_payload = rrc_html_path.read_bytes()
    else:
        rrc_payload = fetch_bytes(RRC_DOWNLOADS_URL)
    rrc_html = rrc_payload.decode("utf-8", errors="replace")
    rrc_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", rrc_html)).strip()
    rrc_requires_registration = bool(
        re.search(r"need to register to get access", rrc_text, flags=re.IGNORECASE)
    )
    rrc_lists_dstext = bool(re.search(r"DSText\s*2023", rrc_text, flags=re.IGNORECASE))

    sample_has_media = bool(archive_audit and archive_audit["media_entries"])
    sample_has_annotations = bool(archive_audit and archive_audit["annotation_entries"])
    annotation_authority_admitted = bool(
        zenodo_annotation_files or sample_has_annotations
    )

    if annotation_authority_admitted:
        verdict = "DSTEXT_V2_ANNOTATION_SOURCE_ADMITTED"
    else:
        verdict = "DSTEXT_V2_ANNOTATION_AUTHORITY_NOT_ADMITTED"

    return {
        "schema_version": "blindassist-l10-dstext-v2-source-canary-v1",
        "source": {
            "record_url": ZENODO_RECORD_URL,
            "record_id": record.get("id"),
            "doi": (record.get("metadata") or {}).get("doi"),
            "title": (record.get("metadata") or {}).get("title"),
            "license": ((record.get("metadata") or {}).get("license") or {}).get("id"),
            "file_count": len(public_files),
            "total_bytes": sum(item["size_bytes"] for item in public_files),
            "files": public_files,
        },
        "sample_archive": archive_audit,
        "rrc": {
            "downloads_url": RRC_DOWNLOADS_URL,
            "html_path": str(rrc_html_path.resolve()) if rrc_html_path else None,
            "html_sha256": hashlib.sha256(rrc_payload).hexdigest(),
            "lists_dstext_2023": rrc_lists_dstext,
            "requires_registration": rrc_requires_registration,
        },
        "checks": {
            "official_record_accessible": True,
            "public_video_archives_present": bool(public_files),
            "sample_media_readable": sample_has_media if sample_archive else None,
            "zenodo_annotation_files": zenodo_annotation_files,
            "sample_annotation_files": (
                archive_audit["annotation_entries"] if archive_audit else []
            ),
            "annotation_authority_admitted": annotation_authority_admitted,
        },
        "verdict": verdict,
        "decision": (
            "Do not run L10 reacquisition evaluation. The public Zenodo payload "
            "admits media access but not evaluator-authoritative annotations; "
            "official ground truth remains behind RRC registration."
            if not annotation_authority_admitted
            else "Annotation authority is present; a frozen track-gap audit may proceed."
        ),
        "claim_ceiling": (
            "Public media availability only. No track-gap, presence, semantic "
            "reacquisition, active-view, arrival, or handoff conclusion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-archive", type=Path)
    parser.add_argument("--rrc-html", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.sample_archive, args.rrc_html)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
