#!/usr/bin/env python3
"""Download a small, explicit Wikimedia Commons candidate-video expansion.

The catalog is intentionally narrow and auditable: each item is a public
Commons file with a source page, author, and declared Creative Commons license.
Downloaded bytes stay under F:\\ba-data and are appended to the candidate-mining
project index only after a byte hash has been computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.research.candidate_event_mining.init_project_index import run as append_index
from scripts.research.candidate_event_mining.pipeline import ContractError, write_json


PROJECT_ROOT = Path(r"F:\ba-data\blindassist-candidate-event-mining")
INDEX_PATH = PROJECT_ROOT / "project_index.json"
MEDIA_ROOT = PROJECT_ROOT / "media"
RECORD_ROOT = PROJECT_ROOT / "source-records"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BlindAssistCandidateEventMining/0.1 (research; local reproducible intake)"


CATALOG: tuple[dict[str, str], ...] = (
    {
        "source_id": "wikimedia_commons_boston_crowd_walk_2017",
        "session_id": "cem-r0-20260802-s005-boston",
        "title": "Walking through the crowd in Boston 8 19 17.webm",
        "file_title": "File:Walking through the crowd in Boston 8 19 17.webm",
        "source_url": "https://commons.wikimedia.org/wiki/File:Walking_through_the_crowd_in_Boston_8_19_17.webm",
        "author": "DaTechGuyBlog",
        "license": "Creative Commons Attribution 3.0 Unported",
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
    },
    {
        "source_id": "wikimedia_commons_man_descending_staircase_2015",
        "session_id": "cem-r0-20260802-s006-staircase",
        "title": "Man descending a staircase.webm",
        "file_title": "File:Man descending a staircase.webm",
        "source_url": "https://commons.wikimedia.org/wiki/File:Man_descending_a_staircase.webm",
        "author": "Specialpaguni",
        "license": "Creative Commons Attribution-ShareAlike 4.0 International",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    },
    {
        "source_id": "wikimedia_commons_walking_in_sands_2018",
        "session_id": "cem-r0-20260802-s007-sands",
        "title": "Walking in the sands.webm",
        "file_title": "File:Walking in the sands.webm",
        "source_url": "https://commons.wikimedia.org/wiki/File:Walking_in_the_sands.webm",
        "author": "sgu18ify",
        "license": "Creative Commons Attribution 3.0 Unported",
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _api_file_info(file_title: str, timeout: float) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|extmetadata",
            "format": "json",
            "formatversion": "2",
        }
    )
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list) or len(pages) != 1:
        raise ContractError(f"Commons API returned no unique page: {file_title}")
    page = pages[0]
    imageinfo = page.get("imageinfo")
    if not isinstance(imageinfo, list) or len(imageinfo) != 1:
        raise ContractError(f"Commons API returned no imageinfo: {file_title}")
    info = imageinfo[0]
    if not isinstance(info, dict) or not info.get("url"):
        raise ContractError(f"Commons API returned no direct media URL: {file_title}")
    return info


def _download(url: str, target: Path, timeout: float) -> tuple[str, int]:
    if target.exists():
        raise ContractError(f"refusing to overwrite existing media: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".part", dir=target.parent
            )
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(descriptor, "wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
        return digest.hexdigest(), size
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def _source_record(
    item: dict[str, str],
    info: dict[str, Any],
    media_path: Path,
    content_sha256: str,
    byte_count: int,
    retrieved_at_utc: str,
) -> dict[str, Any]:
    return {
        "author": item["author"],
        "candidate_only": True,
        "confirmation": False,
        "content_sha256": content_sha256,
        "data_role": "THESIS_DEVELOPMENT_CONSUMED_DISCOVERY",
        "event_truth": False,
        "fresh_holdout": False,
        "license": item["license"],
        "license_url": item["license_url"],
        "materialization": "direct byte-verified download from the Wikimedia Commons original file URL",
        "media_path": str(media_path.resolve()),
        "media_source_url": str(info["url"]),
        "media_source_sha1": str(info.get("sha1", "")),
        "media_declared_bytes": int(info.get("size", byte_count)),
        "media_downloaded_bytes": byte_count,
        "original_file_title": item["file_title"],
        "production": False,
        "retrieval_status": "verified",
        "retrieved_at_utc": retrieved_at_utc,
        "session_id": item["session_id"],
        "source_id": item["source_id"],
        "source_platform": "wikimedia_commons",
        "source_url": item["source_url"],
        "title": item["title"],
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = set(args.source_id or [item["source_id"] for item in CATALOG])
    known = {item["source_id"] for item in CATALOG}
    unknown = selected - known
    if unknown:
        raise ContractError(f"unknown catalog source_id: {sorted(unknown)}")
    output: list[dict[str, Any]] = []
    for item in CATALOG:
        if item["source_id"] not in selected:
            continue
        info = _api_file_info(item["file_title"], args.timeout)
        media_path = MEDIA_ROOT / item["source_id"] / "source.webm"
        retrieved_at_utc = datetime.now(timezone.utc).isoformat()
        content_sha256, byte_count = _download(str(info["url"]), media_path, args.timeout)
        record = _source_record(
            item,
            info,
            media_path,
            content_sha256,
            byte_count,
            retrieved_at_utc,
        )
        record_path = RECORD_ROOT / f"{item['source_id']}.json"
        write_json(record_path, record)
        append_index(
            argparse.Namespace(
                output=INDEX_PATH,
                data_root=r"F:\ba-data",
                source_record=record_path,
            )
        )
        output.append(record)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    try:
        records = run(parse_args())
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "source_count": len(records),
                "sources": [
                    {
                        "source_id": record["source_id"],
                        "media_path": record["media_path"],
                        "content_sha256": record["content_sha256"],
                        "bytes": record["media_downloaded_bytes"],
                    }
                    for record in records
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
