#!/usr/bin/env python3
"""Create a bounded license-aware video candidate ledger from Wikimedia Commons."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


SCHEMA = "blindassist_wikimedia_public_video_candidate_ledger_v1"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BlindAssist-public-directional-obstruction-audit/1.0"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
POSITIVE_TERMS = (
    "walk", "walking", "pedestrian", "pov", "first person", "sidewalk",
    "roadwork", "road work", "construction", "barricade", "detour", "obstruction",
)
NEGATIVE_TERMS = (
    "drone", "aerial", "timelapse", "time lapse", "montage", "animation",
    "interview", "speech", "fixed camera", "dashcam", "train cab", "bus ride",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def reject_independent_direction(path: Path) -> None:
    if "secondary-corridor-causal" in str(path).lower().replace("_", "-"):
        raise ValueError("independent-direction paths are forbidden")


def api_query_url(query: str, limit: int) -> str:
    query = query.strip()
    if not query:
        raise ValueError("search query must not be empty")
    parameters = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
    }
    return f"{API_URL}?{urlencode(parameters)}"


def fetch_one(url: str, timeout_seconds: float) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=timeout_seconds) as response:
        final = urlparse(response.geturl())
        if final.netloc.lower() != "commons.wikimedia.org":
            raise RuntimeError("Commons API redirected to an unexpected host")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Commons API response exceeded the frozen size limit")
    return payload


def metadata_value(metadata: dict[str, Any], name: str) -> str | None:
    row = metadata.get(name)
    if not isinstance(row, dict):
        return None
    value = row.get("value")
    return str(value).strip() if value is not None and str(value).strip() else None


def title_priority(title: str, description: str, query: str) -> dict[str, Any]:
    combined = f"{title} {description}".casefold()
    positive_hits = [term for term in POSITIVE_TERMS if term in combined]
    negative_hits = [term for term in NEGATIVE_TERMS if term in combined]
    query_hits = sorted({token.casefold() for token in query.split()
                         if len(token) >= 3 and token.casefold() in combined})
    return {
        "title_priority_score": 2 * len(positive_hits) + len(query_hits) - 3 * len(negative_hits),
        "positive_text_hits": positive_hits,
        "negative_text_hits": negative_hits,
        "query_text_hits": query_hits,
    }


def parse_api_payload(payload: bytes, query: str) -> list[dict[str, Any]]:
    document = json.loads(payload.decode("utf-8"))
    pages = (document.get("query") or {}).get("pages") or []
    candidates = []
    for page in pages:
        info_rows = page.get("imageinfo") or []
        if len(info_rows) != 1:
            continue
        info = info_rows[0]
        mime = str(info.get("mime") or "")
        if not mime.startswith("video/"):
            continue
        metadata = info.get("extmetadata") or {}
        description = metadata_value(metadata, "ImageDescription") or ""
        license_short = metadata_value(metadata, "LicenseShortName")
        license_url = metadata_value(metadata, "LicenseUrl")
        title = str(page.get("title") or "")
        row = {
            "page_id": int(page["pageid"]),
            "source_title": title,
            "source_page_url": str(info.get("descriptionurl") or ""),
            "direct_media_url": str(info.get("url") or ""),
            "mime": mime,
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
            "duration_seconds": float(info.get("duration") or 0.0),
            "author": metadata_value(metadata, "Artist"),
            "license_short_name": license_short,
            "license_url": license_url,
            "usage_terms": metadata_value(metadata, "UsageTerms"),
            "item_license_metadata_present": bool(license_short and license_url),
            "continuity_status": "unreviewed",
            "pedestrian_view_status": "unreviewed",
            "training_eligible": False,
        }
        row.update(title_priority(title, description, query))
        candidates.append(row)
    return sorted(candidates, key=lambda row: (-int(row["title_priority_score"]), row["source_title"]))


def build_report(contract: dict[str, Any], responses: list[tuple[str, str, bytes]]) -> dict[str, Any]:
    seen: set[int] = set()
    candidates = []
    evidence = []
    for query, url, payload in responses:
        parsed = parse_api_payload(payload, query)
        evidence.append({"query": query, "request_url": url, "response_sha256": sha256_bytes(payload),
                         "parsed_video_count": len(parsed)})
        for row in parsed:
            if row["page_id"] in seen:
                continue
            row["discovery_query"] = query
            candidates.append(row)
            seen.add(row["page_id"])
    candidates.sort(key=lambda row: (-int(row["title_priority_score"]), row["source_title"]))
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract["contract_id"],
        "request_count": len(responses),
        "pagination_used": False,
        "responses": evidence,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "download_gate": "An ordinary public file URL is sufficient for isolated internal research; preserve item license metadata when available without blocking download.",
        "evidence_limit": "Discovery ledger only; no row is event truth, a training label, calibration, blind evidence, or production authorization.",
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    reject_independent_direction(output)
    if output.exists() or Path(str(output) + ".sha256").exists():
        raise ValueError("refusing to overwrite Commons discovery ledger")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    Path(str(output) + ".sha256").write_text(sha256_bytes(encoded) + "\n", encoding="ascii")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--responses-dir", type=Path,
                        help="Parse 00.json, 01.json, ... fetched once by the Windows TLS wrapper.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reject_independent_direction(args.contract)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    queries = list(contract["queries"]["wikimedia_commons"])
    limit = int(contract["request_limits"]["maximum_results_per_query"])
    responses = []
    for index, query in enumerate(queries):
        url = api_query_url(query, limit)
        if args.responses_dir is None:
            payload = fetch_one(url, args.timeout_seconds)
        else:
            reject_independent_direction(args.responses_dir)
            payload = (args.responses_dir / f"{index:02d}.json").read_bytes()
        responses.append((query, url, payload))
    report = build_report(contract, responses)
    write_report(report, args.output)
    print(json.dumps({"candidate_count": report["candidate_count"], "request_count": report["request_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
