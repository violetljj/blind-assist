#!/usr/bin/env python3
"""Build a bounded, license-aware Internet Archive video candidate ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlencode


SCHEMA = "blindassist_internet_archive_public_video_candidate_ledger_v1"
API_URL = "https://archive.org/advancedsearch.php"
POSITIVE_TERMS = (
    "walk", "walking", "pedestrian", "first person", "pov", "sidewalk",
    "roadwork", "road work", "construction", "barricade", "detour", "obstruction",
)
NEGATIVE_TERMS = (
    "podcast", "radio", "meeting", "interview", "news", "animation", "gameplay",
    "drone", "aerial", "timelapse", "time lapse", "dashcam", "movie trailer",
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
    quoted = " AND ".join(f'({token})' for token in query.split())
    search = f"mediatype:movies AND licenseurl:* AND ({quoted})"
    fields = ["identifier", "title", "description", "creator", "licenseurl", "subject", "date", "downloads"]
    params: list[tuple[str, str]] = [("q", search)]
    params.extend(("fl[]", field) for field in fields)
    params.extend((("rows", str(limit)), ("page", "1"), ("output", "json"), ("sort[]", "downloads desc")))
    return f"{API_URL}?{urlencode(params)}"


def as_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "")


def score_candidate(document: dict[str, Any], query: str) -> dict[str, Any]:
    combined = " ".join((as_text(document.get("title")), as_text(document.get("description")),
                         as_text(document.get("subject")))).casefold()
    positive = [term for term in POSITIVE_TERMS if term in combined]
    negative = [term for term in NEGATIVE_TERMS if term in combined]
    query_hits = sorted({token.casefold() for token in query.split()
                         if len(token) >= 3 and token.casefold() in combined})
    return {
        "title_priority_score": 2 * len(positive) + len(query_hits) - 3 * len(negative),
        "positive_text_hits": positive,
        "negative_text_hits": negative,
        "query_text_hits": query_hits,
    }


def parse_api_payload(payload: bytes, query: str) -> list[dict[str, Any]]:
    body = json.loads(payload.decode("utf-8"))
    documents = (body.get("response") or {}).get("docs") or []
    rows = []
    for document in documents:
        identifier = as_text(document.get("identifier")).strip()
        license_url = as_text(document.get("licenseurl")).strip()
        if not identifier or not license_url:
            continue
        row = {
            "identifier": identifier,
            "source_title": as_text(document.get("title")).strip(),
            "source_page_url": f"https://archive.org/details/{identifier}",
            "creator": as_text(document.get("creator")).strip() or None,
            "date": as_text(document.get("date")).strip() or None,
            "downloads": int(document.get("downloads") or 0),
            "license_url": license_url,
            "item_license_status": "metadata_present_requires_item_page_verification",
            "downloadable_video_status": "unverified",
            "continuity_status": "unreviewed",
            "training_eligible": False,
        }
        row.update(score_candidate(document, query))
        rows.append(row)
    return sorted(rows, key=lambda row: (-int(row["title_priority_score"]), -int(row["downloads"]), row["identifier"]))


def build_report(contract: dict[str, Any], responses: list[tuple[str, str, bytes]]) -> dict[str, Any]:
    seen: set[str] = set()
    candidates = []
    evidence = []
    for query, url, payload in responses:
        parsed = parse_api_payload(payload, query)
        evidence.append({"query": query, "request_url": url, "response_sha256": sha256_bytes(payload),
                         "parsed_licensed_item_count": len(parsed)})
        for row in parsed:
            if row["identifier"] in seen:
                continue
            row["discovery_query"] = query
            candidates.append(row)
            seen.add(row["identifier"])
    candidates.sort(key=lambda row: (-int(row["title_priority_score"]), -int(row["downloads"]), row["identifier"]))
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract["contract_id"],
        "request_count": len(responses),
        "pagination_used": False,
        "responses": evidence,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "license_gate": "Verify the item page, exact license, and downloadable video file before download.",
        "evidence_limit": "Discovery only; no candidate is event truth, training data, calibration, blind evidence, Android authorization, or production evidence.",
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    reject_independent_direction(output)
    if output.exists() or Path(str(output) + ".sha256").exists():
        raise ValueError("refusing to overwrite Internet Archive discovery ledger")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    Path(str(output) + ".sha256").write_text(sha256_bytes(encoded) + "\n", encoding="ascii")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True,
                        help="Directory containing exactly one 00.json, 01.json, ... response per frozen query.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.contract, args.responses_dir, args.output):
        reject_independent_direction(path)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    queries = list(contract["queries"])
    limit = int(contract["request_limits"]["maximum_results_per_query"])
    responses = [(query, api_query_url(query, limit), (args.responses_dir / f"{index:02d}.json").read_bytes())
                 for index, query in enumerate(queries)]
    report = build_report(contract, responses)
    write_report(report, args.output)
    print(json.dumps({"candidate_count": report["candidate_count"], "request_count": report["request_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
