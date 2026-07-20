#!/usr/bin/env python3
"""Create an auditable, bounded candidate ledger from Vimeo's CC-BY search.

The command reads exactly one result page per online invocation. Search-page
membership is only a discovery hint: every selected video still needs an
item-level license check, continuity review, and the normal source registry
gate before download or training use.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


SCHEMA = "blindassist_vimeo_ccby_candidate_ledger_v1"
BASE_URL = "https://vimeo.com/creativecommons/by"
USER_AGENT = "BlindAssist-public-candidate-audit/1.0"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
POSITIVE_TITLE_TERMS = (
    "walking",
    "walk",
    "walkthrough",
    "first person",
    "pov",
    "pedestrian",
    "sidewalk",
    "street",
    "roadwork",
    "construction",
    "site visit",
)
NEGATIVE_TITLE_TERMS = (
    "aerial",
    "drone",
    "timelapse",
    "time lapse",
    "montage",
    "promo",
    "showreel",
    "sold",
    "real estate",
    "construction update",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def reject_independent_direction(path: Path) -> None:
    if "secondary-corridor-causal" in str(path).lower().replace("_", "-"):
        raise ValueError("independent-direction paths are forbidden")


def build_search_url(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("search query must not be empty")
    return f"{BASE_URL}?{urlencode({'search': query}, quote_via=quote)}"


class VimeoSearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.candidates: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {key: value for key, value in attrs}
        if tag == "li" and (values.get("id") or "").startswith("clip_"):
            clip_id = (values["id"] or "").removeprefix("clip_")
            if clip_id.isdigit():
                position = values.get("data-position")
                self._current = {
                    "video_id": clip_id,
                    "search_position": int(position) if position and position.isdigit() else None,
                }
            return
        if tag != "a" or self._current is None or "source_title" in self._current:
            return
        href = values.get("href") or ""
        title = values.get("title")
        if title and _href_video_id(href) == self._current["video_id"]:
            self._current["source_title"] = html.unescape(title).strip()
            self._current["source_page_url"] = urljoin("https://vimeo.com/", href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._current is not None:
            if self._current.get("source_title"):
                self.candidates.append(self._current)
            self._current = None


def _href_video_id(href: str) -> str | None:
    parsed = urlparse(urljoin("https://vimeo.com/", href))
    if parsed.netloc.lower() not in {"vimeo.com", "www.vimeo.com"}:
        return None
    first_segment = parsed.path.strip("/").split("/", 1)[0]
    return first_segment if first_segment.isdigit() else None


def parse_search_html(payload: str) -> list[dict[str, Any]]:
    parser = VimeoSearchParser()
    parser.feed(payload)
    parser.close()
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in parser.candidates:
        video_id = str(candidate["video_id"])
        if video_id not in seen:
            unique.append(candidate)
            seen.add(video_id)
    return unique


def title_priority(title: str, query: str) -> dict[str, Any]:
    lowered = title.casefold()
    positive_hits = [term for term in POSITIVE_TITLE_TERMS if term in lowered]
    negative_hits = [term for term in NEGATIVE_TITLE_TERMS if term in lowered]
    query_hits = sorted({
        token.casefold()
        for token in query.split()
        if len(token) >= 3 and token.casefold() in lowered
    })
    score = 2 * len(positive_hits) + len(query_hits) - 2 * len(negative_hits)
    return {
        "title_priority_score": score,
        "positive_title_hits": positive_hits,
        "negative_title_hits": negative_hits,
        "query_title_hits": query_hits,
    }


def fetch_one_page(url: str, *, timeout_seconds: float) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout_seconds) as response:
        final_url = response.geturl()
        if urlparse(final_url).netloc.lower() not in {"vimeo.com", "www.vimeo.com"}:
            raise RuntimeError("Vimeo search redirected to an unexpected host")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise RuntimeError("Vimeo search response exceeded the frozen size limit")
        charset = response.headers.get_content_charset() or "utf-8"
    return payload, charset


def build_report(
    *,
    query: str,
    search_url: str,
    html_payload: bytes,
    charset: str,
    retrieval_mode: str,
    max_results: int,
) -> dict[str, Any]:
    decoded = html_payload.decode(charset, errors="replace")
    candidates = parse_search_html(decoded)[:max_results]
    for candidate in candidates:
        candidate.update(title_priority(str(candidate["source_title"]), query))
        candidate.update({
            "source_platform": "vimeo",
            "discovery_license_filter": "CC BY",
            "item_license_status": "unverified_requires_item_level_attestation",
            "continuity_status": "unreviewed",
            "training_eligible": False,
        })
    candidates.sort(key=lambda item: (
        -int(item["title_priority_score"]),
        int(item["search_position"] or 1_000_000),
    ))
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "search_url": search_url,
        "retrieval_mode": retrieval_mode,
        "request_page_count": 1 if retrieval_mode.startswith("online_single_page") else 0,
        "response_sha256": sha256_bytes(html_payload),
        "parsed_candidate_count": len(candidates),
        "candidates": candidates,
        "license_gate": "Search-page membership is not item-level proof. Re-check the selected video's license metadata before download.",
        "evidence_limit": "This ledger is acquisition triage only. It is not a source registry, an event label, human truth, calibration evidence, blind evidence, production evidence, or permission to train.",
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    reject_independent_direction(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output_path.write_bytes(encoded)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{sha256_bytes(encoded)}\n", encoding="ascii"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--html",
        type=Path,
        help="Use a saved HTML page instead of making the one allowed online request.",
    )
    parser.add_argument(
        "--html-retrieval-mode",
        choices=("offline_saved_html", "online_single_page_external_fetch"),
        default="offline_saved_html",
        help="Declare whether --html came from an offline fixture or this run's single external fetch.",
    )
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.max_results <= 50:
        raise ValueError("max results must be in [1, 50]")
    search_url = build_search_url(args.query)
    if args.html is not None:
        reject_independent_direction(args.html)
        payload = args.html.read_bytes()
        charset = "utf-8"
        retrieval_mode = args.html_retrieval_mode
    else:
        payload, charset = fetch_one_page(
            search_url, timeout_seconds=args.timeout_seconds
        )
        retrieval_mode = "online_single_page"
    report = build_report(
        query=args.query,
        search_url=search_url,
        html_payload=payload,
        charset=charset,
        retrieval_mode=retrieval_mode,
        max_results=args.max_results,
    )
    write_report(report, args.output)
    print(json.dumps({
        "output": str(args.output),
        "candidate_count": report["parsed_candidate_count"],
        "response_sha256": report["response_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
