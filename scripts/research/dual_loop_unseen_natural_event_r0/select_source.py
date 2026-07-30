from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API = "https://commons.wikimedia.org/w/api.php"
CATEGORY = "Category:First-person videos on foot"
USER_AGENT = "BlindAssistResearch/1.0 (Codex isolated internal evaluation)"
R1_COMMIT = "039757b2da41c051373f8ee3189c4b06028f5295"
TITLE_PATTERN = re.compile(r"(walk|walking|on foot|city tour|tour)", re.IGNORECASE)
USED_EXACT_TITLES = {
    "File:Explore Shanghai's Iconic Street - Shanxi South Street.webm",
    "File:Matoaka walks.webm",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def api_get(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{API}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def normalize_page(page: dict[str, Any]) -> dict[str, Any]:
    video = page["videoinfo"][0]
    metadata = video.get("extmetadata", {})

    def meta(name: str) -> str | None:
        item = metadata.get(name)
        return item.get("value") if isinstance(item, dict) else None

    return {
        "pageid": int(page["pageid"]),
        "title": page["title"],
        "canonical_description_url": video.get("descriptionurl"),
        "original_url": video["url"],
        "original_sha1_base36": video["sha1"],
        "size_bytes": int(video["size"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "duration_s": float(video["duration"]),
        "mime": video["mime"],
        "mediatype": video["mediatype"],
        "license_short_name": meta("LicenseShortName"),
        "artist": meta("Artist"),
        "date_time_original": meta("DateTimeOriginal"),
    }


def eligible_reason(row: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if row["mediatype"] != "VIDEO" or row["mime"] != "video/webm":
        failures.append("NOT_WEBM_VIDEO")
    if not (180.0 <= row["duration_s"] <= 900.0):
        failures.append("DURATION_OUTSIDE_180_TO_900_S")
    if row["width"] < 1280 or row["height"] < 720:
        failures.append("RESOLUTION_BELOW_1280X720")
    if row["size_bytes"] > 800_000_000:
        failures.append("ORIGINAL_ABOVE_800MB")
    if not str(row.get("license_short_name") or "").startswith("CC"):
        failures.append("LICENSE_NOT_CC")
    if not TITLE_PATTERN.search(row["title"]):
        failures.append("TITLE_NOT_WALK_OR_TOUR")
    if row["title"] in USED_EXACT_TITLES:
        failures.append("EXACT_SOURCE_ALREADY_USED")
    return not failures, failures


def fetch_registry() -> list[dict[str, Any]]:
    payload = api_get(
        {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": CATEGORY,
            "gcmtype": "file",
            "gcmlimit": "500",
            "prop": "videoinfo",
            "viprop": "url|size|mime|mediatype|extmetadata|sha1",
            "format": "json",
            "formatversion": "2",
        }
    )
    if payload.get("continue"):
        raise RuntimeError("category pagination changed; selector refuses partial registry")
    pages = payload.get("query", {}).get("pages", [])
    return sorted((normalize_page(page) for page in pages), key=lambda row: row["title"])


def fetch_derivatives(title: str) -> list[dict[str, Any]]:
    payload = api_get(
        {
            "action": "query",
            "titles": title,
            "prop": "videoinfo",
            "viprop": "derivatives",
            "format": "json",
            "formatversion": "2",
        }
    )
    video = payload["query"]["pages"][0]["videoinfo"][0]
    result: list[dict[str, Any]] = []
    for item in video.get("derivatives", []):
        if item.get("transcodekey") in {"480p.vp9.webm", "720p.vp9.webm"}:
            result.append(
                {
                    "transcode_key": item["transcodekey"],
                    "width": int(item["width"]),
                    "height": int(item["height"]),
                    "url": item["src"],
                    "type": item.get("type"),
                }
            )
    return sorted(result, key=lambda item: (item["height"], item["transcode_key"]))


def select(registry: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for row in registry:
        admitted, failures = eligible_reason(row)
        audited_row = {**row, "eligible": admitted, "exclusion_reasons": failures}
        audited.append(audited_row)
        if admitted:
            eligible.append(row)
    if not eligible:
        raise RuntimeError("no eligible source in frozen registry")
    selected = dict(eligible[0])
    selected["derivatives"] = fetch_derivatives(selected["title"])
    if not any(item["transcode_key"] == "480p.vp9.webm" for item in selected["derivatives"]):
        raise RuntimeError("rank-1 source has no 480p VP9 derivative")
    return audited, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    registry = fetch_registry()
    audited, selected = select(registry)
    registry_payload = {
        "schema_version": 1,
        "api": API,
        "category": CATEGORY,
        "selection_stage": "METADATA_ONLY_PRE_PAYLOAD_PRE_OUTCOME",
        "pages": audited,
    }
    registry_bytes = canonical_json_bytes(registry_payload)
    registry_path = args.output_dir / "source_registry.json"
    registry_path.write_bytes(registry_bytes)

    eligible_titles = [row["title"] for row in audited if row["eligible"]]
    receipt = {
        "schema_version": 1,
        "protocol_id": "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0",
        "status": "SOURCE_SELECTED_BEFORE_PAYLOAD_AND_OUTPUT_ACCESS",
        "r1_commit": R1_COMMIT,
        "source_registry_sha256": sha256_bytes(registry_bytes),
        "source_count": len(audited),
        "eligibility": {
            "category": CATEGORY,
            "mime": "video/webm",
            "mediatype": "VIDEO",
            "duration_s_inclusive": [180.0, 900.0],
            "minimum_resolution": [1280, 720],
            "maximum_original_size_bytes": 800_000_000,
            "license_prefix": "CC",
            "title_regex": TITLE_PATTERN.pattern,
            "used_exact_titles_excluded": sorted(USED_EXACT_TITLES),
            "ordering": "UNICODE_TITLE_ASCENDING",
        },
        "eligible_count": len(eligible_titles),
        "eligible_titles_in_order": eligible_titles,
        "selected_rank": 1,
        "selected": selected,
        "outcome_access": {
            "video_payload_opened": False,
            "truth_ledger_opened": False,
            "baseline_output_opened": False,
            "candidate_output_opened": False,
        },
    }
    (args.output_dir / "source_selection_receipt.json").write_bytes(
        canonical_json_bytes(receipt)
    )
    sys.stdout.buffer.write(
        (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
