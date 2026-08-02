#!/usr/bin/env python3
"""Record a credential-free Project Aria Explorer access probe.

The Explorer shell may be public while its dataset metadata, previews, and
download-link APIs require an API key.  This probe records that boundary
without submitting credentials or downloading source media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from pipeline import canonical_sha256, utc_now, write_json


DATASETS: dict[str, dict[str, Any]] = {
    "Project-Aria-Everyday-Activities": {
        "version": "aea",
        "license": "DATASET_TERMS_TO_VERIFY",
        "explorer_url": "https://explorer.projectaria.com/aea/loc3_script2_seq3_rec1",
    },
    "Project-Aria-Everyday-Objects": {
        "version": "aeo",
        "license": "PROJECT_ARIA_DATASET_LICENSE_NONCOMMERCIAL",
        "explorer_url": "https://explorer.projectaria.com/aeo/aeo_seq17_1615689348857273",
    },
}


def _probe_url(
    url: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hftf-d7-projectaria-access-probe/1.0"},
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_s) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
            return {
                "url": url,
                "http_status": int(response.status),
                "content_type": content_type,
                "bytes_read": len(body),
                "body_sha256": hashlib.sha256(body).hexdigest(),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return {
            "url": url,
            "http_status": int(exc.code),
            "content_type": exc.headers.get("Content-Type", "") if exc.headers else "",
            "bytes_read": len(body),
            "body_sha256": hashlib.sha256(body).hexdigest(),
        }
    except (OSError, urllib.error.URLError) as exc:
        return {
            "url": url,
            "http_status": None,
            "content_type": "",
            "bytes_read": 0,
            "body_sha256": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def build_probe(
    dataset_id: str,
    *,
    api_base: str = "https://dtc.projectaria.com",
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    if dataset_id not in DATASETS:
        raise ValueError(f"unsupported Project Aria dataset: {dataset_id}")
    item = DATASETS[dataset_id]
    version = str(item["version"])
    api_root = api_base.rstrip("/")
    urls = [
        str(item["explorer_url"]),
        f"{api_root}/version_config/{version}",
        f"{api_root}/data/{version}",
        f"{api_root}/data/{version}/previews",
        f"{api_root}/data/{version}/download_links",
    ]
    probes = [_probe_url(url, opener=opener, timeout_s=timeout_s) for url in urls]
    api_statuses = [
        row["http_status"]
        for row in probes[1:]
        if row.get("http_status") is not None
    ]
    if api_statuses and all(status == 401 for status in api_statuses):
        access_status = "ACCESS_BLOCKED_API_KEY_REQUIRED"
    elif any(status == 200 for status in api_statuses):
        access_status = "PUBLIC_API_ACCESS_REQUIRES_FURTHER_LAWFUL_TERMS_REVIEW"
    else:
        access_status = "ACCESS_UNRESOLVED_NO_CREDENTIALS_SUBMITTED"
    payload: dict[str, Any] = {
        "schema": "hftf_d7_public_real_projectaria_access_probe_v1",
        "dataset_id": dataset_id,
        "version": version,
        "generated_at_utc": utc_now(),
        "official_url": str(item["explorer_url"]),
        "license_status": str(item["license"]),
        "access_status": access_status,
        "credentials_submitted": False,
        "media_download_attempted": False,
        "event_truth_authority": False,
        "probes": probes,
    }
    payload["probe_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "generated_at_utc"}
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-id", choices=sorted(DATASETS), action="append")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_ids = args.dataset_id or sorted(DATASETS)
    root = Path(args.output_root).resolve()
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        payload = build_probe(dataset_id, timeout_s=args.timeout_s)
        payload["run_id"] = args.run_id
        path = root / "receipts" / f"projectaria_access_probe_{payload['version']}_receipt_{args.run_id}.json"
        write_json(path, payload)
        rows.append({"dataset_id": dataset_id, "path": str(path), "access_status": payload["access_status"]})
    return rows


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
