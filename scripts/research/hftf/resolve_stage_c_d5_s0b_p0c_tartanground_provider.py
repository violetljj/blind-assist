#!/usr/bin/env python3
"""Resolve the locked TartanGround catalog to a pinned official provider.

This is a repairable Development utility, not a one-shot protocol. Network or
filesystem failures do not consume a cohort; rerun the same command after
repairing the infrastructure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO_ID = "theairlabcmu/TartanGround"
HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets"
DEFAULT_CATALOG = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0a1-tartanground-catalog-20260802/catalog.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0b-p0c-provider-resolution-20260802"
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "BlindAssist-HFTF/Development"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def resolve_url(revision: str, path: str) -> str:
    encoded_path = quote(path, safe="/")
    return f"{HF_RESOLVE}/{REPO_ID}/resolve/{revision}/{encoded_path}"


def build_resolution(
    catalog: dict[str, Any],
    provider: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observation = catalog["catalog_observation"]
    parents = observation["parents"]
    revision = provider["sha"]
    provider_paths = {row["rfilename"] for row in provider["siblings"]}

    mapping_parents = []
    catalog_paths: set[str] = set()
    for parent in parents:
        archive_urls = {}
        for archive_name, archive_path in sorted(parent["archive_paths"].items()):
            catalog_paths.add(archive_path)
            archive_urls[archive_name] = {
                "path": archive_path,
                "url": resolve_url(revision, archive_path),
            }
        mapping_parents.append(
            {
                "environment": parent["environment"],
                "parent_id": parent["parent_id"],
                "robot_version": parent["robot_version"],
                "trajectory_id": parent["trajectory_id"],
                "archive_urls": archive_urls,
            }
        )

    missing = sorted(catalog_paths - provider_paths)
    mapping = {
        "schema": "blindassist_hftf_tartanground_pinned_archive_url_map",
        "provider": {
            "kind": "huggingface_dataset",
            "repo_id": REPO_ID,
            "revision": revision,
            "resolve_url_template": (
                f"{HF_RESOLVE}/{REPO_ID}/resolve/{revision}/{{archive_path}}"
            ),
        },
        "parents": mapping_parents,
    }
    mapping_payload = canonical_bytes(mapping)
    result = {
        "schema": "blindassist_hftf_stage_c_d5_s0b_p0c_provider_resolution",
        "status": (
            "PROVIDER_AND_EXACT_ARCHIVE_MAPPING_RESOLVED"
            if not missing
            else "PROVIDER_CATALOG_PATH_MISMATCH"
        ),
        "development_policy": {
            "repairable_after_engineering_failure": True,
            "one_shot": False,
            "source_burned_by_resolution": False,
        },
        "provider": {
            "kind": "huggingface_dataset",
            "repo_id": REPO_ID,
            "revision": revision,
            "private": provider.get("private"),
            "gated": provider.get("gated"),
            "disabled": provider.get("disabled"),
            "last_modified": provider.get("lastModified"),
            "provider_file_count": len(provider_paths),
        },
        "catalog": {
            "parent_count": len(parents),
            "archive_path_count": len(catalog_paths),
            "all_archive_paths_present_at_revision": not missing,
            "missing_archive_paths": missing,
        },
        "archive_url_map": {
            "filename": "archive_url_map.json",
            "sha256": sha256_bytes(mapping_payload),
        },
        "claim_boundary": {
            "provider_source_and_exact_archive_mapping_established": not missing,
            "archive_payload_structure_established": False,
            "hftf_opportunity_or_effect_established": False,
            "mainline_or_product_claim_authorized": False,
        },
    }
    return mapping, result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--revision",
        help="Pinned provider revision. Defaults to the provider's current revision.",
    )
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    provider_url = f"{HF_API}/{REPO_ID}"
    if args.revision:
        provider_url += f"/revision/{quote(args.revision, safe='')}"
    provider = fetch_json(provider_url)
    mapping, result = build_resolution(catalog, provider)

    write_json(args.output_root / "archive_url_map.json", mapping)
    write_json(args.output_root / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PROVIDER_AND_EXACT_ARCHIVE_MAPPING_RESOLVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
