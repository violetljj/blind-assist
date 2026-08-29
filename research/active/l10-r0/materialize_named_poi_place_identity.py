"""Materialize the metadata-frozen PB2-A place-identity cohort from Commons."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BlindAssist-L10-PB2/1.0 (research prototype)"
REQUEST_GAP_SECONDS = 0.8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open(request: urllib.request.Request, timeout: int):
    for attempt in range(5):
        time.sleep(REQUEST_GAP_SECONDS)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 3.0 * (2**attempt)
            time.sleep(min(delay, 30.0))
    raise RuntimeError("UNREACHABLE_RETRY_STATE")


def _api_json(**parameters: str) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {"action": "query", "format": "json", "formatversion": "2", **parameters}
    )
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with _open(request, timeout=60) as response:
        return json.load(response)


def _commons_info(filename: str) -> dict[str, Any]:
    payload = _api_json(
        prop="imageinfo",
        titles=f"File:{filename}",
        iiprop="url|size|mime|sha1|extmetadata",
        iiurlwidth="1280",
    )
    page = payload["query"]["pages"][0]
    if page.get("missing"):
        raise ValueError(f"COMMONS_FILE_NOT_FOUND:{filename}")
    info = page["imageinfo"][0]
    metadata = info.get("extmetadata", {})
    return {
        "commons_file": page["title"].removeprefix("File:"),
        "source_page": info["descriptionurl"],
        "download_url": info.get("thumburl", info["url"]),
        "mime": info.get("mime"),
        "source_width": info.get("width"),
        "source_height": info.get("height"),
        "commons_sha1": info.get("sha1"),
        "license_short_name": metadata.get("LicenseShortName", {}).get("value"),
        "artist": metadata.get("Artist", {}).get("value"),
        "credit": metadata.get("Credit", {}).get("value"),
    }


def _download(filename: str, destination: Path, reuse_allowed: bool) -> dict[str, Any]:
    info = _commons_info(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or not reuse_allowed:
        request = urllib.request.Request(info["download_url"], headers={"User-Agent": USER_AGENT})
        partial = destination.with_suffix(destination.suffix + ".partial")
        with _open(request, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        partial.replace(destination)
    with Image.open(destination) as image:
        image.verify()
    with Image.open(destination) as image:
        local_size = list(image.size)
    return {
        **info,
        "local_path": str(destination.resolve()),
        "local_size": local_size,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _prior_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]) for row in manifest["entities"]}


def materialize(source_path: Path, output_root: Path, excluded_manifest: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    prior = _prior_ids(excluded_manifest)
    entities = source["entities"]
    ids = [row["id"] for row in entities]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_ENTITY_ID")
    overlap = sorted(set(ids) & prior)
    if overlap:
        raise ValueError(f"PB1_ENTITY_OVERLAP:{overlap}")
    split_sets = {
        split: {row["id"] for row in entities if row["split"] == split}
        for split in ("development", "test")
    }
    if split_sets["development"] & split_sets["test"]:
        raise ValueError("BUILDING_SPLIT_OVERLAP")

    output_root.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = output_root / "dataset_manifest.json"
    previous_rows: dict[str, dict[str, Any]] = {}
    if previous_manifest_path.exists():
        previous = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        for previous_entity in previous.get("entities", []):
            for image in previous_entity.get("references", []) + previous_entity.get("queries", []):
                previous_rows[str(Path(image["local_path"]).resolve())] = image

    def download(filename: str, destination: Path) -> dict[str, Any]:
        previous_row = previous_rows.get(str(destination.resolve()))
        if previous_row and previous_row.get("commons_file") == filename and destination.exists():
            if previous_row.get("sha256") != _sha256(destination):
                raise ValueError(f"REUSED_FILE_HASH_MISMATCH:{destination}")
            return previous_row
        return _download(filename, destination, reuse_allowed=False)

    rows: list[dict[str, Any]] = []
    for entity in entities:
        entity_root = output_root / "images" / entity["id"]
        reference = download(entity["reference"], entity_root / "reference.jpg")
        queries = []
        for index, query in enumerate(entity["queries"], start=1):
            queries.append(
                {
                    "key": f"{entity['split']}:{entity['id']}:{index:02d}",
                    "facet": query["facet"],
                    **download(query["commons_file"], entity_root / f"query-{index:02d}.jpg"),
                }
            )
            time.sleep(0.15)
        rows.append(
            {
                "id": entity["id"],
                "name": entity["name"],
                "split": entity["split"],
                "references": [reference],
                "queries": queries,
            }
        )

    inventory = [
        {
            "entity": row["id"],
            "role": role,
            "facet": image.get("facet"),
            "commons_file": image["commons_file"],
            "sha256": image["sha256"],
        }
        for row in rows
        for role, images in (("reference", row["references"]), ("query", row["queries"]))
        for image in images
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": "l10-named-poi-place-identity-manifest-v1",
        "created_at": "2026-08-29",
        "source_spec": str(source_path.resolve()),
        "source_spec_sha256": _sha256(source_path),
        "excluded_prior_manifest": str(excluded_manifest.resolve()),
        "excluded_prior_manifest_sha256": _sha256(excluded_manifest),
        "inventory_sha256": inventory_sha256,
        "entity_counts": {split: len(values) for split, values in split_sets.items()},
        "query_counts": {
            split: sum(len(row["queries"]) for row in rows if row["split"] == split)
            for split in split_sets
        },
        "entities": rows,
        "model_calls": 0,
        "claim_boundary": source["claim_boundary"],
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("named_poi_place_identity_source_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts.local/knowledge/named-poi-place-identity-v1",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=ROOT / "artifacts.local/knowledge/named-poi-portal-binding-v1/dataset_manifest.json",
    )
    args = parser.parse_args()
    manifest = materialize(args.source, args.output_root, args.exclude_manifest)
    print(json.dumps({
        "schema": manifest["schema"],
        "entity_counts": manifest["entity_counts"],
        "query_counts": manifest["query_counts"],
        "inventory_sha256": manifest["inventory_sha256"],
        "model_calls": manifest["model_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
