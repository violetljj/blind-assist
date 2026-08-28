"""Expand a frozen named-POI library with metadata-prioritized Commons facets."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


USER_AGENT = "BlindAssist-L10-Multifacet/1.0 (research prototype)"
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}
FACET_TERMS = {
    "entrance": ("entrance", "entry", "exit", "gate", "door", "portal", "入口", "出口", "大門", "门"),
    "facade": ("facade", "façade", "front", "exterior", "building", "tower", "大樓", "大楼", "外觀", "外观"),
    "wayfinding": ("station", "hospital", "campus", "concourse", "platform", "sign", "directory", "mtr", "港鐵", "醫院", "医院"),
}


def _api_json(endpoint: str, **params: str) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint + "?" + urllib.parse.urlencode(params), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _claim_values(entity: dict[str, Any], property_id: str) -> list[Any]:
    output = []
    for statement in entity.get("claims", {}).get(property_id, []):
        snak = statement.get("mainsnak", {})
        if snak.get("snaktype") == "value":
            output.append(snak["datavalue"]["value"])
    return output


def _category_files(category: str) -> list[str]:
    payload = _api_json(
        "https://commons.wikimedia.org/w/api.php",
        action="query",
        list="categorymembers",
        cmtitle=f"Category:{category}",
        cmnamespace="6",
        cmlimit="50",
        format="json",
    )
    return [row["title"].removeprefix("File:") for row in payload["query"]["categorymembers"]]


def _file_metadata(names: Iterable[str]) -> dict[str, dict[str, Any]]:
    titles = "|".join(f"File:{name}" for name in names)
    if not titles:
        return {}
    payload = _api_json(
        "https://commons.wikimedia.org/w/api.php",
        action="query",
        prop="imageinfo",
        titles=titles,
        format="json",
        iiprop="url|mime|size|extmetadata",
        iiurlwidth="1280",
    )
    output = {}
    for page in payload["query"]["pages"].values():
        info = (page.get("imageinfo") or [None])[0]
        if info:
            output[page["title"].removeprefix("File:")] = info
    return output


def _clean_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def _facet(filename: str, aliases: list[str]) -> tuple[str, int, list[str]]:
    normalized = filename.casefold()
    reasons = []
    for facet, terms in FACET_TERMS.items():
        matches = [term for term in terms if term.casefold() in normalized]
        if matches:
            reasons.extend(f"{facet}:{term}" for term in matches)
    alias_tokens = {
        token.casefold()
        for alias in aliases
        for token in re.findall(r"[A-Za-z0-9]{3,}|[\u3400-\u9fff]{2,}", alias)
    }
    alias_matches = sorted(token for token in alias_tokens if token in normalized)
    reasons.extend(f"alias:{token}" for token in alias_matches)
    if any(reason.startswith("entrance:") for reason in reasons):
        facet = "entrance"
    elif alias_matches and any(reason.startswith("facade:") for reason in reasons):
        facet = "facade"
    elif alias_matches or any(reason.startswith("wayfinding:") for reason in reasons):
        facet = "wayfinding"
    elif any(reason.startswith("facade:") for reason in reasons):
        facet = "facade"
    else:
        facet = "context"
    priority = {"entrance": 0, "facade": 1, "wayfinding": 2, "context": 3}[facet]
    return facet, priority, reasons


def _download(url: str, path: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
            time.sleep(0.25)
            return digest.hexdigest()
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            retryable = not isinstance(error, urllib.error.HTTPError) or error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt == 3:
                raise
            time.sleep(min(30.0, 5.0 * (attempt + 1)))
    raise AssertionError("unreachable")


def materialize(
    source_manifest: Path,
    exclude_manifests: list[Path],
    output_root: Path,
    target_ids: set[str],
    images_per_target: int,
) -> Path:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    targets_by_id = {str(row["id"]): row for row in source["targets"]}
    unknown = target_ids - set(targets_by_id)
    if unknown:
        raise ValueError(f"UNKNOWN_TARGETS:{sorted(unknown)}")
    selected_targets = [targets_by_id[target_id] for target_id in sorted(target_ids)]
    excluded_names: dict[str, set[str]] = {target_id: set() for target_id in target_ids}
    for manifest_path in [source_manifest, *exclude_manifests]:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for target in payload.get("targets", []):
            target_id = str(target.get("id", ""))
            if target_id not in excluded_names:
                continue
            for row in [*target.get("reference_images", []), *target.get("facets", [])]:
                excluded_names[target_id].add(str(row["commons_file"]).casefold())
    qids = "|".join(str(row["wikidata_qid"]) for row in selected_targets)
    entities = _api_json(
        "https://www.wikidata.org/w/api.php",
        action="wbgetentities",
        ids=qids,
        format="json",
        props="claims",
    )["entities"]
    result_targets = []
    for target in selected_targets:
        entity = entities[target["wikidata_qid"]]
        categories = [str(value) for value in _claim_values(entity, "P373")]
        if not categories:
            result_targets.append({**target, "facets": [], "source_status": "NO_COMMONS_CATEGORY"})
            continue
        candidates = [
            name for name in _category_files(categories[0]) if name.casefold() not in excluded_names[target["id"]]
        ]
        metadata = _file_metadata(candidates)
        ranked = []
        for filename in candidates:
            info = metadata.get(filename)
            if not info or info.get("mime") not in IMAGE_MIMES:
                continue
            facet, priority, reasons = _facet(filename, [str(value) for value in target["names"]])
            ranked.append((priority, filename.casefold(), filename, facet, reasons, info))
        ranked.sort()
        facets = []
        for _, _, filename, facet, reasons, info in ranked[:images_per_target]:
            url = str(info.get("thumburl") or info["url"])
            suffix = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
            path = output_root / "images" / target["id"] / f"{len(facets) + 1:02d}{suffix}"
            digest = _download(url, path)
            ext = info.get("extmetadata", {})
            facets.append(
                {
                    "local_path": str(path.resolve()),
                    "sha256": digest,
                    "commons_file": filename,
                    "source_page": info.get("descriptionurl"),
                    "download_url": url,
                    "license": _clean_html(ext.get("LicenseShortName", {}).get("value", "")),
                    "artist": _clean_html(ext.get("Artist", {}).get("value", "")),
                    "width": int(info.get("width", 0)),
                    "height": int(info.get("height", 0)),
                    "facet_hint": facet,
                    "facet_reasons": reasons,
                    "selection_authority": "COMMONS_FILENAME_METADATA_ONLY_BEFORE_PIXEL_ACCESS",
                }
            )
        result_targets.append(
            {
                "id": target["id"],
                "kind": target["kind"],
                "wikidata_qid": target["wikidata_qid"],
                "names": target["names"],
                "latitude": target["latitude"],
                "longitude": target["longitude"],
                "coordinate_source": target["coordinate_source"],
                "commons_category": categories[0],
                "facets": facets,
                "source_status": "MATERIALIZED" if facets else "NO_NEW_ELIGIBLE_IMAGES",
            }
        )
    manifest = {
        "schema": "blindassist.named_poi_multifacet_library.v1",
        "claim_scope": "PUBLIC_REFERENCE_INFORMATION_SOURCE_ONLY",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": source_hash,
        "exclude_manifests": [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in exclude_manifests
        ],
        "selection_rule": "metadata facet priority entrance, facade, wayfinding, context; filename order within facet; prior library files excluded",
        "target_count": len(result_targets),
        "targets": result_targets,
    }
    manifest_path = output_root / "multifacet_library.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts.local/knowledge/named-poi-facade-v2/target_library.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--images-per-target", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.images_per_target <= 30:
        raise ValueError("images_per_target must be in [1, 30]")
    path = materialize(
        args.source_manifest.resolve(),
        [path.resolve() for path in args.exclude_manifest],
        args.output_root.resolve(),
        set(args.target),
        args.images_per_target,
    )
    print(json.dumps({"multifacet_library": str(path.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
