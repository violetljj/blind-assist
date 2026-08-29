"""Materialize the PB1 building-disjoint public-image dataset.

The source JSON freezes exact Commons filenames and reuses already frozen local
portal/query truth. This command downloads only the additional public reference
and training-query files, verifies every local source hash, and emits one
content-addressed manifest. It performs no model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BlindAssist-L10-PB1/1.0 (research prototype)"


def _urlopen_with_retry(request: urllib.request.Request, timeout: int) -> Any:
    """Open a public Commons request with bounded Retry-After handling."""
    delays = (2, 4, 8, 16, 30)
    for attempt in range(len(delays) + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt >= len(delays):
                raise
            retry_after = error.headers.get("Retry-After")
            delay = delays[attempt]
            if retry_after is not None:
                try:
                    delay = min(30, max(delay, int(retry_after)))
                except ValueError:
                    pass
            time.sleep(delay)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label}_HASH_MISMATCH:expected={expected}:observed={observed}")


def _commons_info(filename: str) -> dict[str, Any]:
    title = filename if filename.startswith("File:") else f"File:{filename}"
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|sha1",
        "iiurlwidth": "1280",
        "format": "json",
        "formatversion": "2",
    }
    request = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
    )
    with _urlopen_with_retry(request, timeout=60) as response:
        payload = json.load(response)
    pages = payload.get("query", {}).get("pages", [])
    if len(pages) != 1 or "missing" in pages[0] or not pages[0].get("imageinfo"):
        raise ValueError(f"COMMONS_FILE_NOT_FOUND:{filename}")
    page = pages[0]
    info = page["imageinfo"][0]
    return {
        "commons_file": page["title"].removeprefix("File:"),
        "source_page": info["descriptionurl"],
        "download_url": info.get("thumburl") or info["url"],
        "mime": info["mime"],
        "source_width": int(info["width"]),
        "source_height": int(info["height"]),
        "commons_sha1": info.get("sha1"),
    }


def _download_commons(filename: str, destination: Path) -> dict[str, Any]:
    info = _commons_info(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        part = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(
            info["download_url"],
            headers={"User-Agent": USER_AGENT, "Referer": info["source_page"]},
        )
        with _urlopen_with_retry(request, timeout=120) as response, part.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        with Image.open(part) as image:
            image.verify()
        part.replace(destination)
    with Image.open(destination) as image:
        local_size = [int(image.width), int(image.height)]
    return {
        **info,
        "local_path": str(destination.resolve()),
        "local_size": local_size,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _portal_entities(
    spec: dict[str, Any],
    split: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    source = spec["splits"][split]["prior_portal_source"]
    manifest_path = ROOT / source["manifest"]
    audit_path = ROOT / source["audit"]
    _require_hash(manifest_path, source["manifest_sha256"], f"{split.upper()}_PORTAL_MANIFEST")
    _require_hash(audit_path, source["audit_sha256"], f"{split.upper()}_PORTAL_AUDIT")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    frames = {int(row["index"]): row for row in manifest["frames"]}
    truth = audit["portal_set_truth"]["frames"]
    entities = []
    for entity in source["entities"]:
        frame_index = int(entity["frame"])
        frame = frames[frame_index]
        query_path = Path(frame["local_path"])
        if not query_path.exists() or _sha256(query_path) != frame["sha256"]:
            raise ValueError(f"{split.upper()}_QUERY_MISSING_OR_HASH_MISMATCH:{entity['id']}")
        reference = _download_commons(
            entity["reference_commons_file"],
            output_root / "images" / entity["id"] / "reference.jpg",
        )
        truth_row = truth[str(frame_index)]
        entities.append(
            {
                "id": entity["id"],
                "name": entity["name"],
                "split": split,
                "references": [reference],
                "queries": [
                    {
                        "key": f"{split}:{entity['id']}:01",
                        "local_path": str(query_path.resolve()),
                        "sha256": frame["sha256"],
                        "image_size": list(truth_row["local_image_size"]),
                        "truth_boxes_xyxy": [truth_row["portal_set_box_xyxy"]],
                        "truth_kind": truth_row["kind"],
                        "truth_authority": audit["portal_set_truth"]["authority"],
                    }
                ],
            }
        )
    return entities


def _added_train_entities(spec: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    audit_path = Path(__file__).with_name("named_poi_portal_binding_source_audit_v1.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    truth_by_id = audit["portal_set_truth"]["entities"]
    entities = []
    for entity in spec["splits"]["train"]["added_entities"]:
        entity_root = output_root / "images" / entity["id"]
        query = _download_commons(entity["query_commons_file"], entity_root / "query.jpg")
        reference = _download_commons(entity["reference_commons_file"], entity_root / "reference.jpg")
        truth = truth_by_id.get(entity["id"])
        if truth is None:
            raise ValueError(f"ADDED_TRAIN_TRUTH_MISSING:{entity['id']}")
        if query["local_size"] != truth["image_size"]:
            raise ValueError(f"ADDED_TRAIN_TRUTH_SIZE_MISMATCH:{entity['id']}")
        entities.append(
            {
                "id": entity["id"],
                "name": entity["name"],
                "split": "train",
                "references": [reference],
                "queries": [
                    {
                        "key": f"train:{entity['id']}:01",
                        "local_path": query["local_path"],
                        "sha256": query["sha256"],
                        "image_size": query["local_size"],
                        "truth_boxes_xyxy": [truth["portal_set_box_xyxy"]],
                        "truth_kind": truth["kind"],
                        "truth_authority": audit["portal_set_truth"]["authority"],
                        "source": query,
                    }
                ],
            }
        )
    return entities


def _development_entities(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source = spec["splits"]["development"]
    protocol_path = ROOT / source["protocol"]
    reference_path = ROOT / source["reference_library"]
    query_path = ROOT / source["query_library"]
    _require_hash(protocol_path, source["protocol_sha256"], "DEVELOPMENT_PROTOCOL")
    _require_hash(reference_path, source["reference_library_sha256"], "DEVELOPMENT_REFERENCE_LIBRARY")
    _require_hash(query_path, source["query_library_sha256"], "DEVELOPMENT_QUERY_LIBRARY")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    query_payload = json.loads(query_path.read_text(encoding="utf-8"))
    references_by_id = {str(row["id"]): row for row in reference_payload["targets"]}
    queries_by_id = {str(row["id"]): row for row in query_payload["targets"]}
    entities = []
    for target_id, roles in protocol["targets"].items():
        if not roles["evaluation_indices"]:
            continue
        reference_target = references_by_id[target_id]
        references = []
        for index in roles["reference_indices"]:
            row = reference_target["facets"][int(index) - 1]
            path = reference_path.parent / "images" / target_id / Path(row["local_path"]).name
            if _sha256(path) != row["sha256"]:
                raise ValueError(f"DEVELOPMENT_REFERENCE_HASH_MISMATCH:{target_id}:{index}")
            references.append(
                {
                    "commons_file": row["commons_file"],
                    "local_path": str(path.resolve()),
                    "sha256": row["sha256"],
                    "local_size": list(Image.open(path).size),
                }
            )
        query_target = queries_by_id[target_id]
        queries = []
        for index in roles["evaluation_indices"]:
            row = query_target["facets"][int(index) - 1]
            path = query_path.parent / "images" / target_id / Path(row["local_path"]).name
            if _sha256(path) != row["sha256"]:
                raise ValueError(f"DEVELOPMENT_QUERY_HASH_MISMATCH:{target_id}:{index}")
            truth = protocol["exact_entrance_truth"].get(target_id, {}).get(str(index))
            image_size = list(Image.open(path).size)
            if truth is not None and image_size != truth["image_size"]:
                raise ValueError(f"DEVELOPMENT_TRUTH_SIZE_MISMATCH:{target_id}:{index}")
            queries.append(
                {
                    "key": f"development:{target_id}:{int(index):02d}",
                    "local_path": str(path.resolve()),
                    "sha256": row["sha256"],
                    "image_size": image_size,
                    "truth_boxes_xyxy": [] if truth is None else truth["boxes_xyxy"],
                    "truth_kind": None if truth is None else truth["kind"],
                    "truth_authority": "REUSED_DISCLOSED_V7_DEVELOPMENT_PIXEL_TRUTH",
                }
            )
        entities.append(
            {
                "id": target_id,
                "name": reference_target.get("name", target_id),
                "split": "development",
                "references": references,
                "queries": queries,
            }
        )
    return entities


def _inventory(entities: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for entity in entities:
        for role in ("references", "queries"):
            for row in entity[role]:
                rows.append(f"{entity['split']}\t{entity['id']}\t{role}\t{row['sha256']}\n")
    return {
        "entity_count": len(entities),
        "reference_count": sum(len(row["references"]) for row in entities),
        "query_count": sum(len(row["queries"]) for row in entities),
        "inventory_sha256": hashlib.sha256("".join(sorted(rows)).encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("named_poi_portal_binding_source_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts.local/knowledge/named-poi-portal-binding-v1",
    )
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    spec = json.loads(source_path.read_text(encoding="utf-8"))
    entities = []
    entities.extend(_portal_entities(spec, "train", output_root))
    entities.extend(_added_train_entities(spec, output_root))
    entities.extend(_development_entities(spec))
    entities.extend(_portal_entities(spec, "test", output_root))
    split_ids = {
        split: {row["id"] for row in entities if row["split"] == split}
        for split in ("train", "development", "test")
    }
    if any(split_ids[left] & split_ids[right] for left, right in (("train", "development"), ("train", "test"), ("development", "test"))):
        raise ValueError("BUILDING_SPLIT_OVERLAP")
    manifest = {
        "schema": "l10-named-poi-portal-binding-manifest-v1",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "added_source_audit": str(Path(__file__).with_name("named_poi_portal_binding_source_audit_v1.json").resolve()),
        "added_source_audit_sha256": _sha256(Path(__file__).with_name("named_poi_portal_binding_source_audit_v1.json")),
        "split_entity_ids": {key: sorted(value) for key, value in split_ids.items()},
        "inventory": _inventory(entities),
        "entities": entities,
        "model_calls": 0,
        "claim_boundary": spec["claim_boundary"],
    }
    destination = output_root / "dataset_manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(destination), **manifest["inventory"], "split_entity_ids": manifest["split_entity_ids"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
