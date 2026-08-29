"""Materialize the source-only PB4 bilingual identity cohort from Commons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from materialize_named_poi_place_identity import _download, _sha256


ROOT = Path(__file__).resolve().parents[3]


def _prior_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result.update(str(row["id"]) for row in manifest["entities"])
    return result


def materialize(source_path: Path, output_root: Path, excluded: list[Path]) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    entities = source["entities"]
    ids = [str(row["id"]) for row in entities]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_ENTITY_ID")
    overlap = sorted(set(ids) & _prior_ids(excluded))
    if overlap:
        raise ValueError(f"PRIOR_ENTITY_OVERLAP:{overlap}")
    development = {row["id"] for row in entities if row["split"] == "development"}
    test = {row["id"] for row in entities if row["split"] == "test"}
    if development & test or len(development) != 4 or len(test) != 4:
        raise ValueError("INVALID_BUILDING_DISJOINT_SPLIT")

    previous_rows: dict[str, dict[str, Any]] = {}
    previous_manifest = output_root / "dataset_manifest.json"
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        for previous_entity in previous.get("entities", []):
            for image in previous_entity.get("references", []) + previous_entity.get("queries", []):
                previous_rows[str(Path(image["local_path"]).resolve())] = image

    def download(filename: str, destination: Path) -> dict[str, Any]:
        previous = previous_rows.get(str(destination.resolve()))
        if previous and previous.get("commons_file") == filename and destination.exists():
            observed = _sha256(destination)
            if previous.get("sha256") != observed:
                raise ValueError(f"REUSED_FILE_HASH_MISMATCH:{destination}")
            return previous
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
        rows.append(
            {
                "id": entity["id"],
                "name": entity["name"],
                "aliases": entity["aliases"],
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
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "l10-named-poi-biscript-manifest-v1",
        "created_at": "2026-08-29",
        "source_spec": str(source_path.resolve()),
        "source_spec_sha256": _sha256(source_path),
        "excluded_manifests": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in excluded
        ],
        "inventory_sha256": inventory_sha256,
        "entity_counts": {"development": len(development), "test": len(test)},
        "query_counts": {
            split: sum(len(row["queries"]) for row in rows if row["split"] == split)
            for split in ("development", "test")
        },
        "entities": rows,
        "model_calls": 0,
        "claim_boundary": source["claim_boundary"],
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("named_poi_biscript_source_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts.local/knowledge/named-poi-biscript-v1",
    )
    parser.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = materialize(
        args.source.resolve(),
        args.output_root.resolve(),
        [path.resolve() for path in args.exclude_manifest],
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("entity_counts", "query_counts", "inventory_sha256", "model_calls")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
