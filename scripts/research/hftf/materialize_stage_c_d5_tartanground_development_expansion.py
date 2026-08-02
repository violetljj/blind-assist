#!/usr/bin/env python3
"""Materialize an outcome-open TartanGround environment expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import fsspec

from materialize_stage_c_d5_tartanground_development_corpus import (
    DEFAULT_METADATA_ROOT,
    materialize_source,
    sha256,
    write_json,
)
from run_stage_c_d5_tartanground_development_pilot import REPO_ID, REVISION


SELECTION_SEED = "HFTF_D5_EXPANSION_V1"
DIAGNOSTIC_PARENT = "WaterMillDay/Data_diff/P1002"
GENERALIZATION_ENVIRONMENTS = (
    "Downtown",
    "JapaneseAlley",
    "NordicHarbor",
    "Supermarket",
    "OldTownNight",
    "GreatMarsh",
)
SOURCES = (DIAGNOSTIC_PARENT,) + tuple(
    f"{environment}/Data_diff/P1000"
    for environment in GENERALIZATION_ENVIRONMENTS
)
ROLE = "expansion_dev"
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-development-expansion-v1"
)


def selection_rank(environment: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{environment}".encode("utf-8")
    ).hexdigest()


def metadata_remote(parent_id: str) -> str:
    return (
        f"hf://datasets/{REPO_ID}@{REVISION}/"
        f"{parent_id}/metadata.zip"
    )


def fetch_metadata(metadata_root: Path, parent_id: str) -> Path:
    destination = metadata_root / parent_id / "metadata.zip"
    if destination.is_file():
        with zipfile.ZipFile(destination) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(
                    f"{parent_id} metadata has corrupt member {corrupt}"
                )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with fsspec.open(metadata_remote(parent_id), "rb") as source:
        temporary.write_bytes(source.read())
    with zipfile.ZipFile(temporary) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(
                f"{parent_id} metadata has corrupt member {corrupt}"
            )
    temporary.replace(destination)
    return destination


def materialize_parent(
    metadata_root: Path,
    output_root: Path,
    parent_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_path = fetch_metadata(metadata_root, parent_id)
    source, records = materialize_source(
        metadata_root,
        output_root,
        ROLE,
        parent_id,
    )
    source["metadata"] = {
        "bytes": metadata_path.stat().st_size,
        "sha256": sha256(metadata_path),
    }
    return source, records


def write_corpus(
    output_root: Path,
    sources: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    samples_path = output_root / "samples.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = samples_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    temporary.replace(samples_path)
    manifest = {
        "schema": (
            "blindassist_hftf_stage_c_d5_tartanground_"
            "development_expansion_v1"
        ),
        "status": "DEVELOPMENT_EXPANSION_MATERIALIZED",
        "policy": {
            "outcome_open": True,
            "repairable": True,
            "one_shot": False,
            "heldout": False,
            "promotion_evidence": False,
        },
        "provider": {"repo_id": REPO_ID, "revision": REVISION},
        "selection": {
            "diagnostic_parent": DIAGNOSTIC_PARENT,
            "generalization_seed": SELECTION_SEED,
            "generalization_rule": (
                "ascending_sha256(seed_colon_environment), among unused "
                "environments with P1000, first six"
            ),
            "generalization_environments": [
                {
                    "environment": environment,
                    "rank_sha256": selection_rank(environment),
                }
                for environment in GENERALIZATION_ENVIRONMENTS
            ],
        },
        "sources": sources,
        "source_count": len(sources),
        "sample_count": len(records),
        "samples": {
            "path": str(samples_path.resolve()),
            "bytes": samples_path.stat().st_size,
            "sha256": sha256(samples_path),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--prefetch-parent",
        action="append",
        default=[],
        help=(
            "Materialize only the named expansion parent(s), then stop "
            "without replacing the expansion manifest or samples JSONL."
        ),
    )
    args = parser.parse_args()

    selected = args.prefetch_parent or list(SOURCES)
    unknown = sorted(set(selected) - set(SOURCES))
    if unknown:
        parser.error(f"Unknown expansion parent(s): {unknown}")

    source_rows = []
    records = []
    for parent_id in selected:
        source, parent_records = materialize_parent(
            args.metadata_root,
            args.output_root,
            parent_id,
        )
        source_rows.append(source)
        records.extend(parent_records)
        print(
            json.dumps(
                {
                    "parent_id": parent_id,
                    "samples": len(parent_records),
                    "prefetch_only": bool(args.prefetch_parent),
                }
            ),
            flush=True,
        )

    if args.prefetch_parent:
        return 0
    manifest = write_corpus(args.output_root, source_rows, records)
    print(
        json.dumps(
            {
                "source_count": manifest["source_count"],
                "sample_count": manifest["sample_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
