#!/usr/bin/env python3
"""Materialize outcome-unseen TartanGround environments for HFTF transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from materialize_stage_c_d5_tartanground_development_corpus import (
    DEFAULT_METADATA_ROOT,
    materialize_source,
    sha256,
    write_json,
)
from materialize_stage_c_d5_tartanground_development_expansion import (
    fetch_metadata,
)
from run_stage_c_d5_tartanground_development_pilot import REPO_ID, REVISION
from train_stage_c_d5_tartanground_development_student import load_jsonl


SELECTION_SEED = "HFTF_D5_OUTCOME_UNSEEN_TRANSFER_V0"
ENVIRONMENT_COUNT = 6
ROLE = "transfer"
REQUIRED_ARCHIVES = {
    "metadata.zip",
    "image_lcam_front.zip",
    "depth_lcam_front.zip",
}
DEFAULT_ARCHIVE_URL_MAP = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0b-p0c-provider-resolution-20260802/"
    "archive_url_map.json"
)
DEFAULT_BASE_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-development-corpus-v0/samples.jsonl"
)
DEFAULT_EXPANSION_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-development-expansion-v1/samples.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-outcome-unseen-transfer-v0"
)


def selection_rank(environment: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{environment}".encode("utf-8")
    ).hexdigest()


def select_parents(
    archive_map: dict[str, Any],
    used_environments: set[str],
    count: int = ENVIRONMENT_COUNT,
) -> list[dict[str, Any]]:
    candidates = []
    for row in archive_map["parents"]:
        if row["trajectory_id"] != "P1000":
            continue
        if row["environment"] in used_environments:
            continue
        if not REQUIRED_ARCHIVES.issubset(row["archive_urls"]):
            continue
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            selection_rank(row["environment"]),
            row["environment"],
            row["parent_id"],
        )
    )
    if len(candidates) < count:
        raise ValueError(
            f"Need {count} unused eligible environments, found "
            f"{len(candidates)}"
        )
    selected = candidates[:count]
    environments = [row["environment"] for row in selected]
    if len(environments) != len(set(environments)):
        raise ValueError("Selected transfer environments must be unique")
    return selected


def used_environments(
    base_samples: Path,
    expansion_samples: Path,
) -> set[str]:
    return {
        row["environment"]
        for row in (
            load_jsonl(base_samples) + load_jsonl(expansion_samples)
        )
    }


def write_corpus(
    output_root: Path,
    sources: list[dict[str, Any]],
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    used: set[str],
    archive_url_map: Path,
    base_samples: Path,
    expansion_samples: Path,
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
            "outcome_unseen_transfer_v0"
        ),
        "status": "OUTCOME_UNSEEN_TRANSFER_CORPUS_MATERIALIZED",
        "policy": {
            "selection_before_teacher_outcome": True,
            "repairable_engineering_failures": True,
            "kernel_fixed_before_transfer": (
                "height_spatiotemporal_selective_v2"
            ),
            "one_shot": False,
            "human_event_truth": False,
            "system_claim": False,
        },
        "provider": {"repo_id": REPO_ID, "revision": REVISION},
        "selection": {
            "seed": SELECTION_SEED,
            "rule": (
                "exclude every environment in the base and expansion "
                "samples; keep P1000 parents with metadata/front-image/"
                "front-depth archives; ascending "
                "sha256(seed_colon_environment); first six"
            ),
            "used_environments": sorted(used),
            "selected": [
                {
                    "environment": row["environment"],
                    "parent_id": row["parent_id"],
                    "rank_sha256": selection_rank(row["environment"]),
                }
                for row in selected
            ],
        },
        "inputs": {
            "archive_url_map_sha256": sha256(archive_url_map),
            "base_samples_sha256": sha256(base_samples),
            "expansion_samples_sha256": sha256(expansion_samples),
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
        "--archive-url-map",
        type=Path,
        default=DEFAULT_ARCHIVE_URL_MAP,
    )
    parser.add_argument(
        "--base-samples",
        type=Path,
        default=DEFAULT_BASE_SAMPLES,
    )
    parser.add_argument(
        "--expansion-samples",
        type=Path,
        default=DEFAULT_EXPANSION_SAMPLES,
    )
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
    args = parser.parse_args()

    archive_map = json.loads(
        args.archive_url_map.read_text(encoding="utf-8")
    )
    used = used_environments(
        args.base_samples,
        args.expansion_samples,
    )
    selected = select_parents(archive_map, used)
    sources = []
    records = []
    for row in selected:
        parent_id = row["parent_id"]
        metadata_path = fetch_metadata(args.metadata_root, parent_id)
        source, parent_records = materialize_source(
            args.metadata_root,
            args.output_root,
            ROLE,
            parent_id,
        )
        source["metadata"] = {
            "bytes": metadata_path.stat().st_size,
            "sha256": sha256(metadata_path),
        }
        sources.append(source)
        records.extend(parent_records)
        print(
            json.dumps(
                {
                    "parent_id": parent_id,
                    "samples": len(parent_records),
                }
            ),
            flush=True,
        )
    manifest = write_corpus(
        args.output_root,
        sources,
        records,
        selected,
        used,
        args.archive_url_map,
        args.base_samples,
        args.expansion_samples,
    )
    print(
        json.dumps(
            {
                "source_count": manifest["source_count"],
                "sample_count": manifest["sample_count"],
                "selected_environments": [
                    row["environment"] for row in selected
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
