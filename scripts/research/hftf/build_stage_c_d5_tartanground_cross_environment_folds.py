#!/usr/bin/env python3
"""Build outcome-open environment-grouped HFTF Development folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from materialize_stage_c_d5_tartanground_development_corpus import (
    sha256,
    write_json,
)
from train_stage_c_d5_tartanground_development_student import load_jsonl


SELECTION_SEED = "HFTF_D5_CROSS_ENV_V1"
FOLD_COUNT = 3
FAMILY_OVERRIDES = {
    "WaterMillDay": "WaterMill",
    "WaterMillNight": "WaterMill",
}
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
    "stage-c-d5-tartanground-cross-environment-v1"
)


def family(environment: str) -> str:
    return FAMILY_OVERRIDES.get(environment, environment)


def family_rank(value: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{value}".encode("utf-8")
    ).hexdigest()


def assign_folds(environments: list[str]) -> dict[str, int]:
    grouped: dict[str, list[str]] = {}
    for environment in sorted(set(environments)):
        grouped.setdefault(family(environment), []).append(environment)
    ordered = sorted(
        (family_rank(name), name, members)
        for name, members in grouped.items()
    )
    fold_members: list[list[str]] = [[] for _ in range(FOLD_COUNT)]
    for _, _, members in ordered:
        fold_index = min(
            range(FOLD_COUNT),
            key=lambda index: (len(fold_members[index]), index),
        )
        fold_members[fold_index].extend(members)
    return {
        environment: fold_index
        for fold_index, members in enumerate(fold_members)
        for environment in members
    }


def write_fold(
    output_root: Path,
    fold_index: int,
    records: list[dict[str, Any]],
    assignments: dict[str, int],
) -> dict[str, Any]:
    output_directory = output_root / f"fold-{fold_index}"
    rows = []
    for record in records:
        row = dict(record)
        row["source_role"] = record["role"]
        row["role"] = (
            "dev"
            if assignments[record["environment"]] == fold_index
            else "train"
        )
        rows.append(row)
    samples_path = output_directory / "samples.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = samples_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(samples_path)
    train_environments = sorted(
        environment
        for environment, assigned in assignments.items()
        if assigned != fold_index
    )
    dev_environments = sorted(
        environment
        for environment, assigned in assignments.items()
        if assigned == fold_index
    )
    return {
        "fold": fold_index,
        "train_environments": train_environments,
        "dev_environments": dev_environments,
        "train_samples": sum(row["role"] == "train" for row in rows),
        "dev_samples": sum(row["role"] == "dev" for row in rows),
        "samples": {
            "path": str(samples_path.resolve()),
            "bytes": samples_path.stat().st_size,
            "sha256": sha256(samples_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
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
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    records = load_jsonl(args.base_samples) + load_jsonl(
        args.expansion_samples
    )
    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs across input corpora")
    environments = sorted({record["environment"] for record in records})
    assignments = assign_folds(environments)
    fold_rows = [
        write_fold(args.output_root, index, records, assignments)
        for index in range(FOLD_COUNT)
    ]
    manifest = {
        "schema": (
            "blindassist_hftf_stage_c_d5_tartanground_"
            "cross_environment_folds_v1"
        ),
        "status": "DEVELOPMENT_CROSS_ENVIRONMENT_FOLDS_READY",
        "policy": {
            "outcome_open": True,
            "repairable": True,
            "one_shot": False,
            "heldout": False,
            "promotion_evidence": False,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "family_overrides": FAMILY_OVERRIDES,
            "rule": (
                "ascending_sha256(seed_colon_family), greedily assign each "
                "family to the fold with the fewest environments"
            ),
        },
        "inputs": {
            "base_samples_sha256": sha256(args.base_samples),
            "expansion_samples_sha256": sha256(
                args.expansion_samples
            ),
        },
        "environment_count": len(environments),
        "sample_count": len(records),
        "assignments": assignments,
        "folds": fold_rows,
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "environment_count": len(environments),
                "sample_count": len(records),
                "folds": [
                    {
                        "fold": row["fold"],
                        "train_environments": len(
                            row["train_environments"]
                        ),
                        "dev_environments": len(
                            row["dev_environments"]
                        ),
                    }
                    for row in fold_rows
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
