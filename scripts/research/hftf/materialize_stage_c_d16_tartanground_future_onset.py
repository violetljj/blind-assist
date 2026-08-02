#!/usr/bin/env python3
"""Materialize true future-onset targets on the 15-environment TartanGround corpus."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)


SCHEMA = "blindassist_hftf_stage_c_d16_tartanground_future_onset_v0"
DEFAULT_FOLD_MANIFEST = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-cross-environment-v1/manifest.json"
)
DEFAULT_BASE_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-development-corpus-v0/samples.jsonl"
)
DEFAULT_EXPANSION_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-development-expansion-v1/samples.jsonl"
)
HORIZONS = ("near", "far")
HEIGHT_INDICES = {"body": 1, "head": 2}


def onset_target(record: dict[str, Any]) -> dict[str, Any]:
    current = record["labels"]["current"]
    current_known = np.asarray(current["known_target"], dtype=bool)
    current_risk = np.asarray(
        current["risk_score_target_nullable"],
        dtype=np.float64,
    )
    cell_eligible = np.zeros((2, 2, 6, 6), dtype=np.int8)
    cell_onset = np.zeros((2, 2, 6, 6), dtype=np.int8)
    sample_eligible: dict[str, bool] = {}
    sample_onset: dict[str, bool] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        future = record["labels"][horizon]
        future_known = np.asarray(
            future["known_target"],
            dtype=bool,
        )
        future_risk = np.asarray(
            future["risk_score_target_nullable"],
            dtype=np.float64,
        )
        for height_index, (height, source_index) in enumerate(
            HEIGHT_INDICES.items()
        ):
            common_known = (
                current_known[source_index]
                & future_known[source_index]
            )
            eligible = (
                common_known
                & (current_risk[source_index] < 0.5)
            )
            onset = eligible & (future_risk[source_index] >= 0.5)
            cell_eligible[horizon_index, height_index] = eligible
            cell_onset[horizon_index, height_index] = onset
            key = f"{horizon}_{height}"
            sample_eligible[key] = bool(np.any(eligible))
            sample_onset[key] = bool(np.any(onset))
    return {
        "sample_eligible": sample_eligible,
        "sample_onset": sample_onset,
        "cell_eligible": cell_eligible.tolist(),
        "cell_onset": cell_onset.tolist(),
        "eligible_cell_count": int(np.sum(cell_eligible)),
        "onset_cell_count": int(np.sum(cell_onset)),
    }


def target_counts(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    eligible = [
        row for row in rows
        if row["future_onset_target"]["sample_eligible"][key]
    ]
    positive = sum(
        int(row["future_onset_target"]["sample_onset"][key])
        for row in eligible
    )
    return {
        "eligible": len(eligible),
        "positive": positive,
        "negative": len(eligible) - positive,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=DEFAULT_FOLD_MANIFEST,
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
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("Refusing to overwrite D16 onset output")

    manifest = json.loads(
        args.fold_manifest.read_text(encoding="utf-8")
    )
    if manifest["sample_count"] != 495:
        raise ValueError("Expected 495-sample cross-environment manifest")
    if (
        manifest["inputs"]["base_samples_sha256"]
        != sha256(args.base_samples)
        or manifest["inputs"]["expansion_samples_sha256"]
        != sha256(args.expansion_samples)
    ):
        raise ValueError("D16 input binding mismatch")
    assignments = {
        str(environment): int(fold)
        for environment, fold in manifest["assignments"].items()
    }
    records = load_jsonl(args.base_samples) + load_jsonl(
        args.expansion_samples
    )
    records.sort(key=lambda row: row["sample_id"])
    if (
        len(records) != 495
        or len({row["sample_id"] for row in records}) != 495
    ):
        raise ValueError("Expected 495 unique TartanGround samples")

    rows = []
    for record in records:
        environment = str(record["environment"])
        rows.append(
            {
                "sample_id": record["sample_id"],
                "environment": environment,
                "environment_fold": assignments[environment],
                "parent_id": record["parent_id"],
                "anchor_frame_id": int(record["anchor_frame_id"]),
                "history_rgb": record["history_rgb"],
                "future_onset_target": onset_target(record),
            }
        )
    keys = tuple(
        f"{horizon}_{height}"
        for horizon in HORIZONS
        for height in HEIGHT_INDICES
    )
    by_fold = []
    for fold in range(3):
        fold_rows = [
            row for row in rows if row["environment_fold"] == fold
        ]
        by_fold.append(
            {
                "fold": fold,
                "samples": len(fold_rows),
                "environments": sorted(
                    {row["environment"] for row in fold_rows}
                ),
                "targets": {
                    key: target_counts(fold_rows, key)
                    for key in keys
                },
            }
        )
    ready = all(
        fold_row["targets"][key]["positive"] > 0
        and fold_row["targets"][key]["negative"] > 0
        for fold_row in by_fold
        for key in keys
    )
    status = (
        "D16_TARTANGROUND_FUTURE_ONSET_THREE_FOLD_READY"
        if ready
        else "D16_TARTANGROUND_FUTURE_ONSET_NOT_THREE_FOLD_READY"
    )

    args.output_root.mkdir(parents=True)
    samples_path = args.output_root / "samples.jsonl"
    with samples_path.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development synthetic true-onset opportunity corpus",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "fold_manifest_path": str(args.fold_manifest.resolve()),
            "fold_manifest_sha256": sha256(args.fold_manifest),
            "base_samples_path": str(args.base_samples.resolve()),
            "base_samples_sha256": sha256(args.base_samples),
            "expansion_samples_path": str(
                args.expansion_samples.resolve()
            ),
            "expansion_samples_sha256": sha256(
                args.expansion_samples
            ),
        },
        "definition": {
            "horizons": list(HORIZONS),
            "heights": list(HEIGHT_INDICES),
            "cell_eligible": (
                "current and future cell both known, current risk < 0.5"
            ),
            "cell_onset": "eligible cell with future risk >= 0.5",
            "sample_onset": "any onset cell for horizon x height",
            "folds": (
                "inherit HFTF_D5_CROSS_ENV_V1 environment assignments "
                "without reassignment"
            ),
        },
        "counts": {
            "samples": len(rows),
            "environments": len(
                {row["environment"] for row in rows}
            ),
            "eligible_cells": sum(
                row["future_onset_target"]["eligible_cell_count"]
                for row in rows
            ),
            "onset_cells": sum(
                row["future_onset_target"]["onset_cell_count"]
                for row in rows
            ),
            "targets": {
                key: target_counts(rows, key)
                for key in keys
            },
            "by_fold": by_fold,
        },
        "outputs": {
            "samples_path": str(samples_path.resolve()),
            "samples_sha256": sha256(samples_path),
        },
        "next_action": (
            "run equal-capacity current/history onset baseline across the "
            "three inherited environment folds"
            if ready
            else (
                "do not train; repair target granularity or add environments "
                "without changing inherited folds"
            )
        ),
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(report_path)
    Path(str(report_path) + ".sha256").write_text(
        f"{digest}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "counts": report["counts"],
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
