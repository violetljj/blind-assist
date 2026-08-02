#!/usr/bin/env python3
"""Materialize current-safe future onset targets from the JRDB D9 corpus."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)


SCHEMA = "blindassist_hftf_stage_c_d15_jrdb_future_onset_v0"
DEFAULT_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d9-jrdb-local-route-replication-v0/samples.jsonl"
)
PROXIMITY_THRESHOLD_M = 1.25
CORRIDOR_FORWARD_LIMIT_M = 4.0
CORRIDOR_HALF_WIDTH_M = 0.9


def current_state(
    persons: list[dict[str, Any]],
) -> dict[str, Any]:
    minimum_distance = math.inf
    corridor = False
    valid = 0
    for person in persons:
        box = person["box"]
        x = float(box["cx"])
        y = float(box["cy"])
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        valid += 1
        minimum_distance = min(minimum_distance, math.hypot(x, y))
        if (
            0.0 < x <= CORRIDOR_FORWARD_LIMIT_M
            and abs(y) <= CORRIDOR_HALF_WIDTH_M
        ):
            corridor = True
    return {
        "valid_person_count": valid,
        "minimum_distance_m": (
            minimum_distance if math.isfinite(minimum_distance) else None
        ),
        "proximity_le_1_25m": (
            minimum_distance <= PROXIMITY_THRESHOLD_M
        ),
        "corridor_intrusion": corridor,
    }


def derive_row(
    record: dict[str, Any],
    persons: list[dict[str, Any]],
) -> dict[str, Any]:
    current = current_state(persons)
    future_proximity = bool(
        record["target"]["future_proximity_le_1_25m"]
    )
    future_corridor = bool(
        record["target"]["future_corridor_intrusion"]
    )
    current_proximity = bool(current["proximity_le_1_25m"])
    current_corridor = bool(current["corridor_intrusion"])
    return {
        "sample_id": record["sample_id"],
        "source_session_id": record["source_session_id"],
        "ancestry_group": record["ancestry_group"],
        "fold": int(record["fold"]),
        "anchor_frame_index": int(record["anchor_frame_index"]),
        "history_frame_indices": record["history_frame_indices"],
        "history_image_paths": record["history_image_paths"],
        "history_image_sha256": record["history_image_sha256"],
        "source": record["source"],
        "current_state": current,
        "future_ever_state": {
            "proximity_le_1_25m": future_proximity,
            "corridor_intrusion": future_corridor,
        },
        "transition_target": {
            "proximity_eligible": not current_proximity,
            "proximity_onset": (
                not current_proximity and future_proximity
            ),
            "proximity_clearance": (
                current_proximity and not future_proximity
            ),
            "corridor_eligible": not current_corridor,
            "corridor_onset": (
                not current_corridor and future_corridor
            ),
            "corridor_clearance": (
                current_corridor and not future_corridor
            ),
        },
    }


def counts(
    rows: list[dict[str, Any]],
    target: str,
) -> dict[str, int]:
    eligible_key = f"{target}_eligible"
    onset_key = f"{target}_onset"
    clearance_key = f"{target}_clearance"
    eligible = [
        row for row in rows
        if row["transition_target"][eligible_key]
    ]
    positive = sum(
        int(row["transition_target"][onset_key])
        for row in eligible
    )
    return {
        "eligible": len(eligible),
        "onset_positive": positive,
        "onset_negative": len(eligible) - positive,
        "clearance": sum(
            int(row["transition_target"][clearance_key])
            for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("Refusing to overwrite D15 JRDB onset output")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 104:
        raise ValueError("Expected 104 JRDB D9 samples")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source"]["label_path"])].append(record)

    rows = []
    for label_path_value, source_records in sorted(grouped.items()):
        label_path = Path(label_path_value)
        expected_hashes = {
            str(record["source"]["label_sha256"])
            for record in source_records
        }
        if expected_hashes != {sha256(label_path)}:
            raise ValueError(f"JRDB label binding mismatch: {label_path}")
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        labels = {
            int(key.split(".")[0]): value
            for key, value in payload["labels"].items()
        }
        rows.extend(
            derive_row(
                record,
                labels.get(int(record["anchor_frame_index"]), []),
            )
            for record in source_records
        )
    rows.sort(key=lambda row: row["sample_id"])

    by_sequence = []
    for source in sorted({row["source_session_id"] for row in rows}):
        source_rows = [
            row for row in rows if row["source_session_id"] == source
        ]
        by_sequence.append(
            {
                "source_session_id": source,
                "fold": source_rows[0]["fold"],
                "samples": len(source_rows),
                "proximity": counts(source_rows, "proximity"),
                "corridor": counts(source_rows, "corridor"),
            }
        )
    by_fold = []
    for fold in (0, 1):
        fold_rows = [row for row in rows if row["fold"] == fold]
        by_fold.append(
            {
                "fold": fold,
                "samples": len(fold_rows),
                "source_sessions": sorted(
                    {row["source_session_id"] for row in fold_rows}
                ),
                "proximity": counts(fold_rows, "proximity"),
                "corridor": counts(fold_rows, "corridor"),
            }
        )
    corridor_ready = all(
        fold["corridor"]["onset_positive"] > 0
        and fold["corridor"]["onset_negative"] > 0
        for fold in by_fold
    )
    proximity_ready = all(
        fold["proximity"]["onset_positive"] > 0
        and fold["proximity"]["onset_negative"] > 0
        for fold in by_fold
    )
    status = (
        "D15_JRDB_CORRIDOR_FUTURE_ONSET_TWO_FOLD_READY"
        if corridor_ready
        else "D15_JRDB_CORRIDOR_FUTURE_ONSET_NOT_TWO_FOLD_READY"
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
            "role": "Development independent-dataset transition census",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "label_file_count": len(grouped),
        },
        "definition": {
            "future_window": "JRDB D9 frames anchor+1 through anchor+30",
            "onset_eligible": "anchor-frame current state is safe",
            "onset_positive": (
                "current state is safe and future-ever state is risky"
            ),
            "clearance": (
                "current state is risky and every frame in the future "
                "window is safe"
            ),
            "primary_target": "corridor onset",
            "negative_control": "proximity onset",
        },
        "counts": {
            "samples": len(rows),
            "source_sessions": len(
                {row["source_session_id"] for row in rows}
            ),
            "totals": {
                target: counts(rows, target)
                for target in ("proximity", "corridor")
            },
            "by_fold": by_fold,
            "by_sequence": by_sequence,
        },
        "readiness": {
            "corridor_two_fold": corridor_ready,
            "proximity_two_fold": proximity_ready,
        },
        "outputs": {
            "samples_path": str(samples_path.resolve()),
            "samples_sha256": sha256(samples_path),
        },
        "next_action": (
            "run the frozen equal-capacity current/history corridor "
            "replication on the two fixed source-pair folds"
            if corridor_ready
            else (
                "do not train on this bounded JRDB corpus; acquire an "
                "independent onset-rich source"
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
                "readiness": report["readiness"],
                "totals": report["counts"]["totals"],
                "by_fold": by_fold,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
