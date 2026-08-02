#!/usr/bin/env python3
"""Materialize THOR future-onset targets that cannot be solved by current risk."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_stage_c_d11_thor_magni_kinematic_information_ceiling import (
    score_sample,
)
from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    DEFAULT_SAMPLES,
    load_jsonl,
    sha256,
)
from materialize_stage_c_d8_thor_magni_local_route_supervision import (
    PROXIMITY_THRESHOLD_M,
    read_scenario,
)


SCHEMA = "blindassist_hftf_stage_c_d12_thor_magni_future_onset_v0"


def derive_onset_row(
    record: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    current_proximity = (
        float(diagnostic["current_static"]["proximity"])
        >= -PROXIMITY_THRESHOLD_M
    )
    current_corridor = (
        float(diagnostic["current_static"]["corridor"]) >= 0.0
    )
    future_proximity = bool(
        record["target"]["future_proximity_le_1_25m"]
    )
    future_corridor = bool(
        record["target"]["future_corridor_intrusion"]
    )
    return {
        "sample_id": record["sample_id"],
        "source_session_id": record["source_session_id"],
        "ancestry_group": record["ancestry_group"],
        "fold": int(record["fold"]),
        "video_path": record["video_path"],
        "video_sha256": record["video_sha256"],
        "anchor_scene_frame": int(record["anchor_scene_frame"]),
        "history_scene_frames": record["history_scene_frames"],
        "current_state": {
            "proximity_le_1_25m": current_proximity,
            "corridor_intrusion": current_corridor,
        },
        "future_ever_state": {
            "proximity_le_1_25m": future_proximity,
            "corridor_intrusion": future_corridor,
        },
        "future_onset_target": {
            "proximity_eligible": not current_proximity,
            "proximity_onset": (
                not current_proximity and future_proximity
            ),
            "corridor_eligible": not current_corridor,
            "corridor_onset": (
                not current_corridor and future_corridor
            ),
        },
    }


def target_counts(
    rows: list[dict[str, Any]],
    target: str,
) -> dict[str, int]:
    eligible_key = f"{target}_eligible"
    onset_key = f"{target}_onset"
    eligible = [
        row for row in rows
        if row["future_onset_target"][eligible_key]
    ]
    positive = sum(
        int(row["future_onset_target"][onset_key])
        for row in eligible
    )
    return {
        "eligible": len(eligible),
        "positive": positive,
        "negative": len(eligible) - positive,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    samples_output = args.output_root / "samples.jsonl"
    report_output = args.output_root / "report.json"
    if (
        args.output_root.exists()
        or samples_output.exists()
        or report_output.exists()
    ):
        raise ValueError("Refusing to overwrite D12 onset materialization")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 1078:
        raise ValueError("Expected 1,078 THOR samples")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["scenario_csv_path"]),
                str(record["camera_body"]),
            )
        ].append(record)

    rows = []
    for (scenario_path, camera_body), session_records in sorted(
        grouped.items()
    ):
        path = Path(scenario_path)
        expected_hashes = {
            str(record["scenario_csv_sha256"])
            for record in session_records
        }
        if expected_hashes != {sha256(path)}:
            raise ValueError(f"Scenario binding mismatch: {path}")
        data = read_scenario(
            path,
            camera_body,
            f"{camera_body} PPL_SceneFNr",
        )
        for record in session_records:
            rows.append(
                derive_onset_row(
                    record,
                    score_sample(record, data),
                )
            )
    rows.sort(key=lambda row: row["sample_id"])

    monotonicity_violations = {
        "proximity": sum(
            int(
                row["current_state"]["proximity_le_1_25m"]
                and not row["future_ever_state"]["proximity_le_1_25m"]
            )
            for row in rows
        ),
        "corridor": sum(
            int(
                row["current_state"]["corridor_intrusion"]
                and not row["future_ever_state"]["corridor_intrusion"]
            )
            for row in rows
        ),
    }
    if any(monotonicity_violations.values()):
        raise ValueError(
            f"Future-ever monotonicity violation: {monotonicity_violations}"
        )
    by_fold = []
    for fold in range(5):
        fold_rows = [row for row in rows if row["fold"] == fold]
        by_fold.append(
            {
                "fold": fold,
                "samples": len(fold_rows),
                "source_sessions": len(
                    {
                        row["source_session_id"]
                        for row in fold_rows
                    }
                ),
                "proximity": target_counts(fold_rows, "proximity"),
                "corridor": target_counts(fold_rows, "corridor"),
            }
        )
    totals = {
        target: target_counts(rows, target)
        for target in ("proximity", "corridor")
    }
    five_fold_ready = all(
        fold_row[target]["positive"] > 0
        and fold_row[target]["negative"] > 0
        for fold_row in by_fold
        for target in ("proximity", "corridor")
    )
    status = (
        "D12_FUTURE_ONSET_TARGET_FIVE_FOLD_READY"
        if five_fold_ready
        else "D12_FUTURE_ONSET_TARGET_NOT_FIVE_FOLD_READY"
    )

    args.output_root.mkdir(parents=True)
    with samples_output.open("x", encoding="utf-8", newline="\n") as output:
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
            "role": "Development target repair and opportunity census",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "scenario_csv_count": len(grouped),
        },
        "definition": {
            "eligible": "current target state is false",
            "positive": (
                "current target state is false and the original 0-2 second "
                "future-ever target is true"
            ),
            "negative": (
                "current target state is false and the original 0-2 second "
                "future-ever target is false"
            ),
            "purpose": (
                "remove targets already positive at t=0 so a future-onset "
                "model cannot succeed by reproducing current occupancy"
            ),
        },
        "counts": {
            "samples": len(rows),
            "source_sessions": len(
                {row["source_session_id"] for row in rows}
            ),
            "monotonicity_violations": monotonicity_violations,
            "totals": totals,
            "by_fold": by_fold,
        },
        "outputs": {
            "samples_path": str(samples_output.resolve()),
            "samples_sha256": sha256(samples_output),
        },
        "next_action": (
            "compare current-only and true RGB history on onset-only targets"
            if five_fold_ready
            else (
                "increase source diversity or redesign folds before "
                "training; do not collapse onset back into current risk"
            )
        ),
    }
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(report_output)
    Path(str(report_output) + ".sha256").write_text(
        f"{digest}  {report_output.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "totals": totals,
                "by_fold": by_fold,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
