"""Freeze the smallest unscored JRDB cohort meeting the existing C1 gate.

This admission reads only the already sealed C1 truth metadata.  It excludes
every sequence previously exposed to a DTR algorithm, then chooses the
minimum-cardinality subset that meets the unchanged preferred event and
non-CONTACT denominators.  Ties minimize total frames and then use the
lexicographic sequence tuple.  No raw sensor or algorithm output is read.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dtr_c1_global_obb_cohort_admission import ROSTER_SCHEMA, require, sha256_file, write_json


SCHEMA = "blindassist-dtr-c10-fresh-confirmation-admission-v1"
STATUS = "DTR_C10_FRESH_CONFIRMATION_COHORT_ADMITTED_METADATA_ONLY"
ROSTER_STATUS = "DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY"


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "sequences": len(rows),
        "frames": sum(int(row["frames"]) for row in rows),
        "bounded_contact_events": sum(int(row["bounded_contact_events"]) for row in rows),
        "unique_responsible_events": sum(int(row["unique_responsible_events"]) for row in rows),
        "known_non_contact_s": sum(float(row["known_non_contact_s"]) for row in rows),
        "timeline_duration_s": sum(float(row["timeline_duration_s"]) for row in rows),
    }


def _meets(totals: Mapping[str, Any], preferred: Mapping[str, Any]) -> bool:
    return (
        int(totals["bounded_contact_events"]) >= int(preferred["bounded_contact_events"])
        and int(totals["unique_responsible_events"]) >= int(preferred["unique_responsible_events"])
        and float(totals["known_non_contact_s"]) >= float(preferred["known_non_contact_s"])
    )


def _select(
    rows: Sequence[Mapping[str, Any]], preferred: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], int]:
    ordered = sorted(rows, key=lambda row: str(row["sequence"]))
    combinations_considered = 0
    for cardinality in range(1, len(ordered) + 1):
        eligible = []
        for combination in itertools.combinations(ordered, cardinality):
            combinations_considered += 1
            totals = _totals(combination)
            if _meets(totals, preferred):
                eligible.append(combination)
        if eligible:
            selected = min(
                eligible,
                key=lambda values: (
                    sum(int(row["frames"]) for row in values),
                    tuple(str(row["sequence"]) for row in values),
                ),
            )
            return list(selected), combinations_considered
    raise RuntimeError("no_unscored_subset_meets_preferred_gate")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.c1_result.resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    require(
        source.get("schema") == "blindassist-dtr-c1-global-obb-cohort-admission-v1",
        "c1_result_schema_drift",
    )
    consumed = set(str(value) for value in source["selected_sequence_names"])
    consumed.update(str(value) for value in source["source"]["excluded_consumed_sequences"])
    candidates = [
        row for row in source["sequence_scan"] if str(row["sequence"]) not in consumed
    ]
    preferred = source["admission_policy"]["preferred"]
    selected, combinations_considered = _select(candidates, preferred)
    selected_totals = _totals(selected)
    require(_meets(selected_totals, preferred), "selected_gate_not_met")
    roster = {
        "schema": ROSTER_SCHEMA,
        "status": ROSTER_STATUS,
        "claim_ceiling": "METADATA_ONLY_FRESH_CONFIRMATION_COHORT_NO_ALGORITHM_RESULT",
        "dataset": "JRDB public train split",
        "contract": source["contract"],
        "selection_policy": {
            "algorithm_consumed_sequences_excluded": sorted(consumed),
            "gate": preferred,
            "ordering": (
                "minimum sequence cardinality meeting unchanged preferred gate; "
                "then minimum total frames; then lexicographic sequence tuple"
            ),
            "combinations_considered": combinations_considered,
            "candidate_sequences": len(candidates),
        },
        "selected_sequences": [
            {
                "sequence": str(row["sequence"]),
                "first_frame": int(row["first_frame"]),
                "last_frame": int(row["last_frame"]),
                "frames": int(row["frames"]),
                "timeline_duration_s": float(row["timeline_duration_s"]),
                "known_non_contact_s": float(row["known_non_contact_s"]),
                "bounded_contact_events": int(row["bounded_contact_events"]),
                "unique_responsible_events": int(row["unique_responsible_events"]),
                "bounded_contact_event_details": list(row["events"]),
            }
            for row in selected
        ],
        "selected_totals": selected_totals,
        "source_authority": {
            "labels_archive_name": Path(source["source"]["labels"]).name,
            "labels_sha256": source["source"]["labels_sha256"],
            "timestamps_archive_name": Path(source["source"]["timestamps"]).name,
            "timestamps_sha256": source["source"]["timestamps_sha256"],
            "c1_result": str(source_path),
            "c1_result_sha256": sha256_file(source_path),
        },
        "forbidden": [
            "changing C9, route thresholds, lifecycle, confidence, or cohort after raw-sensor acquisition",
            "using these future OBB labels before the C9 prediction artifact is hash sealed",
            "adding an algorithm-exposed sequence",
            "treating confirmation as product or safety evidence",
        ],
    }
    output_path = args.roster.resolve()
    write_json(output_path, roster)
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": "Can C9 receive an algorithm-fresh preferred JRDB confirmation cohort without changing its mechanism?",
        "selected_sequences": [str(row["sequence"]) for row in selected],
        "selected_totals": selected_totals,
        "selection_policy": roster["selection_policy"],
        "source": {
            "c1_result": str(source_path),
            "c1_result_sha256": sha256_file(source_path),
        },
        "artifacts": {
            "roster": str(output_path),
            "roster_sha256": sha256_file(output_path),
        },
        "claim_limits": [
            "C10 admission reuses sealed truth metadata but the selected raw sequences have never been scored by a DTR algorithm.",
            "Selection minimizes acquisition/evaluation size under the unchanged gate; it does not optimize any algorithm metric.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c1-result",
        type=Path,
        default=repo
        / "artifacts.local"
        / "evidence"
        / "dtr-c1"
        / "global-obb-cohort-admission"
        / "result.json",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c10_fresh_confirmation_roster.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo
        / "artifacts.local"
        / "evidence"
        / "dtr-c10"
        / "fresh-confirmation-admission"
        / "result.json",
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "selected": result["selected_sequences"], "totals": result["selected_totals"]}))


if __name__ == "__main__":
    main()
