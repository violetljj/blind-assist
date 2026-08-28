"""Freeze an algorithm-fresh cohort for the fixed C11 probability layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dtr_c1_global_obb_cohort_admission import ROSTER_SCHEMA, require, sha256_file, write_json
from dtr_c10_fresh_confirmation_admission import _meets, _select, _totals
from dtr_c11_route_region_probability import SCHEMA as CALIBRATOR_SCHEMA


SCHEMA = "blindassist-dtr-c11-fresh-confirmation-admission-v1"
STATUS = "DTR_C11_FRESH_CONFIRMATION_COHORT_ADMITTED_METADATA_ONLY"
ROSTER_STATUS = "DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY"


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.c1_result.resolve(strict=True)
    c10_path = args.c10_roster.resolve(strict=True)
    calibrator_path = args.calibrator.resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    c10 = json.loads(c10_path.read_text(encoding="utf-8"))
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    require(
        source.get("schema") == "blindassist-dtr-c1-global-obb-cohort-admission-v1",
        "c1_result_schema_drift",
    )
    require(c10.get("schema") == ROSTER_SCHEMA, "c10_roster_schema_drift")
    require(calibrator.get("schema") == CALIBRATOR_SCHEMA, "calibrator_schema_drift")
    consumed = set(str(value) for value in source["selected_sequence_names"])
    consumed.update(str(value) for value in source["source"]["excluded_consumed_sequences"])
    consumed.update(str(row["sequence"]) for row in c10["selected_sequences"])
    candidates = [
        row for row in source["sequence_scan"] if str(row["sequence"]) not in consumed
    ]
    preferred = source["admission_policy"]["preferred"]
    selected, combinations_considered = _select(candidates, preferred)
    totals = _totals(selected)
    require(_meets(totals, preferred), "selected_gate_not_met")
    roster = {
        "schema": ROSTER_SCHEMA,
        "status": ROSTER_STATUS,
        "claim_ceiling": "METADATA_ONLY_C11_FRESH_CONFIRMATION_NO_RAW_OR_ALGORITHM_RESULT",
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
        "selected_totals": totals,
        "frozen_algorithm": {
            "calibrator": str(calibrator_path),
            "calibrator_sha256": sha256_file(calibrator_path),
            "decision_probability": calibrator["model"]["decision_probability"],
        },
        "source_authority": {
            "labels_archive_name": Path(source["source"]["labels"]).name,
            "labels_sha256": source["source"]["labels_sha256"],
            "timestamps_archive_name": Path(source["source"]["timestamps"]).name,
            "timestamps_sha256": source["source"]["timestamps_sha256"],
            "c1_result": str(source_path),
            "c1_result_sha256": sha256_file(source_path),
        },
        "forbidden": [
            "changing calibrator coefficients, probability threshold, feature, route, or lifecycle after raw acquisition",
            "opening selected native future OBB labels before combined C9/C11 predictions are hash sealed",
            "adding any algorithm-exposed sequence",
            "treating public replay probability as product or safety calibration",
        ],
    }
    roster_path = args.roster.resolve()
    write_json(roster_path, roster)
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "selected_sequences": [str(row["sequence"]) for row in selected],
        "selected_totals": totals,
        "selection_policy": roster["selection_policy"],
        "artifacts": {
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "calibrator": str(calibrator_path),
            "calibrator_sha256": sha256_file(calibrator_path),
        },
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c1-result",
        type=Path,
        default=repo / "artifacts.local" / "evidence" / "dtr-c1" / "global-obb-cohort-admission" / "result.json",
    )
    parser.add_argument(
        "--c10-roster",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c10_fresh_confirmation_roster.json"),
    )
    parser.add_argument(
        "--calibrator",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c11_route_region_calibrator.json"),
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c11_fresh_confirmation_roster.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "artifacts.local" / "evidence" / "dtr-c11" / "fresh-confirmation-admission" / "result.json",
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "selected": result["selected_sequences"], "totals": result["selected_totals"]}))


if __name__ == "__main__":
    main()
