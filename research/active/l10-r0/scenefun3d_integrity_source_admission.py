from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _download_once(url: str, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _candidate_rows(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [
            (str(row["visit_id"]), str(row["video_id"]))
            for row in csv.DictReader(stream)
        ]
    return sorted(set(rows))


def admit_sources(
    protocol: dict[str, Any], cohort_csv: Path, download_root: Path
) -> dict[str, Any]:
    expected_csv_hash = protocol["source"]["cohort_csv_sha256"]
    actual_csv_hash = _sha256(cohort_csv)
    if actual_csv_hash != expected_csv_hash:
        raise ValueError(
            f"Cohort CSV hash mismatch: expected={expected_csv_hash}, actual={actual_csv_hash}"
        )

    selection = protocol["selection"]
    consumed = set(protocol["consumed_visit_ids"])
    maximum_candidates = int(selection["maximum_candidate_scenes"])
    required_scenes = int(selection["selected_scene_count"])
    minimum_descriptions = int(selection["minimum_descriptions_per_scene"])
    minimum_multi_target = int(selection["minimum_multi_target_tasks_per_scene"])
    base_url = protocol["source"]["official_base_url"].rstrip("/")

    scanned: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    failures = 0
    candidates_seen = 0
    for visit_id, video_id in _candidate_rows(cohort_csv):
        if visit_id in consumed:
            continue
        if candidates_seen >= maximum_candidates:
            break
        candidates_seen += 1
        output = download_root / visit_id / f"{visit_id}_descriptions.json"
        url = f"{base_url}/train/{visit_id}/{visit_id}_descriptions.json"
        try:
            _download_once(url, output)
            payload = _load_json(output)
            if str(payload.get("visit_id")) != visit_id:
                raise ValueError("visit_id mismatch in downloaded descriptions")
            descriptions = payload.get("descriptions", [])
            multi_target = sum(
                len(row.get("annot_id", [])) >= 2 for row in descriptions
            )
            eligible = (
                len(descriptions) >= minimum_descriptions
                and multi_target >= minimum_multi_target
            )
            row = {
                "visit_id": visit_id,
                "video_id": video_id,
                "descriptions": len(descriptions),
                "multi_target_tasks": multi_target,
                "eligible": eligible,
                "descriptions_sha256": _sha256(output),
                "url": url,
            }
            scanned.append(row)
            if eligible:
                selected.append(row)
                if len(selected) == required_scenes:
                    break
        except Exception as error:  # source availability is not algorithm evidence
            failures += 1
            scanned.append(
                {
                    "visit_id": visit_id,
                    "video_id": video_id,
                    "eligible": None,
                    "reason": "SOURCE_UNAVAILABLE_OR_INVALID",
                    "error_type": type(error).__name__,
                    "url": url,
                }
            )

    if len(selected) == required_scenes:
        decision = "SC23_SOURCE_ADMITTED"
    elif failures:
        decision = "SC23_SOURCE_ADMISSION_INCOMPLETE"
    else:
        decision = "SC23_NOT_EVALUABLE_INSUFFICIENT_ELIGIBLE_SCENES"
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "protocol_sha256": protocol["protocol_sha256"],
        "cohort_csv_sha256": actual_csv_hash,
        "selection": selection,
        "consumed_visit_ids": sorted(consumed),
        "denominators": {
            "candidate_scenes_scanned": len(scanned),
            "source_failures": failures,
            "eligible_scenes": len(selected),
            "required_scenes": required_scenes,
        },
        "selected": selected,
        "scanned": scanned,
        "authority_boundary": (
            "Admission uses only description count and annot_id list length. "
            "It does not inspect annot_id identity, geometry, candidates, selector output, or evaluator scores."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = _load_json(args.protocol)
    protocol["protocol_sha256"] = _sha256(args.protocol)
    result = admit_sources(
        protocol,
        args.cohort_csv.resolve(),
        args.download_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "denominators": result["denominators"],
                "selected": [
                    {
                        "visit_id": row["visit_id"],
                        "video_id": row["video_id"],
                        "descriptions": row["descriptions"],
                        "multi_target_tasks": row["multi_target_tasks"],
                    }
                    for row in result["selected"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
