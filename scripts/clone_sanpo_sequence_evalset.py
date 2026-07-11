#!/usr/bin/env python3
"""Clone a finalized SANPO sequence into a new draft root for a revised review decision."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


RESET_FIELDS = {
    "primary_object_id": None,
    "source_primary_region_id": None,
    "expected_risk_direction": None,
    "expected_distance_band": None,
    "expected_should_alert": None,
    "expected_risk_level": None,
    "expected_approach_state": None,
    "expected_approach_alert": None,
    "expected_time_to_alert_frames": None,
    "expected_event_phase": None,
    "scene_bucket": None,
    "risk_event_id": None,
    "review_status": "pending_manual_risk_review",
    "status": "pending_review",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if (output / "manifest.jsonl").exists() or (output / "manifest.draft.jsonl").exists():
        raise SystemExit("Refusing to overwrite an existing dataset root")
    source_manifest = source / "manifest.jsonl"
    if not source_manifest.is_file():
        raise SystemExit("source-root must be finalized")
    rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row.update(RESET_FIELDS)
        attributes = row.get("attributes")
        if isinstance(attributes, dict):
            attributes.pop("scene_bucket", None)
            attributes.pop("risk_event_id", None)
        row.pop("review_provenance", None)
        row.pop("review_notes", None)
        row["objects_review_status"] = "pending" if row.get("objects") else "not_applicable"
    for name in ("images", "source_masks"):
        shutil.copytree(source / name, output / name)
    for name in (
        "dataset_spec.json", "source_session_description.json", "source_labelmap.json",
        "source_annotation_types.json", "source_licenses.md", "qa/manual_review_checklist.csv",
    ):
        source_file = source / name
        if source_file.exists():
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, output / name)
    (output / "manifest.draft.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"cloned_draft={output} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
