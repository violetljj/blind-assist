#!/usr/bin/env python3
"""Create an immutable SANPO benchmark clone with explicit risk-event phases."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def phase_for(row: dict) -> str:
    if row.get("expected_approach_state") == "RECEDING" and not row.get("expected_should_alert"):
        return "PASSED"
    if row.get("expected_should_alert"):
        return "APPROACHING"
    return "PASSED" if "passed" in row.get("review_notes", "").lower() else "APPROACHING"


def scene_bucket_for_sequence(rows: list[dict]) -> str:
    """Recover the reviewed v2 profile without relying on a mutable external checklist."""
    if not any(row.get("expected_should_alert") for row in rows):
        return "parallel_curb"
    primary_regions = {row.get("source_primary_region_id") for row in rows}
    if any(region and region.startswith("sanpo_15_") for region in primary_regions):
        return "front_stairs"
    return "center_obstacle"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output root: {output}")
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("source manifest is empty")
    output.mkdir(parents=True)
    for name in ("images", "source_masks", "qa"):
        if (source / name).exists():
            shutil.copytree(source / name, output / name)
    for name in ("dataset_spec.json", "source_session_description.json", "source_labelmap.json", "source_annotation_types.json", "source_licenses.md"):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    by_sequence: dict[str, list[dict]] = {}
    for row in rows:
        by_sequence.setdefault(str(row.get("sequence_id", "")), []).append(row)
    for sequence_id, sequence_rows in by_sequence.items():
        scene_bucket = scene_bucket_for_sequence(sequence_rows)
        for row in sequence_rows:
            row["expected_event_phase"] = phase_for(row)
            row["scene_bucket"] = scene_bucket
            row["risk_event_id"] = sequence_id or row["id"]
            attributes = dict(row.get("attributes") or {})
            attributes["scene_bucket"] = scene_bucket
            attributes["risk_event_id"] = sequence_id or row["id"]
            row["attributes"] = attributes
            row["event_phase_provenance"] = "derived_from_existing_reviewed_approach_and_alert_labels_v1"
            row["event_label_provenance"] = "derived_from_reviewed_v2_sequence_profile_v1"
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"event_phase_clone_ok=true rows={len(rows)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
