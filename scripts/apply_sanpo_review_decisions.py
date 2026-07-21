from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_decisions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    defaults = payload.get("defaults", {})
    return [{**defaults, **frame} for frame in payload["frames"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply provenance-marked review decisions to a SANPO review CSV.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--decisions", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    csv_path = root / "qa" / "model_review_checklist.csv"
    decisions_path = Path(args.decisions).resolve()
    decisions = load_decisions(decisions_path)
    by_frame: dict[int, dict[str, Any]] = {}
    for decision in decisions:
        frame_index = int(decision["frame_index"])
        if frame_index in by_frame:
            raise SystemExit(f"Duplicate decision frame_index: {frame_index}")
        by_frame[frame_index] = decision
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or not fieldnames:
        raise SystemExit(f"Review checklist is empty or has no header: {csv_path}")
    expected_frames = {int(row["frame_index"]) for row in rows}
    if set(by_frame) != expected_frames:
        missing = sorted(expected_frames - set(by_frame))
        extra = sorted(set(by_frame) - expected_frames)
        raise SystemExit(f"Decision coverage mismatch: missing={missing} extra={extra}")
    protected = {"id", "sequence_id", "frame_index", "source_frame_index", "source_annotation_quality"}
    for row in rows:
        decision = by_frame[int(row["frame_index"])]
        for key, value in decision.items():
            if key == "frame_index":
                continue
            if key in protected:
                raise SystemExit(f"Decision cannot overwrite protected field: {key}")
            if key not in row:
                fieldnames.append(key)
            row[key] = "" if value is None else str(value)
    fieldnames = list(dict.fromkeys(fieldnames))
    temp = csv_path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(csv_path)
    summary = {
        "ok": True,
        "row_count": len(rows),
        "decision_file": str(decisions_path),
        "review_status_counts": {},
        "reviewer_type_counts": {},
    }
    for row in rows:
        for key, target in (("review_status", "review_status_counts"), ("reviewer_type", "reviewer_type_counts")):
            value = row.get(key, "")
            summary[target][value] = summary[target].get(value, 0) + 1
    (root / "qa" / "review_application_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"review_apply_ok=true rows={len(rows)} csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
