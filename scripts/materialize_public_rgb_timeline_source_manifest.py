#!/usr/bin/env python3
"""Turn an RGB-only public timeline into a hash-attested silver-label source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class TimelineSourceError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TimelineSourceError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise TimelineSourceError("timeline manifest must contain JSON objects")
    return rows


def materialize(root: Path, timeline: list[dict[str, Any]], candidate_spec: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise TimelineSourceError(f"refusing to overwrite source manifest: {output}")
    format_version = candidate_spec.get("format")
    if format_version not in {"blindassist_sanpo_rgb_timeline_candidate_v1", "blindassist_sanpo_rgb_timeline_candidate_v2"}:
        raise TimelineSourceError("unexpected RGB timeline candidate spec")
    if candidate_spec.get("human_event_truth_present") is not False:
        raise TimelineSourceError("candidate spec must not claim human event truth")
    provisional_training_authorized = format_version == "blindassist_sanpo_rgb_timeline_candidate_v2"
    if provisional_training_authorized:
        if candidate_spec.get("provisional_training_authorized") is not True:
            raise TimelineSourceError("v2 candidate spec must authorize provisional training")
    elif candidate_spec.get("training_execution_authorized") is not False:
        raise TimelineSourceError("v1 candidate spec breaks its weak-source boundary")
    source = candidate_spec.get("source")
    if not isinstance(source, dict) or source.get("license") != "CC-BY-4.0":
        raise TimelineSourceError("expected a CC-BY SANPO RGB-only source")
    frames: list[dict[str, Any]] = []
    for index, row in enumerate(timeline):
        relative = row.get("image_path")
        expected_hash = row.get("image_sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise TimelineSourceError(f"timeline row {index} lacks image path or SHA256")
        image = root / relative
        if not image.is_file() or sha256_file(image) != expected_hash:
            raise TimelineSourceError(f"timeline image does not match SHA256: {image}")
        frames.append({
            "frame_index": row.get("timeline_index"),
            "source_frame_index": row.get("source_frame_index"),
            "file_name": image.name,
            "sha256": expected_hash,
        })
    result = {
        "format": "blindassist_public_rgb_timeline_source_manifest_v2" if provisional_training_authorized else "blindassist_public_rgb_timeline_source_manifest_v1",
        "source_id": f"sanpo_real_{source['session_id']}",
        "source": source,
        "frame_count": len(frames),
        "frames": frames,
        "privacy_audit_required": True,
        "human_event_truth_present": False,
        "source_masks_or_geometry_used": False,
        "provisional_training_authorized": provisional_training_authorized,
        "training_execution_authorized": provisional_training_authorized,
        "production_model_replacement_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = materialize(
            args.timeline_root,
            load_jsonl(args.timeline_root / "manifest.rgb_timeline.jsonl"),
            load_json(args.timeline_root / "candidate_spec.json"),
            args.output,
        )
        print(json.dumps({"ok": True, "frame_count": result["frame_count"], "provisional_training_authorized": result["provisional_training_authorized"]}, ensure_ascii=False))
    except (TimelineSourceError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
