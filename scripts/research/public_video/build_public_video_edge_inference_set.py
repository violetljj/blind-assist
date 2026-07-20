#!/usr/bin/env python3
"""Materialize only silver-label evidence frames for on-device inference.

No event label is copied into the device asset.  The produced set binds the
event report back to the GPT/VLM silver manifest without making that manifest
an Android-test oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from validate_public_video_silver_labels import SilverLabelError, load_json, validate


class InferenceSetError(ValueError):
    """The source frames cannot be prepared for an inference-only device run."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    silver_manifest: dict[str, Any],
    source_manifest: dict[str, Any],
    *,
    silver_path: Path,
    source_path: Path,
    source_images_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    validate(silver_manifest, source_manifest_path=source_path)
    if output_root.exists():
        raise InferenceSetError(f"refusing to overwrite output root: {output_root}")
    if not source_images_dir.is_dir():
        raise InferenceSetError(f"source image directory is missing: {source_images_dir}")
    frames = source_manifest.get("frames")
    if not isinstance(frames, list):
        raise InferenceSetError("source manifest frames must be a list")
    # Repeated source frames are valid in public recordings (the source receipt
    # can contain duplicate PNG payloads).  A silver manifest binds pixels by
    # hash, so choose the first attested occurrence deterministically.
    source_by_hash: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("sha256"), str) or not isinstance(frame.get("file_name"), str):
            raise InferenceSetError(f"source manifest frames[{index}] requires sha256 and file_name")
        source_by_hash.setdefault(frame["sha256"], frame)
    output_images = output_root / "images"
    rows: list[dict[str, Any]] = []
    for episode_order, episode in enumerate(silver_manifest["episodes"]):
        episode_id = episode["episode_id"]
        for evidence_order, image_hash in enumerate(episode["evidence_frame_sha256"]):
            source = source_by_hash.get(image_hash)
            if source is None:
                raise InferenceSetError(f"silver evidence hash is absent from source: {image_hash}")
            original_name = source["file_name"]
            source_file = source_images_dir / original_name
            if not source_file.is_file() or sha256_file(source_file) != image_hash:
                raise InferenceSetError(f"source image does not match its manifest hash: {source_file}")
            destination_name = f"{episode_order:02d}_{evidence_order:02d}_{original_name}"
            destination = output_images / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            rows.append({
                "episode_id": episode_id,
                "episode_order": episode_order,
                "evidence_order": evidence_order,
                # Keep both indices: `frame_index` addresses the sampled
                # timeline manifest, while `source_frame_index` (when
                # available) preserves the original recording's frame clock.
                "timeline_frame_index": source.get("frame_index"),
                "source_frame_index": source.get("source_frame_index", source.get("frame_index")),
                "image_path": f"images/{destination_name}",
                "image_sha256": image_hash,
            })
    spec = {
        "schema": "blindassist_public_video_edge_inference_set_v1",
        "source_id": silver_manifest["source"]["source_id"],
        "source_manifest_sha256": sha256_file(source_path),
        "silver_manifest_sha256": sha256_file(silver_path),
        "frame_count": len(rows),
        "human_event_truth_present": False,
        "privacy_audit_required": True,
        "inference_only": True,
        "training_execution_authorized": False,
        "production_model_replacement_authorized": False,
        "important_limit": "The Android run receives pixels and episode membership only; GPT/VLM verdicts remain outside the device test and are compared afterwards.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output_root / "dataset_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "output_root": str(output_root.resolve()), "frame_count": len(rows), "training_execution_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-images-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(
            load_json(args.silver_manifest), load_json(args.source_manifest),
            silver_path=args.silver_manifest, source_path=args.source_manifest,
            source_images_dir=args.source_images_dir, output_root=args.output_root,
        )
        print(json.dumps(result, ensure_ascii=False))
    except (InferenceSetError, SilverLabelError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
