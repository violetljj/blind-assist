#!/usr/bin/env python3
"""Build a local-only dense-annotation queue from accepted 50-frame drafts.

The queue is intentionally not a dataset manifest: it names the four semantic
classes to annotate, but contains no semantic-mask path and cannot be passed to
training.  Only draft roots with an accepted geometry report and a hash-bound
model review are admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SEMANTIC_CLASSES = {
    0: "walkable",
    1: "boundary_step_curb",
    2: "obstacle",
    3: "unknown_nonwalkable",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def queue_item(draft_root: Path) -> dict[str, Any]:
    root = draft_root.resolve()
    manifest = root / "manifest.draft.jsonl"
    selection_path = root / "qa" / "selection_evidence.json"
    review_path = root / "qa" / "model_review_result.json"
    draft_rows = sorted(rows(manifest), key=lambda item: int(item["frame_index"]))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    manifest_sha = sha256(manifest)
    if len(draft_rows) != 50 or [int(item["frame_index"]) for item in draft_rows] != list(range(50)):
        raise ValueError(f"{root}: requires contiguous 50-frame draft")
    if selection.get("draft_manifest_sha256") != manifest_sha or selection.get("decision") != "accept_for_model_review":
        raise ValueError(f"{root}: geometry selection is not accepted or not bound to draft")
    if review.get("draft_manifest_sha256") != manifest_sha or review.get("selection_evidence_sha256") != sha256(selection_path):
        raise ValueError(f"{root}: model review is not bound to current selection/draft")
    if not review.get("ok") or review.get("promotion") != "dense_annotation_queue":
        raise ValueError(f"{root}: model review did not promote to dense annotation")
    response = review["response"]
    return {
        "format": "blindassist_sanpo_v3_dense_annotation_queue_v1",
        "annotation_status": "queued_not_labeled",
        "training_status": "forbidden_until_masks_and_v3_gate_pass",
        "sequence_id": draft_rows[0]["sequence_id"],
        "session_id": draft_rows[0]["source"]["session_id"],
        "scene_bucket": response["primary_scene_bucket"],
        "expected_alert_outcome": response["expected_alert_outcome"],
        "semantic_class_ids": SEMANTIC_CLASSES,
        "draft_manifest": str(manifest),
        "draft_manifest_sha256": manifest_sha,
        "selection_evidence": str(selection_path),
        "selection_evidence_sha256": sha256(selection_path),
        "model_review": str(review_path),
        "model_review_sha256": sha256(review_path),
        "selection_summary": selection["summary"],
        "frame_count": 50,
        "frames": [{
            "id": item["id"],
            "frame_index": item["frame_index"],
            "image_path": item["image_path"],
            "source_image_sha256": item["source"]["sha256"],
            "source_mask_sha256": item["source"]["mask_sha256"],
        } for item in draft_rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = [queue_item(path) for path in args.draft_root]
    sequence_ids = [item["sequence_id"] for item in items]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("annotation queue contains a duplicate sequence")
    payload = {
        "format": "blindassist_sanpo_v3_dense_annotation_queue_v1",
        "queue_contract": "Queue entries are proposals only. Annotators create a single 0..3 semantic mask per RGB; no trainer may consume this file.",
        "semantic_class_ids": SEMANTIC_CLASSES,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"queued_sequences": len(items), "queued_frames": sum(item["frame_count"] for item in items), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
