from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    by_hash = {row["image_sha256"]: row for row in annotation["frames"]}
    merged = 0
    for source in candidate["sources"]:
        for window in source["windows"]:
            if window["window_type"] != "negative":
                continue
            for row in window["frames"]:
                proposal = by_hash.get(row["image_sha256"])
                if proposal is None or proposal["frame_id"] != row["frame_id"]:
                    raise ValueError(f"missing or misbound annotation: {window['window_id']}/{row['frame_id']}")
                row["yoloe_person_candidates"] = row["all_person_candidates"]
                row["all_person_candidates"] = proposal["person_proposals"]
                row["confirmed_absent_candidate"] = not proposal["person_proposals"]
                merged += 1
    if merged != annotation["frame_count"] or len(by_hash) != annotation["frame_count"]:
        raise ValueError("annotation frame coverage mismatch")
    candidate["schema"] = "blindassist_ustrf_detector_target_truth_dual_annotation_candidate_v1"
    candidate["authority"] = "candidate_only_requires_visual_review_before_truth_freeze"
    candidate["negative_annotation"] = {
        "primary": "yolo11x_coco_person_annotation_proposal",
        "secondary": "yoloe_11s_seg_person_prompt_annotation_proposal",
        "annotation_sha256": sha256(args.annotation),
        "both_are_never_detector_candidate_or_promotion_credit": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"merged_frames": merged, "output": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
