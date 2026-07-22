from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for source in payload["sources"]:
        for window in source["windows"]:
            if window["window_type"] == "negative":
                rows.extend(window["frames"])
    model = YOLO(str(args.model))
    annotations: list[dict] = []
    # Passing the complete path list makes Ultralytics stack every image before
    # inference. Chunk explicitly so the annotation run has a bounded VRAM peak.
    for offset in range(0, len(rows), args.batch):
        chunk = rows[offset : offset + args.batch]
        results = model.predict(
            source=[row["image_path"] for row in chunk],
            imgsz=args.imgsz,
            conf=args.confidence,
            classes=[0],
            device=0,
            batch=args.batch,
            verbose=False,
        )
        for row, result in zip(chunk, results, strict=True):
            boxes = []
            if result.boxes is not None:
                for xyxy, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True):
                    boxes.append({
                        "bbox_xyxy": [round(float(value), 3) for value in xyxy],
                        "proposal_confidence": round(float(confidence), 6),
                    })
            annotations.append({
                "frame_id": row["frame_id"],
                "image_sha256": row["image_sha256"],
                "person_proposals": boxes,
            })
        if len(annotations) % 100 < args.batch:
            print(f"negative_annotation_frames={len(annotations)}/{len(rows)}", flush=True)
    output = {
        "schema": "blindassist_ustrf_negative_person_annotation_yolo11x_v1",
        "authority": "annotation_proposal_only_never_detector_candidate_or_promotion_credit",
        "candidate_sha256": sha256(args.candidate),
        "model_path": str(args.model),
        "model_sha256": sha256(args.model),
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "frame_count": len(annotations),
        "frames": annotations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"frame_count": len(annotations), "output": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
