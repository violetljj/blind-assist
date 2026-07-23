from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from contract import load_json, sha256_file, validate_prereg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite adjudicator output: {args.output}")

    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    adjudicator = config["seen_truth_proposal_protocol"]["fusion"]["third_model_adjudicator"]
    expected_output = repo / adjudicator["planned_output_path"]
    if args.output.resolve() != expected_output.resolve():
        raise ValueError("adjudicator output path differs from preregistration")
    bundle_path = repo / adjudicator["planned_bundle_path"]
    bundle = load_json(bundle_path)
    if bundle.get("candidate_alerts_exposed") is not False or bundle.get("baseline_app_detector_outputs_exposed") is not False:
        raise ValueError("adjudication bundle exposed forbidden detector evidence")

    model_path = Path(adjudicator["model_path"])
    model = YOLO(str(model_path))
    rows = bundle["frames"]
    annotations = []
    for offset in range(0, len(rows), args.batch):
        chunk = rows[offset:offset + args.batch]
        results = model.predict(
            source=[str(repo / row["image_path"]) for row in chunk],
            imgsz=int(adjudicator["imgsz"]),
            conf=float(adjudicator["confidence"]),
            classes=[int(adjudicator["class_id"])],
            device=args.device,
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
                "source_id": row["source_id"],
                "blind_window_id": row["blind_window_id"],
                "frame_id": row["frame_id"],
                "image_sha256": row["image_sha256"],
                "person_proposals": boxes,
            })
        if len(annotations) % 100 < args.batch:
            print(f"third_model_annotation_frames={len(annotations)}/{len(rows)}", flush=True)

    payload = {
        "schema": "blindassist_ustrf_third_model_person_adjudication_r1",
        "authority": "disagreement_only_annotation_proposal_never_candidate_or_promotion_credit",
        "config_sha256": sha256_file(args.config),
        "adjudication_bundle_sha256": sha256_file(bundle_path),
        "candidate_alerts_exposed": False,
        "baseline_app_detector_outputs_exposed": False,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "imgsz": int(adjudicator["imgsz"]),
        "confidence": float(adjudicator["confidence"]),
        "class_id": int(adjudicator["class_id"]),
        "frame_count": len(annotations),
        "frames": annotations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"frame_count": len(annotations), "sha256": sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
