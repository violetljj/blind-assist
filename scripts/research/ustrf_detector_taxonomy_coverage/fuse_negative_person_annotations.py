from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_target_truth_candidate import iou


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cross-iou", type=float, default=0.30)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    payload = json.loads(args.dual_candidate.read_text(encoding="utf-8"))
    for source in payload["sources"]:
        for window in source["windows"]:
            if window["window_type"] != "negative":
                continue
            for row in window["frames"]:
                primary = row["all_person_candidates"]
                secondary = row["yoloe_person_candidates"]
                fused: list[dict] = []
                used_secondary: set[int] = set()
                for first in primary:
                    ranked = sorted(
                        ((iou(first["bbox_xyxy"], other["bbox_xyxy"]), index, other) for index, other in enumerate(secondary)),
                        reverse=True,
                    )
                    overlap, index, other = ranked[0] if ranked else (0.0, -1, None)
                    corroborated = overlap >= args.cross_iou
                    if corroborated:
                        used_secondary.add(index)
                    confidence = float(first["proposal_confidence"])
                    fused.append({
                        "bbox_xyxy": first["bbox_xyxy"],
                        "proposal_confidence": round(confidence if corroborated else confidence * 0.25, 6),
                        "annotation_evidence": "yolo11x_and_yoloe" if corroborated else "yolo11x_only_downweighted",
                        "primary_confidence": first["proposal_confidence"],
                        "secondary_confidence": other["proposal_confidence"] if corroborated else None,
                        "cross_model_iou": round(overlap, 6),
                    })
                for index, second in enumerate(secondary):
                    if index in used_secondary:
                        continue
                    confidence = float(second["proposal_confidence"])
                    fused.append({
                        "bbox_xyxy": second["bbox_xyxy"],
                        "proposal_confidence": round(confidence * 0.25, 6),
                        "annotation_evidence": "yoloe_only_downweighted",
                        "primary_confidence": None,
                        "secondary_confidence": second["proposal_confidence"],
                        "cross_model_iou": 0.0,
                    })
                row["all_person_candidates"] = fused
                row["confirmed_absent_candidate"] = not fused
    payload["schema"] = "blindassist_ustrf_detector_target_truth_fused_annotation_candidate_v1"
    payload["authority"] = "candidate_only_requires_visual_review_before_truth_freeze"
    payload["negative_annotation_fusion"] = {
        "cross_model_iou": args.cross_iou,
        "cross_confirmed_seed_weight": 1.0,
        "single_model_seed_weight": 0.25,
        "rationale": "cross-model support anchors temporal annotation tracks; single-model proposals need four-times stronger confidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
